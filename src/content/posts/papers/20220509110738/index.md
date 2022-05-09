---
draft: false
title: "Survey on graph embeddings and their applications to machine learning problems on graphs"
date: 2022-05-09
author: "akitenkrad"
description: ""
tags: ["Round-1", "Survey", "Graph Neural Network", "Graph Embedding", "Node Classification", "Link Prediction", "Node Clustering", "Graph Visualization"]
menu:
  sidebar:
    name: 2022.05.09
    identifier: 20220509
    parent: papers
    weight: 10
math: true
---

- [x] Round-1: Overview
- [ ] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Makarov, I., Kiselev, D., Nikitinsky, N., & Subelj, L. (2021).  
**Survey on graph embeddings and their applications to machine learning problems on graphs.**  
PeerJ Computer Science, 7, 1–62.  
[Paper Link](https://doi.org/10.7717/PEERJ-CS.357/SUPP-1)
{{< /citation >}}

## Abstract

> Dealing with relational data always required significant computational resources,domain expertise and task-dependent feature engineering to incorporate structuralinformation into a predictive model. Nowadays, a family of automated graphfeature engineering techniques has been proposed in different streams of literature. So-called graph embeddings provide a powerful tool to construct vectorized featurespaces for graphs and their components, such as nodes, edges and subgraphsunder preserving inner graph properties. Using the constructed feature spaces, manymachine learning problems on graphs can be solved via standard frameworkssuitable for vectorized feature representation.  
> Our survey aims to describe the coreconcepts of graph embeddings and provide several taxonomies for their description. First, we start with the methodological approach and extract three types of graphembedding models based on matrix factorization, random-walks and deep learningapproaches. Next, we describe how different types of networks impact the abilityof models to incorporate structural and attributed data into a unified embedding. Going further, we perform a thorough evaluation of graph embedding applications tomachine learning problems on graphs, among which are node classification, linkprediction, clustering, visualization, compression, and a family of the whole graphembedding algorithms suitable for graph classification, similarity and alignmentproblems. Finally, we overview the existing applications of graph embeddings tocomputer science domains, formulate open problems and provide experimentresults, explaining how different networks properties result in graph embeddingsquality in the four classic machine learning problems on graphs, such as nodeclassification, link prediction, clustering and graph visualization.  
> As a result, oursurvey covers a new rapidly growingfield of network feature engineering, presents anin-depth analysis of models based on network types, and overviews a wide range ofapplications to machine learning problems on graphs.  

## What's New

- 既存レビューと本稿との違いについて ← INTRODUCTION

## Preliminaries

#### Definition of Graph

TBD

#### Definition of Graph Embedding

{{< box-with-title title="Definition" >}}

$$
f:V \rightarrow \mathbb{R}^d, d \ll |V|
$$

{{< /box-with-title >}}

#### Definition of First and Second Order Proximities

TBD

## Methods for Constructing Graph Embedding

#### Paper sources

##### Curated Lists

| Name | Link | Description |
|:-----|:-----|:------------|
| by Chen               | https://github.com/chihming/              | awesome-network-embedding |
| by Rozemberczki       | https://github.com/benedekrozemberczki/   | awesome-graph-classification |
| by Rebo               | https://github.com/MaxwellRebo/           | awesome-2vec |
| by Soru               | https://gist.github.com/mommi84/          | awesome-kge |

##### Conferences

| Name | Link | Description |
|:-----|:-----|:------------|
| Complex Networks      | https://complexnetworks.org/    | International Conference on Complex Networks and their Applications |
| The Web               | https://www2020.thewebconf.org/ | The Web Conference is international conference on the World Wide Web. |
| WSDM                  | http://www.wsdm-conference.org/ | Web-inspired research involving search and data mining |
| IJCAI                 | https://www.ijcai.org/          | International Joint Conferences on Artificial Intelligence |
| AAAI                  | https://www.aaai.org/           | Association for the Advancement of Artificial Intelligence |
| ICML                  | https://icml.cc/                | International Conference on Machine Learning |
| SIGKDD                | https://www.kdd.org/            | Special Interest Group in Knowledge Discovery and Databases |

##### Domain conferences

| Name | Link | Description |
|:-----|:-----|:------------|
| ACL                   | http://www.acl2019.org/         | Association for Computational Linguistics |
| CVPR                  | http://cvpr2019.thecvf.com/     | Conference on Computer Vision and Pattern Recognition |

##### Publishers

| Name | Link | Description |
|:-----|:-----|:------------|
| ACM DL                | https://dl.acm.org/             | Full-text articles database by Association for Computing Machinery |
| IEEE Xplore           | https://ieeexplore.ieee.org/Xplore/home.jsp | Research published by Institute of Electrical and Electronics Engineers | 
| Link Springer         | https://link.springer.com/      | Online collection of scientific journals, books and reference works |

##### Indexing services

| Name | Link | Description |
|:-----|:-----|:------------|
| Scopus                | https://www.scopus.com/         | Abstract and citation database |
| Web of Science        | https://www.webofknowledge.com/ | Citation Indexer |
| Scholar Google        | https://scholar.google.com/     | Web search engine for indexing full-text papers or its metadata

### Dimensionality reduction (matrix factorization) methods

## References


{{< ci-details summary="Recent developments in exponential random graph (p*) models for social networks (G. Robins et al., 2007)">}}

G. Robins, T. Snijders, Peng Wang, M. Handcock, P. Pattison. (2007)  
**Recent developments in exponential random graph (p*) models for social networks**  
Soc. Networks  
[Paper Link](https://www.semanticscholar.org/paper/00350a2b4adbaba0293ec10f73b759cfddde166e)  
Influential Citation Count (58), SS-ID (00350a2b4adbaba0293ec10f73b759cfddde166e)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="A Comprehensive Survey of Graph Embedding: Problems, Techniques, and Applications (Hongyun Cai et al., 2017)">}}

Hongyun Cai, V. Zheng, K. Chang. (2017)  
**A Comprehensive Survey of Graph Embedding: Problems, Techniques, and Applications**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/006906b6bbe5c1f378cde9fd86de1ce9e6b131da)  
Influential Citation Count (42), SS-ID (006906b6bbe5c1f378cde9fd86de1ce9e6b131da)  

**ABSTRACT**  
Graph is an important data representation which appears in a wide diversity of real-world scenarios. Effective graph analytics provides users a deeper understanding of what is behind the data, and thus can benefit a lot of useful applications such as node classification, node recommendation, link prediction, etc. However, most graph analytics methods suffer the high computation and space cost. Graph embedding is an effective yet efficient way to solve the graph analytics problem. It converts the graph data into a low dimensional space in which the graph structural information and graph properties are maximumly preserved. In this survey, we conduct a comprehensive review of the literature in graph embedding. We first introduce the formal definition of graph embedding as well as the related concepts. After that, we propose two taxonomies of graph embedding which correspond to what challenges exist in different graph embedding problem settings and how the existing work addresses these challenges in their solutions. Finally, we summarize the applications that graph embedding enables and suggest four promising future research directions in terms of computation efficiency, problem settings, techniques, and application scenarios.

{{< /ci-details >}}

{{< ci-details summary="mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding (Wentao Zhang et al., 2020)">}}

Wentao Zhang, Yuan Fang, Zemin Liu, Min Wu, Xinming Zhang. (2020)  
**mg2vec: Learning Relationship-Preserving Heterogeneous Graph Representations via Metagraph Embedding**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/00a33da57a6beef0abb5e315a2018433e8659429)  
Influential Citation Count (0), SS-ID (00a33da57a6beef0abb5e315a2018433e8659429)  

**ABSTRACT**  
Given that heterogeneous information networks (HIN) encompass nodes and edges belonging to different semantic types, they can model complex data in real-world scenarios. Thus, HIN embedding has received increasing attention, which aims to learn node representations in a low-dimensional space, in order to preserve the structural and semantic information on the HIN. In this regard, metagraphs, which model common and recurring patterns on HINs, emerge as a powerful tool to capture semantic-rich and often latent relationships on HINs. Although metagraphs have been employed to address several specific data mining tasks, they have not been thoroughly explored for the more general HIN embedding. In this paper, we leverage metagraphs to learn relationship-preserving HIN embedding in a self-supervised setting, to support various relationship mining tasks. In particular, we observe that most of the current approaches often under-utilize metagraphs, which are only applied in a pre-processing step and do not actively guide representation learning afterwards. Thus, we propose the novel framework of mg2vec, which learns the embeddings for metagraphs and nodes jointly. That is, metagraphs actively participates in the learning process by mapping themselves to the same embedding space as the nodes do. Moreover, metagraphs guide the learning through both first- and second-order constraints on node embeddings, to model not only latent relationships between a pair of nodes, but also individual preferences of each node. Finally, we conduct extensive experiments on three public datasets. Results show that mg2vec significantly outperforms a suite of state-of-the-art baselines in relationship mining tasks including relationship prediction, search and visualization.

{{< /ci-details >}}

{{< ci-details summary="Hierarchical structure and the prediction of missing links in networks (A. Clauset et al., 2008)">}}

A. Clauset, C. Moore, M. Newman. (2008)  
**Hierarchical structure and the prediction of missing links in networks**  
Nature  
[Paper Link](https://www.semanticscholar.org/paper/00b7ffd43e9b6b70c80449872a8c9ec49c7d045a)  
Influential Citation Count (83), SS-ID (00b7ffd43e9b6b70c80449872a8c9ec49c7d045a)  

**ABSTRACT**  
Networks have in recent years emerged as an invaluable tool for describing and quantifying complex systems in many branches of science. Recent studies suggest that networks often exhibit hierarchical organization, in which vertices divide into groups that further subdivide into groups of groups, and so forth over multiple scales. In many cases the groups are found to correspond to known functional units, such as ecological niches in food webs, modules in biochemical networks (protein interaction networks, metabolic networks or genetic regulatory networks) or communities in social networks. Here we present a general technique for inferring hierarchical structure from network data and show that the existence of hierarchy can simultaneously explain and quantitatively reproduce many commonly observed topological properties of networks, such as right-skewed degree distributions, high clustering coefficients and short path lengths. We further show that knowledge of hierarchical structure can be used to predict missing connections in partly known networks with high accuracy, and for more general network structures than competing techniques. Taken together, our results suggest that hierarchy is a central organizing principle of complex networks, capable of offering insight into many network phenomena.

{{< /ci-details >}}

{{< ci-details summary="Deep Graph Kernels (Pinar Yanardag et al., 2015)">}}

Pinar Yanardag, S. Vishwanathan. (2015)  
**Deep Graph Kernels**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/00d736c540f80582279093cfc5ffe454a3226da9)  
Influential Citation Count (160), SS-ID (00d736c540f80582279093cfc5ffe454a3226da9)  

**ABSTRACT**  
In this paper, we present Deep Graph Kernels, a unified framework to learn latent representations of sub-structures for graphs, inspired by latest advancements in language modeling and deep learning. Our framework leverages the dependency information between sub-structures by learning their latent representations. We demonstrate instances of our framework on three popular graph kernels, namely Graphlet kernels, Weisfeiler-Lehman subtree kernels, and Shortest-Path graph kernels. Our experiments on several benchmark datasets show that Deep Graph Kernels achieve significant improvements in classification accuracy over state-of-the-art graph kernels.

{{< /ci-details >}}

{{< ci-details summary="Graph Embedding With Data Uncertainty (Firas Laakom et al., 2020)">}}

Firas Laakom, Jenni Raitoharju, N. Passalis, Alexandros Iosifidis, M. Gabbouj. (2020)  
**Graph Embedding With Data Uncertainty**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/021e3e3f63194c5508e71706122372e96fdb1cdd)  
Influential Citation Count (0), SS-ID (021e3e3f63194c5508e71706122372e96fdb1cdd)  

**ABSTRACT**  
Spectral-based subspace learning is a common data preprocessing step in many machine learning pipelines. The main aim is to learn a meaningful low dimensional embedding of the data. However, most subspace learning methods do not take into consideration possible measurement inaccuracies or artifacts that can lead to data with high uncertainty. Thus, learning directly from raw data can be misleading and can negatively impact the accuracy. In this paper, we propose to model artifacts in training data using probability distributions; each data point is represented by a Gaussian distribution centered at the original data point and having a variance modeling its uncertainty. We reformulate the Graph Embedding framework to make it suitable for learning from distributions and we study as special cases the Linear Discriminant Analysis and the Marginal Fisher Analysis techniques. Furthermore, we propose two schemes for modeling data uncertainty based on pair-wise distances in an unsupervised and a supervised contexts.

{{< /ci-details >}}

{{< ci-details summary="Learning to Represent Knowledge Graphs with Gaussian Embedding (Shizhu He et al., 2015)">}}

Shizhu He, Kang Liu, Guoliang Ji, Jun Zhao. (2015)  
**Learning to Represent Knowledge Graphs with Gaussian Embedding**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/02e2059c328bd9fad4e676266435199663bed804)  
Influential Citation Count (30), SS-ID (02e2059c328bd9fad4e676266435199663bed804)  

**ABSTRACT**  
The representation of a knowledge graph (KG) in a latent space recently has attracted more and more attention. To this end, some proposed models (e.g., TransE) embed entities and relations of a KG into a "point" vector space by optimizing a global loss function which ensures the scores of positive triplets are higher than negative ones. We notice that these models always regard all entities and relations in a same manner and ignore their (un)certainties. In fact, different entities and relations may contain different certainties, which makes identical certainty insufficient for modeling. Therefore, this paper switches to density-based embedding and propose KG2E for explicitly modeling the certainty of entities and relations, which learn the representations of KGs in the space of multi-dimensional Gaussian distributions. Each entity/relation is represented by a Gaussian distribution, where the mean denotes its position and the covariance (currently with diagonal covariance) can properly represent its certainty. In addition, compared with the symmetric measures used in point-based methods, we employ the KL-divergence for scoring triplets, which is a natural asymmetry function for effectively modeling multiple types of relations. We have conducted extensive experiments on link prediction and triplet classification with multiple benchmark datasets (WordNet and Freebase). Our experimental results demonstrate that our method can effectively model the (un)certainties of entities and relations in a KG, and it significantly outperforms state-of-the-art methods (including TransH and TransR).

{{< /ci-details >}}

{{< ci-details summary="Inductive matrix completion for predicting gene–disease associations (Nagarajan Natarajan et al., 2014)">}}

Nagarajan Natarajan, I. Dhillon. (2014)  
**Inductive matrix completion for predicting gene–disease associations**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/02e34d24ebab52ee516c2ab50d93d360bde68187)  
Influential Citation Count (30), SS-ID (02e34d24ebab52ee516c2ab50d93d360bde68187)  

**ABSTRACT**  
Motivation: Most existing methods for predicting causal disease genes rely on specific type of evidence, and are therefore limited in terms of applicability. More often than not, the type of evidence available for diseases varies—for example, we may know linked genes, keywords associated with the disease obtained by mining text, or co-occurrence of disease symptoms in patients. Similarly, the type of evidence available for genes varies—for example, specific microarray probes convey information only for certain sets of genes. In this article, we apply a novel matrix-completion method called Inductive Matrix Completion to the problem of predicting gene-disease associations; it combines multiple types of evidence (features) for diseases and genes to learn latent factors that explain the observed gene–disease associations. We construct features from different biological sources such as microarray expression data and disease-related textual data. A crucial advantage of the method is that it is inductive; it can be applied to diseases not seen at training time, unlike traditional matrix-completion approaches and network-based inference methods that are transductive. Results: Comparison with state-of-the-art methods on diseases from the Online Mendelian Inheritance in Man (OMIM) database shows that the proposed approach is substantially better—it has close to one-in-four chance of recovering a true association in the top 100 predictions, compared to the recently proposed Catapult method (second best) that has <15% chance. We demonstrate that the inductive method is particularly effective for a query disease with no previously known gene associations, and for predicting novel genes, i.e. genes that are previously not linked to diseases. Thus the method is capable of predicting novel genes even for well-characterized diseases. We also validate the novelty of predictions by evaluating the method on recently reported OMIM associations and on associations recently reported in the literature. Availability: Source code and datasets can be downloaded from http://bigdata.ices.utexas.edu/project/gene-disease. Contact: naga86@cs.utexas.edu

{{< /ci-details >}}

{{< ci-details summary="Learning latent representations of nodes for classifying in heterogeneous social networks (Yann Jacob et al., 2014)">}}

Yann Jacob, Ludovic Denoyer, P. Gallinari. (2014)  
**Learning latent representations of nodes for classifying in heterogeneous social networks**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/030d436cb0465fd6cec0d5140b2534a8f1b8aeca)  
Influential Citation Count (6), SS-ID (030d436cb0465fd6cec0d5140b2534a8f1b8aeca)  

**ABSTRACT**  
Social networks are heterogeneous systems composed of different types of nodes (e.g. users, content, groups, etc.) and relations (e.g. social or similarity relations). While learning and performing inference on homogeneous networks have motivated a large amount of research, few work exists on heterogeneous networks and there are open and challenging issues for existing methods that were previously developed for homogeneous networks. We address here the specific problem of nodes classification and tagging in heterogeneous social networks, where different types of nodes are considered, each type with its own label or tag set. We propose a new method for learning node representations onto a latent space, common to all the different node types. Inference is then performed in this latent space. In this framework, two nodes connected in the network will tend to share similar representations regardless of their types. This allows bypassing limitations of the methods based on direct extensions of homogenous frameworks and exploiting the dependencies and correlations between the different node types. The proposed method is tested on two representative datasets and compared to state-of-the-art methods and to baselines.

{{< /ci-details >}}

{{< ci-details summary="A Review of Relational Machine Learning for Knowledge Graphs (Maximilian Nickel et al., 2015)">}}

Maximilian Nickel, K. Murphy, Volker Tresp, E. Gabrilovich. (2015)  
**A Review of Relational Machine Learning for Knowledge Graphs**  
Proceedings of the IEEE  
[Paper Link](https://www.semanticscholar.org/paper/033f25ad905ef2ed32a8331cf38b83953ff15922)  
Influential Citation Count (111), SS-ID (033f25ad905ef2ed32a8331cf38b83953ff15922)  

**ABSTRACT**  
Relational machine learning studies methods for the statistical analysis of relational, or graph-structured, data. In this paper, we provide a review of how such statistical models can be “trained” on large knowledge graphs, and then used to predict new facts about the world (which is equivalent to predicting new edges in the graph). In particular, we discuss two fundamentally different kinds of statistical relational models, both of which can scale to massive data sets. The first is based on latent feature models such as tensor factorization and multiway neural networks. The second is based on mining observable patterns in the graph. We also show how to combine these latent and observable models to get improved modeling power at decreased computational cost. Finally, we discuss how such statistical models of graphs can be combined with text-based information extraction methods for automatically constructing knowledge graphs from the Web. To this end, we also discuss Google's knowledge vault project as an example of such combination.

{{< /ci-details >}}

{{< ci-details summary="DistDGL: Distributed Graph Neural Network Training for Billion-Scale Graphs (Da Zheng et al., 2020)">}}

Da Zheng, Chao Ma, Minjie Wang, Jinjing Zhou, Qidong Su, Xiang Song, Quan Gan, Zheng Zhang, G. Karypis. (2020)  
**DistDGL: Distributed Graph Neural Network Training for Billion-Scale Graphs**  
2020 IEEE/ACM 10th Workshop on Irregular Applications: Architectures and Algorithms (IA3)  
[Paper Link](https://www.semanticscholar.org/paper/037df1500b9b8d4a57455b7ad205f86cc94a0b13)  
Influential Citation Count (9), SS-ID (037df1500b9b8d4a57455b7ad205f86cc94a0b13)  

**ABSTRACT**  
Graph neural networks (GNN) have shown great success in learning from graph-structured data. They are widely used in various applications, such as recommendation, fraud detection, and search. In these domains, the graphs are typically large, containing hundreds of millions of nodes and several billions of edges. To tackle this challenge, we develop DistDGL, a system for training GNNs in a mini-batch fashion on a cluster of machines. DistDGL is based on the Deep Graph Library (DGL), a popular GNN development framework. DistDGL distributes the graph and its associated data (initial features and embeddings) across the machines and uses this distribution to derive a computational decomposition by following an owner-compute rule. DistDGL follows a synchronous training approach and allows ego-networks forming the mini-batches to include non-local nodes. To minimize the overheads associated with distributed computations, DistDGL uses a high-quality and light-weight min-cut graph partitioning algorithm along with multiple balancing constraints. This allows it to reduce communication overheads and statically balance the computations. It further reduces the communication by replicating halo nodes and by using sparse embedding updates. The combination of these design choices allows DistDGL to train high-quality models while achieving high parallel efficiency and memory scalability. We demonstrate our optimizations on both inductive and transductive GNN models. Our results show that DistDGL achieves linear speedup without compromising model accuracy and requires only 13 seconds to complete a training epoch for a graph with 100 million nodes and 3 billion edges on a cluster with 16 machines.

{{< /ci-details >}}

{{< ci-details summary="Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems (Qitian Wu et al., 2019)">}}

Qitian Wu, Hengrui Zhang, Xiaofeng Gao, Peng He, Paul Weng, Han Gao, Guihai Chen. (2019)  
**Dual Graph Attention Networks for Deep Latent Representation of Multifaceted Social Effects in Recommender Systems**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/03ed44b85886a7a95d1533fb1d1a142e60ae292c)  
Influential Citation Count (20), SS-ID (03ed44b85886a7a95d1533fb1d1a142e60ae292c)  

**ABSTRACT**  
Social recommendation leverages social information to solve data sparsity and cold-start problems in traditional collaborative filtering methods. However, most existing models assume that social effects from friend users are static and under the forms of constant weights or fixed constraints. To relax this strong assumption, in this paper, we propose dual graph attention networks to collaboratively learn representations for two-fold social effects, where one is modeled by a user-specific attention weight and the other is modeled by a dynamic and context-aware attention weight. We also extend the social effects in user domain to item domain, so that information from related items can be leveraged to further alleviate the data sparsity problem. Furthermore, considering that different social effects in two domains could interact with each other and jointly influence users' preferences for items, we propose a new policy-based fusion strategy based on contextual multi-armed bandit to weigh interactions of various social effects. Experiments on one benchmark dataset and a commercial dataset verify the efficacy of the key components in our model. The results show that our model achieves great improvement for recommendation accuracy compared with other state-of-the-art social recommendation methods.

{{< /ci-details >}}

{{< ci-details summary="Principal component analysis (S. Wold et al., 1987)">}}

S. Wold, K. Esbensen, P. Geladi. (1987)  
**Principal component analysis**  
  
[Paper Link](https://www.semanticscholar.org/paper/040777d8e65e94a525a4e1cb778bb2a747ae8cb8)  
Influential Citation Count (338), SS-ID (040777d8e65e94a525a4e1cb778bb2a747ae8cb8)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Modeling polypharmacy side effects with graph convolutional networks (M. Zitnik et al., 2018)">}}

M. Zitnik, Monica Agrawal, J. Leskovec. (2018)  
**Modeling polypharmacy side effects with graph convolutional networks**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/046c4276b72e21731150c0655519ec717d8f5bad)  
Influential Citation Count (20), SS-ID (046c4276b72e21731150c0655519ec717d8f5bad)  

**ABSTRACT**  
Motivation The use of drug combinations, termed polypharmacy, is common to treat patients with complex diseases or co‐existing conditions. However, a major consequence of polypharmacy is a much higher risk of adverse side effects for the patient. Polypharmacy side effects emerge because of drug‐drug interactions, in which activity of one drug may change, favorably or unfavorably, if taken with another drug. The knowledge of drug interactions is often limited because these complex relationships are rare, and are usually not observed in relatively small clinical testing. Discovering polypharmacy side effects thus remains an important challenge with significant implications for patient mortality and morbidity. Results Here, we present Decagon, an approach for modeling polypharmacy side effects. The approach constructs a multimodal graph of protein‐protein interactions, drug‐protein target interactions and the polypharmacy side effects, which are represented as drug‐drug interactions, where each side effect is an edge of a different type. Decagon is developed specifically to handle such multimodal graphs with a large number of edge types. Our approach develops a new graph convolutional neural network for multirelational link prediction in multimodal networks. Unlike approaches limited to predicting simple drug‐drug interaction values, Decagon can predict the exact side effect, if any, through which a given drug combination manifests clinically. Decagon accurately predicts polypharmacy side effects, outperforming baselines by up to 69%. We find that it automatically learns representations of side effects indicative of co‐occurrence of polypharmacy in patients. Furthermore, Decagon models particularly well polypharmacy side effects that have a strong molecular basis, while on predominantly non‐molecular side effects, it achieves good performance because of effective sharing of model parameters across edge types. Decagon opens up opportunities to use large pharmacogenomic and patient population data to flag and prioritize polypharmacy side effects for follow‐up analysis via formal pharmacological studies. Availability and implementation Source code and preprocessed datasets are at: http://snap.stanford.edu/decagon.

{{< /ci-details >}}

{{< ci-details summary="Probabilistic Entity-Relationship Models, PRMs, and Plate Models (D. Heckerman et al., 2004)">}}

D. Heckerman, Christopher Meek, D. Koller. (2004)  
**Probabilistic Entity-Relationship Models, PRMs, and Plate Models**  
  
[Paper Link](https://www.semanticscholar.org/paper/04757f50d0021c8351237fad2f4002e59d5d8430)  
Influential Citation Count (13), SS-ID (04757f50d0021c8351237fad2f4002e59d5d8430)  

**ABSTRACT**  
We introduce a graphical language for re- lational data called the probabilistic entity- relationship (PER) model. The model is an extension of the entity-relationship model, a common model for the abstract repre- sentation of database structure. We con- centrate on the directed version of this model—the directed acyclic probabilistic entity-relationship (DAPER) model. The DAPER model is closely related to the plate model and the probabilistic relational model (PRM), existing models for relational data. The DAPER model is more expressive than either existing model, and also helps to demonstrate their similarity. dinary graphical models (e.g., directed-acyclic graphs and undirected graphs) are to flat data. In this paper, we introduce a new graphical model for relational data—the probabilistic entity-relationship (PER) model. This model class is more expressive than either PRMs or plate models. We concentrate on a particular type of PER model—the directed acyclic probabilistic entity-relationship (DAPER) model—in which all probabilistic arcs are directed. It is this ver- sion of PER model that is most similar to the plate model and PRM. We define new versions of the plate model and PRM such their expressiveness is equivalent to the DAPER model, and then (in the expanded tech report, Heckerman, Meek, and Koller, 2004) compare the new and old definitions. Consequently, we both demonstrate the similarity among the original lan- guages as well as enhance their abilities to express con- ditional independence in relational data. Our hope is that this demonstration of similarity will foster greater communication and collaboration among statisticians who mostly use plate models and computer scientists who mostly use PRMs. We in fact began this work with an effort to unify traditional PRMs and plate models. In the process, we discovered that it was important to make both entities and relationships (concepts discussed in de- tail in the next section) first class objects in the lan- guage. We in turn discovered an existing language that does this—the entity-relationship (ER) model—a commonly used model for the abstract representation of database structure. We then extended this language to handle probabilistic relationships, creating the PER model. We should emphasize that the languages we discuss are neither meant to serve as a database schema nor meant to be built on top of one. In practice, database schemas are built up over a long period of time as the needs of the database consumers change. Conse-

{{< /ci-details >}}

{{< ci-details summary="Deep Reinforcement Learning for Electric Vehicle Routing Problem with Time Windows (Bo Lin et al., 2020)">}}

Bo Lin, Bissan Ghaddar, J. Nathwani. (2020)  
**Deep Reinforcement Learning for Electric Vehicle Routing Problem with Time Windows**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/05b919f0ff46a9c3828d0899275a003bdcb1039d)  
Influential Citation Count (1), SS-ID (05b919f0ff46a9c3828d0899275a003bdcb1039d)  

**ABSTRACT**  
The past decade has seen a rapid penetration of electric vehicles (EV) in the market, more and more logistics and transportation companies start to deploy EVs for service provision. In order to model the operations of a commercial EV fleet, we utilize the EV routing problem with time windows (EVRPTW). In this research, we propose an end-to-end deep reinforcement learning framework to solve the EVRPTW. In particular, we develop an attention model incorporating the pointer network and a graph embedding technique to parameterize a stochastic policy for solving the EVRPTW. The model is then trained using policy gradient with rollout baseline. Our numerical studies show that the proposed model is able to efficiently solve EVRPTW instances of large sizes that are not solvable with any existing approaches.

{{< /ci-details >}}

{{< ci-details summary="Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks (Wei-Lin Chiang et al., 2019)">}}

Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh. (2019)  
**Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/05c4eb154ad9512a69569c18d68bc4428ee8bb83)  
Influential Citation Count (65), SS-ID (05c4eb154ad9512a69569c18d68bc4428ee8bb83)  

**ABSTRACT**  
Graph convolutional network (GCN) has been successfully applied to many graph-based applications; however, training a large-scale GCN remains challenging. Current SGD-based algorithms suffer from either a high computational cost that exponentially grows with number of GCN layers, or a large space requirement for keeping the entire graph and the embedding of each node in memory. In this paper, we propose Cluster-GCN, a novel GCN algorithm that is suitable for SGD-based training by exploiting the graph clustering structure. Cluster-GCN works as the following: at each step, it samples a block of nodes that associate with a dense subgraph identified by a graph clustering algorithm, and restricts the neighborhood search within this subgraph. This simple but effective strategy leads to significantly improved memory and computational efficiency while being able to achieve comparable test accuracy with previous algorithms. To test the scalability of our algorithm, we create a new Amazon2M data with 2 million nodes and 61 million edges which is more than 5 times larger than the previous largest publicly available dataset (Reddit). For training a 3-layer GCN on this data, Cluster-GCN is faster than the previous state-of-the-art VR-GCN (1523 seconds vs 1961 seconds) and using much less memory (2.2GB vs 11.2GB). Furthermore, for training 4 layer GCN on this data, our algorithm can finish in around 36 minutes while all the existing GCN training algorithms fail to train due to the out-of-memory issue. Furthermore, Cluster-GCN allows us to train much deeper GCN without much time and memory overhead, which leads to improved prediction accuracy---using a 5-layer Cluster-GCN, we achieve state-of-the-art test F1 score 99.36 on the PPI dataset, while the previous best result was 98.71 by~\citezhang2018gaan.

{{< /ci-details >}}

{{< ci-details summary="Community-Based Question Answering via Heterogeneous Social Network Learning (Hanyin Fang et al., 2016)">}}

Hanyin Fang, Fei Wu, Zhou Zhao, Xinyu Duan, Yueting Zhuang, Martin Ester. (2016)  
**Community-Based Question Answering via Heterogeneous Social Network Learning**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/06b0e8c067e249a2e71897c00730ec734d18c922)  
Influential Citation Count (9), SS-ID (06b0e8c067e249a2e71897c00730ec734d18c922)  

**ABSTRACT**  
Community-based question answering (cQA) sites have accumulated vast amount of questions and corresponding crowdsourced answers over time. How to efficiently share the underlying information and knowledge from reliable (usually highly-reputable) answerers has become an increasingly popular research topic. A major challenge in cQA tasks is the accurate matching of high-quality answers w.r.t given questions. Many of traditional approaches likely recommend corresponding answers merely depending on the content similarity between questions and answers, therefore suffer from the sparsity bottleneck of cQA data. In this paper, we propose a novel framework which encodes not only the contents of question-answer(Q-A) but also the social interaction cues in the community to boost the cQA tasks. More specifically, our framework collaboratively utilizes the rich interaction among questions, answers and answerers to learn the relative quality rank of different answers w.r.t a same question. Moreover, the information in heterogeneous social networks is comprehensively employed to enhance the quality of question-answering (QA) matching by our deep random walk learning framework. Extensive experiments on a large-scale dataset from a real world cQA site show that leveraging the heterogeneous social information indeed achieves better performance than other state-of-the-art cQA methods.

{{< /ci-details >}}

{{< ci-details summary="An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition (Chenyang Si et al., 2019)">}}

Chenyang Si, Wentao Chen, Wei Wang, Liang Wang, T. Tan. (2019)  
**An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/074611d0c9f527bc0ad06f00df779f3361e38b83)  
Influential Citation Count (29), SS-ID (074611d0c9f527bc0ad06f00df779f3361e38b83)  

**ABSTRACT**  
Skeleton-based action recognition is an important task that requires the adequate understanding of movement characteristics of a human action from the given skeleton sequence. Recent studies have shown that exploring spatial and temporal features of the skeleton sequence is vital for this task. Nevertheless, how to effectively extract discriminative spatial and temporal features is still a challenging problem. In this paper, we propose a novel Attention Enhanced Graph Convolutional LSTM Network (AGC-LSTM) for human action recognition from skeleton data. The proposed AGC-LSTM can not only capture discriminative features in spatial configuration and temporal dynamics but also explore the co-occurrence relationship between spatial and temporal domains. We also present a temporal hierarchical architecture to increase temporal receptive fields of the top AGC-LSTM layer, which boosts the ability to learn the high-level semantic representation and significantly reduces the computation cost. Furthermore, to select discriminative spatial information, the attention mechanism is employed to enhance information of key joints in each AGC-LSTM layer. Experimental results on two datasets are provided: NTU RGB+D dataset and Northwestern-UCLA dataset. The comparison results demonstrate the effectiveness of our approach and show that our approach outperforms the state-of-the-art methods on both datasets.

{{< /ci-details >}}

{{< ci-details summary="Asymmetric Transitivity Preserving Graph Embedding (Mingdong Ou et al., 2016)">}}

Mingdong Ou, Peng Cui, J. Pei, Ziwei Zhang, Wenwu Zhu. (2016)  
**Asymmetric Transitivity Preserving Graph Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  
Influential Citation Count (111), SS-ID (07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  

**ABSTRACT**  
Graph embedding algorithms embed a graph into a vector space where the structure and the inherent properties of the graph are preserved. The existing graph embedding methods cannot preserve the asymmetric transitivity well, which is a critical property of directed graphs. Asymmetric transitivity depicts the correlation among directed edges, that is, if there is a directed path from u to v, then there is likely a directed edge from u to v. Asymmetric transitivity can help in capturing structures of graphs and recovering from partially observed graphs. To tackle this challenge, we propose the idea of preserving asymmetric transitivity by approximating high-order proximity which are based on asymmetric transitivity. In particular, we develop a novel graph embedding algorithm, High-Order Proximity preserved Embedding (HOPE for short), which is scalable to preserve high-order proximities of large scale graphs and capable of capturing the asymmetric transitivity. More specifically, we first derive a general formulation that cover multiple popular high-order proximity measurements, then propose a scalable embedding algorithm to approximate the high-order proximity measurements based on their general formulation. Moreover, we provide a theoretical upper bound on the RMSE (Root Mean Squared Error) of the approximation. Our empirical experiments on a synthetic dataset and three real-world datasets demonstrate that HOPE can approximate the high-order proximities significantly better than the state-of-art algorithms and outperform the state-of-art algorithms in tasks of reconstruction, link prediction and vertex recommendation.

{{< /ci-details >}}

{{< ci-details summary="LINE: Large-scale Information Network Embedding (Jian Tang et al., 2015)">}}

Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Q. Mei. (2015)  
**LINE: Large-scale Information Network Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
Influential Citation Count (832), SS-ID (0834e74304b547c9354b6d7da6fa78ef47a48fa8)  

**ABSTRACT**  
This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for real world information networks which usually contain millions of nodes. In this paper, we propose a novel network embedding method called the ``LINE,'' which is suitable for arbitrary types of information networks: undirected, directed, and/or weighted. The method optimizes a carefully designed objective function that preserves both the local and global network structures. An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the effectiveness and the efficiency of the inference. Empirical experiments prove the effectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very efficient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online\footnote{\url{https://github.com/tangjianpku/LINE}}.

{{< /ci-details >}}

{{< ci-details summary="Data Poisoning Attack against Unsupervised Node Embedding Methods (Mingjie Sun et al., 2018)">}}

Mingjie Sun, Jian Tang, Huichen Li, Bo Li, Chaowei Xiao, Yao Chen, D. Song. (2018)  
**Data Poisoning Attack against Unsupervised Node Embedding Methods**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/088569b34d7f58bf4525ade7f391de98d62d601a)  
Influential Citation Count (8), SS-ID (088569b34d7f58bf4525ade7f391de98d62d601a)  

**ABSTRACT**  
Unsupervised node embedding methods (e.g., DeepWalk, LINE, and node2vec) have attracted growing interests given their simplicity and effectiveness. However, although these methods have been proved effective in a variety of applications, none of the existing work has analyzed the robustness of them. This could be very risky if these methods are attacked by an adversarial party. In this paper, we take the task of link prediction as an example, which is one of the most fundamental problems for graph analysis, and introduce a data positioning attack to node embedding methods. We give a complete characterization of attacker's utilities and present efficient solutions to adversarial attacks for two popular node embedding methods: DeepWalk and LINE. We evaluate our proposed attack model on multiple real-world graphs. Experimental results show that our proposed model can significantly affect the results of link prediction by slightly changing the graph structures (e.g., adding or removing a few edges). We also show that our proposed model is very general and can be transferable across different embedding methods. Finally, we conduct a case study on a coauthor network to better understand our attack method.

{{< /ci-details >}}

{{< ci-details summary="Edge-Labeling Graph Neural Network for Few-Shot Learning (Jongmin Kim et al., 2019)">}}

Jongmin Kim, Taesup Kim, Sungwoong Kim, C. Yoo. (2019)  
**Edge-Labeling Graph Neural Network for Few-Shot Learning**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/098b138f58e43338248e3bc35cb36adfed8008d1)  
Influential Citation Count (29), SS-ID (098b138f58e43338248e3bc35cb36adfed8008d1)  

**ABSTRACT**  
In this paper, we propose a novel edge-labeling graph neural network (EGNN), which adapts a deep neural network on the edge-labeling graph, for few-shot learning. The previous graph neural network (GNN) approaches in few-shot learning have been based on the node-labeling framework, which implicitly models the intra-cluster similarity and the inter-cluster dissimilarity. In contrast, the proposed EGNN learns to predict the edge-labels rather than the node-labels on the graph that enables the evolution of an explicit clustering by iteratively updating the edge-labels with direct exploitation of both intra-cluster similarity and the inter-cluster dissimilarity. It is also well suited for performing on various numbers of classes without retraining, and can be easily extended to perform a transductive inference. The parameters of the EGNN are learned by episodic training with an edge-labeling loss to obtain a well-generalizable model for unseen low-data problem. On both of the supervised and semi-supervised few-shot image classification tasks with two benchmark datasets, the proposed EGNN significantly improves the performances over the existing GNNs.

{{< /ci-details >}}

{{< ci-details summary="A measure of betweenness centrality based on random walks (M. Newman, 2003)">}}

M. Newman. (2003)  
**A measure of betweenness centrality based on random walks**  
Soc. Networks  
[Paper Link](https://www.semanticscholar.org/paper/0a575498f9e6bc0cc43b977c6e952101f89be90c)  
Influential Citation Count (123), SS-ID (0a575498f9e6bc0cc43b977c6e952101f89be90c)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Bootstrap Methods: Another Look at the Jackknife (D. Hinkley, 2008)">}}

D. Hinkley. (2008)  
**Bootstrap Methods: Another Look at the Jackknife**  
  
[Paper Link](https://www.semanticscholar.org/paper/0ae3682872d58216559f7d69de1537a1b15a9592)  
Influential Citation Count (704), SS-ID (0ae3682872d58216559f7d69de1537a1b15a9592)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="GNN Explainer: A Tool for Post-hoc Explanation of Graph Neural Networks (Rex Ying et al., 2019)">}}

Rex Ying, Dylan Bourgeois, Jiaxuan You, M. Zitnik, J. Leskovec. (2019)  
**GNN Explainer: A Tool for Post-hoc Explanation of Graph Neural Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/0b00413174f2474d72cfc5c3e248b958ddcc8ee1)  
Influential Citation Count (13), SS-ID (0b00413174f2474d72cfc5c3e248b958ddcc8ee1)  

**ABSTRACT**  
Graph Neural Networks (GNNs) are a powerful tool for machine learning on graphs. GNNs combine node feature information with the graph structure by using neural networks to pass messages through edges in the graph. However, incorporating both graph structure and feature information leads to complex non-linear models and explaining predictions made by GNNs remains to be a challenging task. Here we propose GnnExplainer, a general model-agnostic approach for providing interpretable explanations for predictions of any GNN-based model on any graph-based machine learning task (node and graph classification, link prediction). In order to explain a given node's predicted label, GnnExplainer provides a local interpretation by highlighting relevant features as well as an important subgraph structure by identifying the edges that are most relevant to the prediction. Additionally, the model provides single-instance explanations when given a single prediction as well as multi-instance explanations that aim to explain predictions for an entire class of instances/nodes. We formalize GnnExplainer as an optimization task that maximizes the mutual information between the prediction of the full model and the prediction of simplified explainer model. We experiment on synthetic as well as real-world data. On synthetic data we demonstrate that our approach is able to highlight relevant topological structures from noisy graphs. We also demonstrate GnnExplainer to provide a better understanding of pre-trained models on real-world tasks. GnnExplainer provides a variety of benefits, from the identification of semantically relevant structures to explain predictions to providing guidance when debugging faulty graph neural network models.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Directed Graph Embedding (Shijie Zhu et al., 2020)">}}

Shijie Zhu, Jianxin Li, Hao Peng, Senzhang Wang, Philip S. Yu, Lifang He. (2020)  
**Adversarial Directed Graph Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/0d4677ce389c1c1ef02c7d4e206d902c4f51bdb3)  
Influential Citation Count (0), SS-ID (0d4677ce389c1c1ef02c7d4e206d902c4f51bdb3)  

**ABSTRACT**  
Node representation learning for directed graphs is critically important to facilitate many graph mining tasks. To capture the directed edges between nodes, existing methods mostly learn two embedding vectors for each node, source vector and target vector. However, these methods learn the source and target vectors separately. For the node with very low indegree or outdegree, the corresponding target vector or source vector cannot be effectively learned. In this paper, we propose a novel Directed Graph embedding framework based on Generative Adversarial Network, called DGGAN. The main idea is to use adversarial mechanisms to deploy a discriminator and two generators that jointly learn each node's source and target vectors. For a given node, the two generators are trained to generate its fake target and source neighbor nodes from the same underlying distribution, and the discriminator aims to distinguish whether a neighbor node is real or fake. The two generators are formulated into a unified framework and could mutually reinforce each other to learn more robust source and target vectors. Extensive experiments show that DGGAN consistently and significantly outperforms existing state-of-the-art methods across multiple graph mining tasks on directed graphs.

{{< /ci-details >}}

{{< ci-details summary="Transition-based Knowledge Graph Embedding with Relational Mapping Properties (M. Fan et al., 2014)">}}

M. Fan, Qiang Zhou, E. Chang, T. Zheng. (2014)  
**Transition-based Knowledge Graph Embedding with Relational Mapping Properties**  
PACLIC  
[Paper Link](https://www.semanticscholar.org/paper/0dddf37145689e5f2899f8081d9971882e6ff1e9)  
Influential Citation Count (7), SS-ID (0dddf37145689e5f2899f8081d9971882e6ff1e9)  

**ABSTRACT**  
Many knowledge repositories nowadays contain billions of triplets, i.e. (head-entity, relationship, tail-entity), as relation instances. These triplets form a directed graph with entities as nodes and relationships as edges. However, this kind of symbolic and discrete storage structure makes it difficult for us to exploit the knowledge to enhance other intelligenceacquired applications (e.g. the QuestionAnswering System), as many AI-related algorithms prefer conducting computation on continuous data. Therefore, a series of emerging approaches have been proposed to facilitate knowledge computing via encoding the knowledge graph into a low-dimensional embedding space. TransE is the latest and most promising approach among them, and can achieve a higher performance with fewer parameters by modeling the relationship as a transitional vector from the head entity to the tail entity. Unfortunately, it is not flexible enough to tackle well with the various mapping properties of triplets, even though its authors spot the harm on performance. In this paper, we thus propose a superior model called TransM to leverage the structure of the knowledge graph via pre-calculating the distinct weight for each training triplet according to its relational mapping property. In this way, the optimal function deals with each triplet depending on its own weight. We carry out extensive experiments to compare TransM with the state-of-the-art method TransE and other prior arts. The performance of each approach is evaluated within two different application scenarios on several benchmark datasets. Results show that the model we proposed significantly outperforms the former ones with lower parameter complexity as TransE.

{{< /ci-details >}}

{{< ci-details summary="Geometric Deep Learning: Going beyond Euclidean data (M. Bronstein et al., 2016)">}}

M. Bronstein, Joan Bruna, Yann LeCun, Arthur D. Szlam, P. Vandergheynst. (2016)  
**Geometric Deep Learning: Going beyond Euclidean data**  
IEEE Signal Processing Magazine  
[Paper Link](https://www.semanticscholar.org/paper/0e779fd59353a7f1f5b559b9d65fa4bfe367890c)  
Influential Citation Count (113), SS-ID (0e779fd59353a7f1f5b559b9d65fa4bfe367890c)  

**ABSTRACT**  
Many scientific fields study data with an underlying structure that is non-Euclidean. Some examples include social networks in computational social sciences, sensor networks in communications, functional networks in brain imaging, regulatory networks in genetics, and meshed surfaces in computer graphics. In many applications, such geometric data are large and complex (in the case of social networks, on the scale of billions) and are natural targets for machine-learning techniques. In particular, we would like to use deep neural networks, which have recently proven to be powerful tools for a broad range of problems from computer vision, natural-language processing, and audio analysis. However, these tools have been most successful on data with an underlying Euclidean or grid-like structure and in cases where the invariances of these structures are built into networks used to model them.

{{< /ci-details >}}

{{< ci-details summary="Graph Kernels (S. Vishwanathan et al., 2008)">}}

S. Vishwanathan, N. Schraudolph, R. Kondor, K. Borgwardt. (2008)  
**Graph Kernels**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/0ed97826dec2ae59a10da5dd5b9bae8e0164b624)  
Influential Citation Count (43), SS-ID (0ed97826dec2ae59a10da5dd5b9bae8e0164b624)  

**ABSTRACT**  
We present a unified framework to study graph kernels, special cases of which include the random walk (Gartner et al., 2003; Borgwardt et al., 2005) and marginalized (Kashima et al., 2003, 2004; Mahet al., 2004) graph kernels. Through reduction to a Sylvester equation we improve the time complexity of kernel computation between unlabeled graphs with n vertices from O(n6) to O(n3). We find a spectral decomposition approach even more efficient when computing entire kernel matrices. For labeled graphs we develop conjugate gradient and fixed-point methods that take O(dn3) time per iteration, where d is the size of the label set. By extending the necessary linear algebra to Reproducing Kernel Hilbert Spaces (RKHS) we obtain the same result for d-dimensional edge kernels, and O(n4) in the infinite-dimensional case; on sparse graphs these algorithms only take O(n2) time per iteration in all cases. Experiments on graphs from bioinformatics and other application domains show that these techniques can speed up computation of the kernel by an order of magnitude or more. We also show that certain rational kernels (Cortes et al., 2002, 2003, 2004) when specialized to graphs reduce to our random walk graph kernel. Finally, we relate our framework to R-convolution kernels (Haussler, 1999) and provide a kernel that is close to the optimal assignment kernel of kernel of Frohlich et al. (2006) yet provably positive semi-definite.

{{< /ci-details >}}

{{< ci-details summary="Matching Node Embeddings for Graph Similarity (Giannis Nikolentzos et al., 2017)">}}

Giannis Nikolentzos, Polykarpos Meladianos, M. Vazirgiannis. (2017)  
**Matching Node Embeddings for Graph Similarity**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/0f3d2a17809f999cd4ab9d97fd5eb71086580685)  
Influential Citation Count (8), SS-ID (0f3d2a17809f999cd4ab9d97fd5eb71086580685)  

**ABSTRACT**  
Graph kernels have emerged as a powerful tool for graph comparison. Most existing graph kernels focus on local properties of graphs and ignore global structure. In this paper, we compare graphs based on their global properties as these are captured by the eigenvectors of their adjacency matrices. We present two algorithms for both labeled and unlabeled graph comparison. These algorithms represent each graph as a set of vectors corresponding to the embeddings of its vertices. The similarity between two graphs is then determined using the Earth Mover’s Distance metric. These similarities do not yield a positive semidefinite matrix. To address for this, we employ an algorithm for SVM classification using indefinite kernels. We also present a graph kernel based on the Pyramid Match kernel that finds an approximate correspondence between the sets of vectors of the two graphs. We further improve the proposed kernel using the Weisfeiler-Lehman framework. We evaluate the proposed methods on several benchmark datasets for graph classification and compare their performance to state-of-the-art graph kernels. In most cases, the proposed algorithms outperform the competing methods, while their time complexity remains very attractive.

{{< /ci-details >}}

{{< ci-details summary="struc2vec: Learning Node Representations from Structural Identity (Leonardo F. R. Ribeiro et al., 2017)">}}

Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo. (2017)  
**struc2vec: Learning Node Representations from Structural Identity**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/0f7f5679615effcc4c9b98cf2deb17c30744a6d7)  
Influential Citation Count (87), SS-ID (0f7f5679615effcc4c9b98cf2deb17c30744a6d7)  

**ABSTRACT**  
Structural identity is a concept of symmetry in which network nodes are identified according to the network structure and their relationship to other nodes. Structural identity has been studied in theory and practice over the past decades, but only recently has it been addressed with representational learning techniques. This work presents struc2vec, a novel and flexible framework for learning latent representations for the structural identity of nodes. struc2vec uses a hierarchy to measure node similarity at different scales, and constructs a multilayer graph to encode structural similarities and generate structural context for nodes. Numerical experiments indicate that state-of-the-art techniques for learning node representations fail in capturing stronger notions of structural identity, while struc2vec exhibits much superior performance in this task, as it overcomes limitations of prior approaches. As a consequence, numerical experiments indicate that struc2vec improves performance on classification tasks that depend more on structural identity.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Attacks on Classification Models for Graphs (Daniel Zügner et al., 2018)">}}

Daniel Zügner, Amir Akbarnejad, Stephan Günnemann. (2018)  
**Adversarial Attacks on Classification Models for Graphs**  
  
[Paper Link](https://www.semanticscholar.org/paper/0fc40cdd2ddecf98511a4fc2dc1f0bca6e4bae47)  
Influential Citation Count (4), SS-ID (0fc40cdd2ddecf98511a4fc2dc1f0bca6e4bae47)  

**ABSTRACT**  
Deep learning models for graphs have achieved strong performance for the task of node classification. Despite their proliferation, currently there is no study of their robustness to adversarial attacks. Yet, in domains where they are likely to be used, e.g. the web, adversaries are common. Can deep learning models for graphs be easily fooled? In this work, we introduce the first study of adversarial attacks on attributed graphs, specifically focusing on models exploiting ideas of graph convolutions. We generate adversarial perturbations targeting the node’s features and the graph structure, thus, taking the dependencies between instances in account. To cope with the underlying discrete domain we propose an efficient algorithm Nettack exploiting incremental computations. Our experimental study shows that accuracy of node classification significantly drops even when performing only few perturbations. Even more, our attacks are transferable: the learned attacks generalize to other state-of-the-art node classification models.

{{< /ci-details >}}

{{< ci-details summary="Learning Temporal Interaction Graph Embedding via Coupled Memory Networks (Zhen Zhang et al., 2020)">}}

Zhen Zhang, Jiajun Bu, Martin Ester, Jianfeng Zhang, Chengwei Yao, Z. Li, Can Wang. (2020)  
**Learning Temporal Interaction Graph Embedding via Coupled Memory Networks**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/1184a394fa785a61e125128e699950d42df22c37)  
Influential Citation Count (1), SS-ID (1184a394fa785a61e125128e699950d42df22c37)  

**ABSTRACT**  
Graph embedding has become the research focus in both academic and industrial communities due to its powerful capabilities. The majority of existing work overwhelmingly learn node embeddings in the context of static, plain or attributed, homogeneous graphs. However, many real-world applications frequently involve bipartite graphs with temporal and attributed interaction edges, named temporal interaction graphs. The temporal interactions usually imply different facets of interest and might even evolve over time, thus putting forward huge challenges in learning effective node representations. In this paper, we propose a novel framework named TigeCMN to learn node representations from a sequence of temporal interactions. Specifically, we devise two coupled memory networks to store and update node embeddings in external matrices explicitly and dynamically, which forms deep matrix representations and could enhance the expressiveness of the node embeddings. We conduct experiments on two real-world datasets and the experimental results empirically demonstrate that TigeCMN can outperform the state-of-the-arts with different gains.

{{< /ci-details >}}

{{< ci-details summary="A General Framework for Content-enhanced Network Representation Learning (Xiaofei Sun et al., 2016)">}}

Xiaofei Sun, Jiang Guo, Xiao Ding, Ting Liu. (2016)  
**A General Framework for Content-enhanced Network Representation Learning**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1268d2ae95f0b128678d6ce033ba8ea7f0d98be1)  
Influential Citation Count (11), SS-ID (1268d2ae95f0b128678d6ce033ba8ea7f0d98be1)  

**ABSTRACT**  
This paper investigates the problem of network embedding, which aims at learning low-dimensional vector representation of nodes in networks. Most existing network embedding methods rely solely on the network structure, i.e., the linkage relationships between nodes, but ignore the rich content information associated with it, which is common in real world networks and beneficial to describing the characteristics of a node. In this paper, we propose content-enhanced network embedding (CENE), which is capable of jointly leveraging the network structure and the content information. Our approach integrates text modeling and structure modeling in a general framework by treating the content information as a special kind of node. Experiments on several real world net- works with application to node classification show that our models outperform all existing network embedding methods, demonstrating the merits of content information and joint learning.

{{< /ci-details >}}

{{< ci-details summary="GeniePath: Graph Neural Networks with Adaptive Receptive Paths (Ziqi Liu et al., 2018)">}}

Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song. (2018)  
**GeniePath: Graph Neural Networks with Adaptive Receptive Paths**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/127af6effc74f073ac2442f6d82c944f562e2c0f)  
Influential Citation Count (18), SS-ID (127af6effc74f073ac2442f6d82c944f562e2c0f)  

**ABSTRACT**  
We present, GeniePath, a scalable approach for learning adaptive receptive fields of neural networks defined on permutation invariant graph data. In GeniePath, we propose an adaptive path layer consists of two complementary functions designed for breadth and depth exploration respectively, where the former learns the importance of different sized neighborhoods, while the latter extracts and filters signals aggregated from neighbors of different hops away. Our method works in both transductive and inductive settings, and extensive experiments compared with competitive methods show that our approaches yield state-of-the-art results on large graphs.

{{< /ci-details >}}

{{< ci-details summary="Applications of Link Prediction (V. Srinivas et al., 2016)">}}

V. Srinivas, Pabitra Mitra. (2016)  
**Applications of Link Prediction**  
  
[Paper Link](https://www.semanticscholar.org/paper/1303eb6de99d55c8046bf286a0d341fa614d71d0)  
Influential Citation Count (0), SS-ID (1303eb6de99d55c8046bf286a0d341fa614d71d0)  

**ABSTRACT**  
Link prediction has a wide variety of applications. Graphs provide a natural abstraction to represent interactions between different entities in a network. We can have graphs representing social networks, transportation networks, disease networks, email/telephone calls network to list a few. Link prediction can specifically be applied on these networks to analyze and solve interesting problems like predicting outbreak of a disease, controlling privacy in networks, detecting spam emails, suggesting alternative routes for possible navigation based on the current traffic patterns, etc.

{{< /ci-details >}}

{{< ci-details summary="Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks (Federico Monti et al., 2017)">}}

Federico Monti, M. Bronstein, X. Bresson. (2017)  
**Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/137bbe604334584fd4a1d6eb9218a588ae3dda3e)  
Influential Citation Count (51), SS-ID (137bbe604334584fd4a1d6eb9218a588ae3dda3e)  

**ABSTRACT**  
Matrix completion models are among the most common formulations of recommender systems. Recent works have showed a boost of performance of these techniques when introducing the pairwise relationships between users/items in the form of graphs, and imposing smoothness priors on these graphs. However, such techniques do not fully exploit the local stationarity structures of user/item graphs, and the number of parameters to learn is linear w.r.t. the number of users and items. We propose a novel approach to overcome these limitations by using geometric deep learning on graphs. Our matrix completion architecture combines graph convolutional neural networks and recurrent neural networks to learn meaningful statistical graph-structured patterns and the non-linear diffusion process that generates the known ratings. This neural network system requires a constant number of parameters independent of the matrix size. We apply our method on both synthetic and real datasets, showing that it outperforms state-of-the-art techniques.

{{< /ci-details >}}

{{< ci-details summary="Temporal Graph Networks for Deep Learning on Dynamic Graphs (Emanuele Rossi et al., 2020)">}}

Emanuele Rossi, B. Chamberlain, F. Frasca, D. Eynard, Federico Monti, M. Bronstein. (2020)  
**Temporal Graph Networks for Deep Learning on Dynamic Graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/150f95f9c73820e0a0fa1546140e9f2bdfd25954)  
Influential Citation Count (26), SS-ID (150f95f9c73820e0a0fa1546140e9f2bdfd25954)  

**ABSTRACT**  
Graph Neural Networks (GNNs) have recently become increasingly popular due to their ability to learn complex systems of relations or interactions arising in a broad spectrum of problems ranging from biology and particle physics to social networks and recommendation systems. Despite the plethora of different models for deep learning on graphs, few approaches have been proposed thus far for dealing with graphs that present some sort of dynamic nature (e.g. evolving features or connectivity over time). In this paper, we present Temporal Graph Networks (TGNs), a generic, efficient framework for deep learning on dynamic graphs represented as sequences of timed events. Thanks to a novel combination of memory modules and graph-based operators, TGNs are able to significantly outperform previous approaches being at the same time more computationally efficient. We furthermore show that several previous models for learning on dynamic graphs can be cast as specific instances of our framework. We perform a detailed ablation study of different components of our framework and devise the best configuration that achieves state-of-the-art performance on several transductive and inductive prediction tasks for dynamic graphs.

{{< /ci-details >}}

{{< ci-details summary="Learning Image and User Features for Recommendation in Social Networks (Xue Geng et al., 2015)">}}

Xue Geng, Hanwang Zhang, Jingwen Bian, Tat-Seng Chua. (2015)  
**Learning Image and User Features for Recommendation in Social Networks**  
2015 IEEE International Conference on Computer Vision (ICCV)  
[Paper Link](https://www.semanticscholar.org/paper/15f5721502c2905c555a4eb0a110d6fc211c1fb2)  
Influential Citation Count (11), SS-ID (15f5721502c2905c555a4eb0a110d6fc211c1fb2)  

**ABSTRACT**  
Good representations of data do help in many machine learning tasks such as recommendation. It is often a great challenge for traditional recommender systems to learn representative features of both users and images in large social networks, in particular, social curation networks, which are characterized as the extremely sparse links between users and images, and the extremely diverse visual contents of images. To address the challenges, we propose a novel deep model which learns the unified feature representations for both users and images. This is done by transforming the heterogeneous user-image networks into homogeneous low-dimensional representations, which facilitate a recommender to trivially recommend images to users by feature similarity. We also develop a fast online algorithm that can be easily scaled up to large networks in an asynchronously parallel way. We conduct extensive experiments on a representative subset of Pinterest, containing 1,456,540 images and 1,000,000 users. Results of image recommendation experiments demonstrate that our feature learning approach significantly outperforms other state-of-the-art recommendation methods.

{{< /ci-details >}}

{{< ci-details summary="A Tutorial on Network Embeddings (Haochen Chen et al., 2018)">}}

Haochen Chen, Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2018)  
**A Tutorial on Network Embeddings**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1706a4ef5556ecdd680416d46e033e0476290361)  
Influential Citation Count (1), SS-ID (1706a4ef5556ecdd680416d46e033e0476290361)  

**ABSTRACT**  
Network embedding methods aim at learning low-dimensional latent representation of nodes in a network. These representations can be used as features for a wide range of tasks on graphs such as classification, clustering, link prediction, and visualization. In this survey, we give an overview of network embeddings by summarizing and categorizing recent advancements in this research field. We first discuss the desirable properties of network embeddings and briefly introduce the history of network embedding algorithms. Then, we discuss network embedding methods under different scenarios, such as supervised versus unsupervised learning, learning embeddings for homogeneous networks versus for heterogeneous networks, etc. We further demonstrate the applications of network embeddings, and conclude the survey with future work in this area.

{{< /ci-details >}}

{{< ci-details summary="Diffusion-Convolutional Neural Networks (James Atwood et al., 2015)">}}

James Atwood, D. Towsley. (2015)  
**Diffusion-Convolutional Neural Networks**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/18b47b83a373f33d6b902a3615f42c10f7600d72)  
Influential Citation Count (69), SS-ID (18b47b83a373f33d6b902a3615f42c10f7600d72)  

**ABSTRACT**  
We present diffusion-convolutional neural networks (DCNNs), a new model for graph-structured data. Through the introduction of a diffusion-convolution operation, we show how diffusion-based representations can be learned from graph-structured data and used as an effective basis for node classification. DCNNs have several attractive qualities, including a latent representation for graphical data that is invariant under isomorphism, as well as polynomial-time prediction and learning that can be represented as tensor operations and efficiently implemented on the GPU. Through several experiments with real structured datasets, we demonstrate that DCNNs are able to outperform probabilistic relational models and kernel-on-graph methods at relational node classification tasks.

{{< /ci-details >}}

{{< ci-details summary="Automatic Virtual Network Embedding: A Deep Reinforcement Learning Approach With Graph Convolutional Networks (Zhongxia Yan et al., 2020)">}}

Zhongxia Yan, Jingguo Ge, Yulei Wu, Liangxiong Li, Tong Li. (2020)  
**Automatic Virtual Network Embedding: A Deep Reinforcement Learning Approach With Graph Convolutional Networks**  
IEEE Journal on Selected Areas in Communications  
[Paper Link](https://www.semanticscholar.org/paper/18b619339f0abfdae1edb19abb11f43ca4f90842)  
Influential Citation Count (7), SS-ID (18b619339f0abfdae1edb19abb11f43ca4f90842)  

**ABSTRACT**  
Virtual network embedding arranges virtual network services onto substrate network components. The performance of embedding algorithms determines the effectiveness and efficiency of a virtualized network, making it a critical part of the network virtualization technology. To achieve better performance, the algorithm needs to automatically detect the network status which is complicated and changes in a time-varying manner, and to dynamically provide solutions that can best fit the current network status. However, most existing algorithms fail to provide automatic embedding solutions in an acceptable running time. In this paper, we combine deep reinforcement learning with a novel neural network structure based on graph convolutional networks, and propose a new and efficient algorithm for automatic virtual network embedding. In addition, a parallel reinforcement learning framework is used in training along with a newly-designed multi-objective reward function, which has proven beneficial to the proposed algorithm for automatic embedding of virtual networks. Extensive simulation results under different scenarios show that our algorithm achieves best performance on most metrics compared with the existing state-of-the-art solutions, with upto 39.6% and 70.6% improvement on acceptance ratio and average revenue, respectively. Moreover, the results also demonstrate that the proposed solution possesses good robustness.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Embedding via Dynamic Mapping Matrix (Guoliang Ji et al., 2015)">}}

Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao. (2015)  
**Knowledge Graph Embedding via Dynamic Mapping Matrix**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/18bd7cd489874ed9976b4f87a6a558f9533316e0)  
Influential Citation Count (130), SS-ID (18bd7cd489874ed9976b4f87a6a558f9533316e0)  

**ABSTRACT**  
Knowledge graphs are useful resources for numerous AI applications, but they are far from completeness. Previous work such as TransE, TransH and TransR/CTransR regard a relation as translation from head entity to tail entity and the CTransR achieves state-of-the-art performance. In this paper, we propose a more fine-grained model named TransD, which is an improvement of TransR/CTransR. In TransD, we use two vectors to represent a named symbol object (entity and relation). The first one represents the meaning of a(n) entity (relation), the other one is used to construct mapping matrix dynamically. Compared with TransR/CTransR, TransD not only considers the diversity of relations, but also entities. TransD has less parameters and has no matrix-vector multiplication operations, which makes it can be applied on large scale graphs. In Experiments, we evaluate our model on two typical tasks including triplets classification and link prediction. Evaluation results show that our approach outperforms state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Deep Neural Networks for Learning Graph Representations (Shaosheng Cao et al., 2016)">}}

Shaosheng Cao, Wei Lu, Qiongkai Xu. (2016)  
**Deep Neural Networks for Learning Graph Representations**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/1a37f07606d60df365d74752857e8ce909f700b3)  
Influential Citation Count (59), SS-ID (1a37f07606d60df365d74752857e8ce909f700b3)  

**ABSTRACT**  
In this paper, we propose a novel model for learning graph representations, which generates a low-dimensional vector representation for each vertex by capturing the graph structural information. Different from other previous research efforts, we adopt a random surfing model to capture graph structural information directly, instead of using the sampling-based method for generating linear sequences proposed by Perozzi et al. (2014). The advantages of our approach will be illustrated from both theorical and empirical perspectives. We also give a new perspective for the matrix factorization method proposed by Levy and Goldberg (2014), in which the pointwise mutual information (PMI) matrix is considered as an analytical solution to the objective function of the skip-gram model with negative sampling proposed by Mikolov et al. (2013). Unlike their approach which involves the use of the SVD for finding the low-dimensitonal projections from the PMI matrix, however, the stacked denoising autoencoder is introduced in our model to extract complex features and model non-linearities. To demonstrate the effectiveness of our model, we conduct experiments on clustering and visualization tasks, employing the learned vertex representations as features. Empirical results on datasets of varying sizes show that our model outperforms other stat-of-the-art models in such tasks.

{{< /ci-details >}}

{{< ci-details summary="Visualizing Data using t-SNE (L. V. D. Maaten et al., 2008)">}}

L. V. D. Maaten, Geoffrey E. Hinton. (2008)  
**Visualizing Data using t-SNE**  
  
[Paper Link](https://www.semanticscholar.org/paper/1c46943103bd7b7a2c7be86859995a4144d1938b)  
Influential Citation Count (826), SS-ID (1c46943103bd7b7a2c7be86859995a4144d1938b)  

**ABSTRACT**  
We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large datasets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of datasets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the datasets.

{{< /ci-details >}}

{{< ci-details summary="Capped Lp-Norm Graph Embedding for Photo Clustering (Mengfan Tang et al., 2016)">}}

Mengfan Tang, F. Nie, R. Jain. (2016)  
**Capped Lp-Norm Graph Embedding for Photo Clustering**  
ACM Multimedia  
[Paper Link](https://www.semanticscholar.org/paper/1cb5547c3f5ff42746bf9c4e083795aed3c8c609)  
Influential Citation Count (0), SS-ID (1cb5547c3f5ff42746bf9c4e083795aed3c8c609)  

**ABSTRACT**  
Photos are a predominant source of information on a global scale. Cluster analysis of photos can be applied to situation recognition and understanding cultural dynamics. Graph-based learning provides a current approach for modeling data in clustering problems. However, the performance of this framework depends heavily on initial graph construction by input data. Data outliers degrade graph quality, leading to poor clustering results. We designed a new capped lp-norm graph-based model to reduce the impact of outliers. This is accomplished by allowing the data graph to self adjust as part of the graph embedding. Furthermore, we derive an iterative algorithm to solve the objective function optimization problem. Experiments on four real-world benchmark data sets and Yahoo Flickr Creative Commons data set show the effectiveness of this new graph-based capped lp-norm clustering method.

{{< /ci-details >}}

{{< ci-details summary="From Visual Data Exploration to Visual Data Mining: A Survey (M. C. Oliveira et al., 2003)">}}

M. C. Oliveira, H. Levkowitz. (2003)  
**From Visual Data Exploration to Visual Data Mining: A Survey**  
IEEE Trans. Vis. Comput. Graph.  
[Paper Link](https://www.semanticscholar.org/paper/1e008a1f5484094eb5794672d7c7318dd86f4fb5)  
Influential Citation Count (22), SS-ID (1e008a1f5484094eb5794672d7c7318dd86f4fb5)  

**ABSTRACT**  
We survey work on the different uses of graphical mapping and interaction techniques for visual data mining of large data sets represented as table data. Basic terminology related to data mining, data sets, and visualization is introduced. Previous work on information visualization is reviewed in light of different categorizations of techniques and systems. The role of interaction techniques is discussed, in addition to work addressing the question of selecting and evaluating visualization techniques. We review some representative work on the use of information visualization techniques in the context of mining data. This includes both visual data exploration and visually expressing the outcome of specific mining algorithms. We also review recent innovative approaches that attempt to integrate visualization into the DM/KDD process, using it to enhance user interaction and comprehension.

{{< /ci-details >}}

{{< ci-details summary="Dropout Training of Matrix Factorization and Autoencoder for Link Prediction in Sparse Graphs (Shuangfei Zhai et al., 2015)">}}

Shuangfei Zhai, Zhongfei Zhang. (2015)  
**Dropout Training of Matrix Factorization and Autoencoder for Link Prediction in Sparse Graphs**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/1e79e7d3247c7fddebaf5242c661de79bf7f31a7)  
Influential Citation Count (3), SS-ID (1e79e7d3247c7fddebaf5242c661de79bf7f31a7)  

**ABSTRACT**  
Matrix factorization (MF) and Autoencoder (AE) are among the most successful approaches of unsupervised learning. While MF based models have been extensively exploited in the graph modeling and link prediction literature, the AE family has not gained much attention. In this paper we investigate both MF and AE's application to the link prediction problem in sparse graphs. We show the connection between AE and MF from the perspective of multiview learning, and further propose MF+AE: a model training MF and AE jointly with shared parameters. We apply dropout to training both the MF and AE parts, and show that it can significantly prevent overfitting by acting as an adaptive regularization. We conduct experiments on six real world sparse graph datasets, and show that MF+AE consistently outperforms the competing methods, especially on datasets that demonstrate strong non-cohesive structures.

{{< /ci-details >}}

{{< ci-details summary="Graph-Based Global Reasoning Networks (Yunpeng Chen et al., 2018)">}}

Yunpeng Chen, Marcus Rohrbach, Zhicheng Yan, Shuicheng Yan, Jiashi Feng, Yannis Kalantidis. (2018)  
**Graph-Based Global Reasoning Networks**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/1eaee16f6395c9602ad1dc17e69a6e235ec9ddd6)  
Influential Citation Count (32), SS-ID (1eaee16f6395c9602ad1dc17e69a6e235ec9ddd6)  

**ABSTRACT**  
Globally modeling and reasoning over relations between regions can be beneficial for many computer vision tasks on both images and videos. Convolutional Neural Networks (CNNs) excel at modeling local relations by convolution operations, but they are typically inefficient at capturing global relations between distant regions and require stacking multiple convolution layers. In this work, we propose a new approach for reasoning globally in which a set of features are globally aggregated over the coordinate space and then projected to an interaction space where relational reasoning can be efficiently computed. After reasoning, relation-aware features are distributed back to the original coordinate space for down-stream tasks. We further present a highly efficient instantiation of the proposed approach and introduce the Global Reasoning unit (GloRe unit) that implements the coordinate-interaction space mapping by weighted global pooling and weighted broadcasting, and the relation reasoning via graph convolution on a small graph in interaction space. The proposed GloRe unit is lightweight, end-to-end trainable and can be easily plugged into existing CNNs for a wide range of tasks. Extensive experiments show our GloRe unit can consistently boost the performance of state-of-the-art backbone architectures, including ResNet, ResNeXt, SE-Net and DPN, for both 2D and 3D CNNs, on image classification, semantic segmentation and video action recognition task.

{{< /ci-details >}}

{{< ci-details summary="Learning Structured Embeddings of Knowledge Bases (Antoine Bordes et al., 2011)">}}

Antoine Bordes, J. Weston, Ronan Collobert, Yoshua Bengio. (2011)  
**Learning Structured Embeddings of Knowledge Bases**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/1f4a4769e4d2fb846e59c2f185e0377190739f18)  
Influential Citation Count (81), SS-ID (1f4a4769e4d2fb846e59c2f185e0377190739f18)  

**ABSTRACT**  
Many Knowledge Bases (KBs) are now readily available and encompass colossal quantities of information thanks to either a long-term funding effort (e.g. WordNet, OpenCyc) or a collaborative process (e.g. Freebase, DBpedia). However, each of them is based on a different rigid symbolic framework which makes it hard to use their data in other systems. It is unfortunate because such rich structured knowledge might lead to a huge leap forward in many other areas of AI like natural language processing (word-sense disambiguation, natural language understanding, ...), vision (scene classification, image semantic annotation, ...) or collaborative filtering. In this paper, we present a learning process based on an innovative neural network architecture designed to embed any of these symbolic representations into a more flexible continuous vector space in which the original knowledge is kept and enhanced. These learnt embeddings would allow data from any KB to be easily used in recent machine learning methods for prediction and information retrieval. We illustrate our method on WordNet and Freebase and also present a way to adapt it to knowledge extraction from raw text.

{{< /ci-details >}}

{{< ci-details summary="CAGE: Constrained deep Attributed Graph Embedding (Debora Nozza et al., 2020)">}}

Debora Nozza, E. Fersini, E. Messina. (2020)  
**CAGE: Constrained deep Attributed Graph Embedding**  
Inf. Sci.  
[Paper Link](https://www.semanticscholar.org/paper/1f90f847e46bacc13364975166ff2c908436735d)  
Influential Citation Count (0), SS-ID (1f90f847e46bacc13364975166ff2c908436735d)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Attention is All you Need (Ashish Vaswani et al., 2017)">}}

Ashish Vaswani, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2017)  
**Attention is All you Need**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/204e3073870fae3d05bcbc2f6a8e263d9b72e776)  
Influential Citation Count (7570), SS-ID (204e3073870fae3d05bcbc2f6a8e263d9b72e776)  

**ABSTRACT**  
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

{{< /ci-details >}}

{{< ci-details summary="CANE: Context-Aware Network Embedding for Relation Modeling (Cunchao Tu et al., 2017)">}}

Cunchao Tu, Han Liu, Zhiyuan Liu, Maosong Sun. (2017)  
**CANE: Context-Aware Network Embedding for Relation Modeling**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/20bb300eb3400f1af766110ff51feada78170674)  
Influential Citation Count (36), SS-ID (20bb300eb3400f1af766110ff51feada78170674)  

**ABSTRACT**  
Network embedding (NE) is playing a critical role in network analysis, due to its ability to represent vertices with efficient low-dimensional embedding vectors. However, existing NE models aim to learn a fixed context-free embedding for each vertex and neglect the diverse roles when interacting with other vertices. In this paper, we assume that one vertex usually shows different aspects when interacting with different neighbor vertices, and should own different embeddings respectively. Therefore, we present Context-Aware Network Embedding (CANE), a novel NE model to address this issue. CANE learns context-aware embeddings for vertices with mutual attention mechanism and is expected to model the semantic relationships between vertices more precisely. In experiments, we compare our model with existing NE models on three real-world datasets. Experimental results show that CANE achieves significant improvement than state-of-the-art methods on link prediction and comparable performance on vertex classification. The source code and datasets can be obtained from https://github.com/thunlp/CANE.

{{< /ci-details >}}

{{< ci-details summary="Predicting MicroRNA-Disease Associations Using Network Topological Similarity Based on DeepWalk (Guanghui Li et al., 2017)">}}

Guanghui Li, Jiawei Luo, Qiu Xiao, C. Liang, Pingjian Ding, Buwen Cao. (2017)  
**Predicting MicroRNA-Disease Associations Using Network Topological Similarity Based on DeepWalk**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/21b9fda2ae02e36e57b5727b9da8be3eda36a7d9)  
Influential Citation Count (0), SS-ID (21b9fda2ae02e36e57b5727b9da8be3eda36a7d9)  

**ABSTRACT**  
Recently, increasing experimental studies have shown that microRNAs (miRNAs) involved in multiple physiological processes are connected with several complex human diseases. Identifying human disease-related miRNAs will be useful in uncovering novel prognostic markers for cancer. Currently, several computational approaches have been developed for miRNA-disease association prediction based on the integration of additional biological information of diseases and miRNAs, such as disease semantic similarity and miRNA functional similarity. However, these methods do not work well when this information is unavailable. In this paper, we present a similarity-based miRNA-disease prediction method that enhances the existing association discovery methods through a topology-based similarity measure. DeepWalk, a deep learning method, is utilized in this paper to calculate similarities within a miRNA-disease association network. It shows superior predictive performance for 22 complex diseases, with area under the ROC curve scores ranging from 0.805 to 0.937 by using five-fold cross-validation. In addition, case studies on breast cancer, lung cancer, and prostatic cancer further justify the use of our method to discover latent miRNA-disease pairs.

{{< /ci-details >}}

{{< ci-details summary="SIGNet: Scalable Embeddings for Signed Networks (Mohammad Raihanul Islam et al., 2017)">}}

Mohammad Raihanul Islam, B. Prakash, Naren Ramakrishnan. (2017)  
**SIGNet: Scalable Embeddings for Signed Networks**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/21bcb27995ae1007f4dabe5973c5fa6df7706f3e)  
Influential Citation Count (3), SS-ID (21bcb27995ae1007f4dabe5973c5fa6df7706f3e)  

**ABSTRACT**  
Recent successes in word embedding and document embedding have motivated researchers to explore similar representations for networks and to use such representations for tasks such as edge prediction, node label prediction, and community detection. Such network embedding methods are largely focused on finding distributed representations for unsigned networks and are unable to discover embeddings that respect polarities inherent in edges. We propose SIGNet, a fast scalable embedding method suitable for signed networks. Our proposed objective function aims to carefully model the social structure implicit in signed networks by reinforcing the principles of social balance theory. Our method builds upon the traditional word2vec family of embedding approaches and adds a new targeted node sampling strategy to maintain structural balance in higher-order neighborhoods. We demonstrate the superiority of SIGNet over state-of-the-art methods proposed for both signed and unsigned networks on several real world datasets from different domains. In particular, SIGNet offers an approach to generate a richer vocabulary of features of signed networks to support representation and reasoning.

{{< /ci-details >}}

{{< ci-details summary="Birds of a Feather: Homophily in Social Networks (M. McPherson et al., 2001)">}}

M. McPherson, L. Smith-Lovin, J. Cook. (2001)  
**Birds of a Feather: Homophily in Social Networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/228bafce55e6f1cbe2c1df75b1949a1fb9c93eb3)  
Influential Citation Count (673), SS-ID (228bafce55e6f1cbe2c1df75b1949a1fb9c93eb3)  

**ABSTRACT**  
Similarity breeds connection. This principle—the homophily principle—structures network ties of every type, including marriage, friendship, work, advice, support, information transfer, exchange, comembership, and other types of relationship. The result is that people's personal networks are homogeneous with regard to many sociodemographic, behavioral, and intrapersonal characteristics. Homophily limits people's social worlds in a way that has powerful implications for the information they receive, the attitudes they form, and the interactions they experience. Homophily in race and ethnicity creates the strongest divides in our personal environments, with age, religion, education, occupation, and gender following in roughly that order. Geographic propinquity, families, organizations, and isomorphic positions in social systems all create contexts in which homophilous relations form. Ties between nonsimilar individuals also dissolve at a higher rate, which sets the stage for the formation of niches (localize...

{{< /ci-details >}}

{{< ci-details summary="Topology and Content Co-Alignment Graph Convolutional Learning (Min Shi et al., 2020)">}}

Min Shi, Yufei Tang, Xingquan Zhu. (2020)  
**Topology and Content Co-Alignment Graph Convolutional Learning**  
IEEE transactions on neural networks and learning systems  
[Paper Link](https://www.semanticscholar.org/paper/249ce8e6bf5db2d0e12a5212330acdff3683550f)  
Influential Citation Count (0), SS-ID (249ce8e6bf5db2d0e12a5212330acdff3683550f)  

**ABSTRACT**  
In traditional graph neural networks (GNNs), graph convolutional learning is carried out through topology-driven recursive node content aggregation for network representation learning. In reality, network topology and node content each provide unique and important information, and they are not always consistent because of noise, irrelevance, or missing links between nodes. A pure topology-driven feature aggregation approach between unaligned neighborhoods may deteriorate learning from nodes with poor structure-content consistency, due to the propagation of incorrect messages over the whole network. Alternatively, in this brief, we advocate a co-alignment graph convolutional learning (CoGL) paradigm, by aligning topology and content networks to maximize consistency. Our theme is to enforce the learning from the topology network to be consistent with the content network while simultaneously optimizing the content network to comply with the topology for optimized representation learning. Given a network, CoGL first reconstructs a content network from node features then co-aligns the content network and the original network through a unified optimization goal with: 1) minimized content loss; 2) minimized classification loss; and 3) minimized adversarial loss. Experiments on six benchmarks demonstrate that CoGL achieves comparable and even better performance compared with existing state-of-the-art GNN models.

{{< /ci-details >}}

{{< ci-details summary="Probabilistic Latent Document Network Embedding (Tuan M. V. Le et al., 2014)">}}

Tuan M. V. Le, Hady W. Lauw. (2014)  
**Probabilistic Latent Document Network Embedding**  
2014 IEEE International Conference on Data Mining  
[Paper Link](https://www.semanticscholar.org/paper/24f7d72e92cadfa5c84949537639ce084b9d2092)  
Influential Citation Count (8), SS-ID (24f7d72e92cadfa5c84949537639ce084b9d2092)  

**ABSTRACT**  
A document network refers to a data type that can be represented as a graph of vertices, where each vertex is associated with a text document. Examples of such a data type include hyperlinked Web pages, academic publications with citations, and user profiles in social networks. Such data have very high-dimensional representations, in terms of text as well as network connectivity. In this paper, we study the problem of embedding, or finding a low-dimensional representation of a document network that "preserves" the data as much as possible. These embedded representations are useful for various applications driven by dimensionality reduction, such as visualization or feature selection. While previous works in embedding have mostly focused on either the textual aspect or the network aspect, we advocate a holistic approach by finding a unified low-rank representation for both aspects. Moreover, to lend semantic interpretability to the low-rank representation, we further propose to integrate topic modeling and embedding within a joint model. The gist is to join the various representations of a document (words, links, topics, and coordinates) within a generative model, and to estimate the hidden representations through MAP estimation. We validate our model on real-life document networks, showing that it outperforms comparable baselines comprehensively on objective evaluation metrics.

{{< /ci-details >}}

{{< ci-details summary="Translating Embeddings for Modeling Multi-relational Data (Antoine Bordes et al., 2013)">}}

Antoine Bordes, Nicolas Usunier, Alberto García-Durán, J. Weston, Oksana Yakhnenko. (2013)  
**Translating Embeddings for Modeling Multi-relational Data**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/2582ab7c70c9e7fcb84545944eba8f3a7f253248)  
Influential Citation Count (1287), SS-ID (2582ab7c70c9e7fcb84545944eba8f3a7f253248)  

**ABSTRACT**  
We consider the problem of embedding entities and relationships of multi-relational data in low-dimensional vector spaces. Our objective is to propose a canonical model which is easy to train, contains a reduced number of parameters and can scale up to very large databases. Hence, we propose TransE, a method which models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities. Despite its simplicity, this assumption proves to be powerful since extensive experiments show that TransE significantly outperforms state-of-the-art methods in link prediction on two knowledge bases. Besides, it can be successfully trained on a large scale data set with 1M entities, 25k relationships and more than 17M training samples.

{{< /ci-details >}}

{{< ci-details summary="SCAN: a structural clustering algorithm for networks (Xiaowei Xu et al., 2007)">}}

Xiaowei Xu, Nurcan Yuruk, Zhidan Feng, T. Schweiger. (2007)  
**SCAN: a structural clustering algorithm for networks**  
KDD '07  
[Paper Link](https://www.semanticscholar.org/paper/25dfac6955913c163a61bc6e2ae8c5c7f3ca8f87)  
Influential Citation Count (107), SS-ID (25dfac6955913c163a61bc6e2ae8c5c7f3ca8f87)  

**ABSTRACT**  
Network clustering (or graph partitioning) is an important task for the discovery of underlying structures in networks. Many algorithms find clusters by maximizing the number of intra-cluster edges. While such algorithms find useful and interesting structures, they tend to fail to identify and isolate two kinds of vertices that play special roles - vertices that bridge clusters (hubs) and vertices that are marginally connected to clusters (outliers). Identifying hubs is useful for applications such as viral marketing and epidemiology since hubs are responsible for spreading ideas or disease. In contrast, outliers have little or no influence, and may be isolated as noise in the data. In this paper, we proposed a novel algorithm called SCAN (Structural Clustering Algorithm for Networks), which detects clusters, hubs and outliers in networks. It clusters vertices based on a structural similarity measure. The algorithm is fast and efficient, visiting each vertex only once. An empirical evaluation of the method using both synthetic and real datasets demonstrates superior performance over other methods such as the modularity-based algorithms.

{{< /ci-details >}}

{{< ci-details summary="A Survey on Graph Drawing Beyond Planarity (W. Didimo et al., 2018)">}}

W. Didimo, G. Liotta, Fabrizio Montecchiani. (2018)  
**A Survey on Graph Drawing Beyond Planarity**  
ACM Comput. Surv.  
[Paper Link](https://www.semanticscholar.org/paper/2900ef06167eb5684020db9ccfcb9fe42a07a919)  
Influential Citation Count (2), SS-ID (2900ef06167eb5684020db9ccfcb9fe42a07a919)  

**ABSTRACT**  
Graph Drawing Beyond Planarity is a rapidly growing research area that classifies and studies geometric representations of nonplanar graphs in terms of forbidden crossing configurations. The aim of this survey is to describe the main research directions in this area, the most prominent known results, and some of the most challenging open problems.

{{< /ci-details >}}

{{< ci-details summary="Impact of different metrics on multi-view clustering (Angela Serra et al., 2015)">}}

Angela Serra, D. Greco, R. Tagliaferri. (2015)  
**Impact of different metrics on multi-view clustering**  
2015 International Joint Conference on Neural Networks (IJCNN)  
[Paper Link](https://www.semanticscholar.org/paper/294b25ce7576ab5cc59ab0de1d36425cae7fab2a)  
Influential Citation Count (0), SS-ID (294b25ce7576ab5cc59ab0de1d36425cae7fab2a)  

**ABSTRACT**  
Clustering of patients allows to find groups of subjects with similar characteristics. This categorization can facilitate diagnosis, treatment decision and prognosis prediction. Heterogeneous genome-wide data sources capture different biological aspects that can be integrated in order to better categorize the patients. Clustering methods work by comparing how patients are similar or dissimilar in a suitable similarity space. While several clustering methods have been proposed, there is no systematic comparative study concerning the impact of similarity metrics on the cluster quality. We compared seven popular similarity measures (Pearson, Spearman and Kendall Correlations; Euclidean, Canberra, Minkowski and Manhattan Distances) in conjunction with two classical single-view clustering algorithms and a late integration approach (partitioning around medoids, hierarchical clustering and matrix factorization approaches), on high dimensional multi-view cancer data coming from the TCGA repository. Performance was measured against tumour subcategories classification. Only Euclidean and Minkowski distances showed similar results in terms of clustering similarity indexes. On the other hand, an absolute best similarity measure did not emerge in terms of misclassification, but it strongly depends on the data.

{{< /ci-details >}}

{{< ci-details summary="Supervised random walks: predicting and recommending links in social networks (L. Backstrom et al., 2010)">}}

L. Backstrom, J. Leskovec. (2010)  
**Supervised random walks: predicting and recommending links in social networks**  
WSDM '11  
[Paper Link](https://www.semanticscholar.org/paper/29efbdf3f95cee97405accafdebd3bd374f1f003)  
Influential Citation Count (79), SS-ID (29efbdf3f95cee97405accafdebd3bd374f1f003)  

**ABSTRACT**  
Predicting the occurrence of links is a fundamental problem in networks. In the link prediction problem we are given a snapshot of a network and would like to infer which interactions among existing members are likely to occur in the near future or which existing interactions are we missing. Although this problem has been extensively studied, the challenge of how to effectively combine the information from the network structure with rich node and edge attribute data remains largely open.  We develop an algorithm based on Supervised Random Walks that naturally combines the information from the network structure with node and edge level attributes. We achieve this by using these attributes to guide a random walk on the graph. We formulate a supervised learning task where the goal is to learn a function that assigns strengths to edges in the network such that a random walker is more likely to visit the nodes to which new links will be created in the future. We develop an efficient training algorithm to directly learn the edge strength estimation function.  Our experiments on the Facebook social graph and large collaboration networks show that our approach outperforms state-of-the-art unsupervised approaches as well as approaches that are based on feature extraction.

{{< /ci-details >}}

{{< ci-details summary="Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding (Xiang Ren et al., 2016)">}}

Xiang Ren, Wenqi He, Meng Qu, Clare R. Voss, Heng Ji, Jiawei Han. (2016)  
**Label Noise Reduction in Entity Typing by Heterogeneous Partial-Label Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/2a1d2b775997d6e2e42a054ea0ed456af1060796)  
Influential Citation Count (18), SS-ID (2a1d2b775997d6e2e42a054ea0ed456af1060796)  

**ABSTRACT**  
Current systems of fine-grained entity typing use distant supervision in conjunction with existing knowledge bases to assign categories (type labels) to entity mentions. However, the type labels so obtained from knowledge bases are often noisy (i.e., incorrect for the entity mention's local context). We define a new task, Label Noise Reduction in Entity Typing (LNR), to be the automatic identification of correct type labels (type-paths) for training examples, given the set of candidate type labels obtained by distant supervision with a given type hierarchy. The unknown type labels for individual entity mentions and the semantic similarity between entity types pose unique challenges for solving the LNR task. We propose a general framework, called PLE, to jointly embed entity mentions, text features and entity types into the same low-dimensional space where, in that space, objects whose types are semantically close have similar representations. Then we estimate the type-path for each training example in a top-down manner using the learned embeddings. We formulate a global objective for learning the embeddings from text corpora and knowledge bases, which adopts a novel margin-based loss that is robust to noisy labels and faithfully models type correlation derived from knowledge bases. Our experiments on three public typing datasets demonstrate the effectiveness and robustness of PLE, with an average of 25% improvement in accuracy compared to next best method.

{{< /ci-details >}}

{{< ci-details summary="VERSE: Versatile Graph Embeddings from Similarity Measures (Anton Tsitsulin et al., 2018)">}}

Anton Tsitsulin, D. Mottin, P. Karras, Emmanuel Müller. (2018)  
**VERSE: Versatile Graph Embeddings from Similarity Measures**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/2a2ec58c7813820592cd487d66ed0b249b846eb0)  
Influential Citation Count (46), SS-ID (2a2ec58c7813820592cd487d66ed0b249b846eb0)  

**ABSTRACT**  
Embedding a web-scale information network into a low-dimensional vector space facilitates tasks such as link prediction, classification, and visualization. Past research has addressed the problem of extracting such embeddings by adopting methods from words to graphs, without defining a clearly comprehensible graph-related objective. Yet, as we show, the objectives used in past works implicitly utilize similarity measures among graph nodes. In this paper, we carry the similarity orientation of previous works to its logical conclusion; we propose VERtex Similarity Embeddings (VERSE), a simple, versatile, and memory-efficient method that derives graph embeddings explicitly calibrated to preserve the distributions of a selected vertex-to-vertex similarity measure. VERSE learns such embeddings by training a single-layer neural network. While its default, scalable version does so via sampling similarity information, we also develop a variant using the full information per vertex. Our experimental study on standard benchmarks and real-world datasets demonstrates that VERSE, instantiated with diverse similarity measures, outperforms state-of-the-art methods in terms of precision and recall in major data mining tasks and supersedes them in time and space efficiency, while the scalable sampling-based variant achieves equally good result as the non-scalable full variant.

{{< /ci-details >}}

{{< ci-details summary="GSSNN: Graph Smoothing Splines Neural Networks (Shichao Zhu et al., 2020)">}}

Shichao Zhu, Lewei Zhou, Shirui Pan, Chuan Zhou, Guiying Yan, Bin Wang. (2020)  
**GSSNN: Graph Smoothing Splines Neural Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/2a3e3607fb9fbfa0d637946ca48035e68e7fee45)  
Influential Citation Count (0), SS-ID (2a3e3607fb9fbfa0d637946ca48035e68e7fee45)  

**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved state-of-the-art performance in many graph data analysis tasks. However, they still suffer from two limitations for graph representation learning. First, they exploit non-smoothing node features which may result in suboptimal embedding and degenerated performance for graph classification. Second, they only exploit neighbor information but ignore global topological knowledge. Aiming to overcome these limitations simultaneously, in this paper, we propose a novel, flexible, and end-to-end framework, Graph Smoothing Splines Neural Networks (GSSNN), for graph classification. By exploiting the smoothing splines, which are widely used to learn smoothing fitting function in regression, we develop an effective feature smoothing and enhancement module Scaled Smoothing Splines (S3) to learn graph embedding. To integrate global topological information, we design a novel scoring module, which exploits closeness, degree, as well as self-attention values, to select important node features as knots for smoothing splines. These knots can be potentially used for interpreting classification results. In extensive experiments on biological and social datasets, we demonstrate that our model achieves state-of-the-arts and GSSNN is superior in learning more robust graph representations. Furthermore, we show that S3 module is easily plugged into existing GNNs to improve their performance.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Embedding by Translating on Hyperplanes (Zhen Wang et al., 2014)">}}

Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen. (2014)  
**Knowledge Graph Embedding by Translating on Hyperplanes**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/2a3f862199883ceff5e3c74126f0c80770653e05)  
Influential Citation Count (372), SS-ID (2a3f862199883ceff5e3c74126f0c80770653e05)  

**ABSTRACT**  
We deal with embedding a large scale knowledge graph composed of entities and relations into a continuous vector space. TransE is a promising method proposed recently, which is very efficient while achieving state-of-the-art predictive performance. We discuss some mapping properties of relations which should be considered in embedding, such as reflexive, one-to-many, many-to-one, and many-to-many. We note that TransE does not do well in dealing with these properties. Some complex models are capable of preserving these mapping properties but sacrifice efficiency in the process. To make a good trade-off between model capacity and efficiency, in this paper we propose TransH which models a relation as a hyperplane together with a translation operation on it. In this way, we can well preserve the above mapping properties of relations with almost the same model complexity of TransE. Additionally, as a practical knowledge graph is often far from completed, how to construct negative examples to reduce false negative labels in training is very important. Utilizing the one-to-many/many-to-one mapping property of a relation, we propose a simple trick to reduce the possibility of false negative labeling. We conduct extensive experiments on link prediction, triplet classification and fact extraction on benchmark datasets like WordNet and Freebase. Experiments show TransH delivers significant improvements over TransE on predictive accuracy with comparable capability to scale up.

{{< /ci-details >}}

{{< ci-details summary="Learning from labeled and unlabeled data with label propagation (Xiaojin Zhu et al., 2002)">}}

Xiaojin Zhu, Zoubin Ghahramani. (2002)  
**Learning from labeled and unlabeled data with label propagation**  
  
[Paper Link](https://www.semanticscholar.org/paper/2a4ca461fa847e8433bab67e7bfe4620371c1f77)  
Influential Citation Count (158), SS-ID (2a4ca461fa847e8433bab67e7bfe4620371c1f77)  

**ABSTRACT**  
We investigate the use of unlabeled data to help labeled data in cl ssification. We propose a simple iterative algorithm, label pro pagation, to propagate labels through the dataset along high density are as d fined by unlabeled data. We analyze the algorithm, show its solution , and its connection to several other algorithms. We also show how to lear n p ameters by minimum spanning tree heuristic and entropy minimiz ation, and the algorithm’s ability to perform feature selection. Expe riment results are promising.

{{< /ci-details >}}

{{< ci-details summary="GraphRNN: A Deep Generative Model for Graphs (Jiaxuan You et al., 2018)">}}

Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, J. Leskovec. (2018)  
**GraphRNN: A Deep Generative Model for Graphs**  
ICML 2018  
[Paper Link](https://www.semanticscholar.org/paper/2afa9966c37b7747d954a4dcd61e986247783683)  
Influential Citation Count (18), SS-ID (2afa9966c37b7747d954a4dcd61e986247783683)  

**ABSTRACT**  
Modeling and generating graphs is fundamental for studying networks in biology, engineering, and social sciences. However, modeling complex distributions over graphs and then efficiently sampling from these distributions is challenging due to the non-unique, high-dimensional nature of graphs and the complex, non-local dependencies that exist between edges in a given graph. Here we propose GraphRNN, a deep autoregressive model that addresses the above challenges and approximates any distribution of graphs with minimal assumptions about their structure. GraphRNN learns to generate graphs by training on a representative set of graphs and decomposes the graph generation process into a sequence of node and edge formations, conditioned on the graph structure generated so far.  In order to quantitatively evaluate the performance of GraphRNN, we introduce a benchmark suite of datasets, baselines and novel evaluation metrics based on Maximum Mean Discrepancy, which measure distances between sets of graphs. Our experiments show that GraphRNN significantly outperforms all baselines, learning to generate diverse graphs that match the structural characteristics of a target set, while also scaling to graphs 50 times larger than previous deep models.

{{< /ci-details >}}

{{< ci-details summary="Recommending Co-authorship via Network Embeddings and Feature Engineering: The case of National Research University Higher School of Economics (Ilya Makarov et al., 2018)">}}

Ilya Makarov, Olga Gerasimova, Pavel Sulimov, L. Zhukov. (2018)  
**Recommending Co-authorship via Network Embeddings and Feature Engineering: The case of National Research University Higher School of Economics**  
JCDL  
[Paper Link](https://www.semanticscholar.org/paper/2b9501a2f4dfe1341bb272f157791a89d712ffd6)  
Influential Citation Count (0), SS-ID (2b9501a2f4dfe1341bb272f157791a89d712ffd6)  

**ABSTRACT**  
Co-authorship networks contain hidden structural patterns of research collaboration. While some people may argue that the process of writing joint papers depends on mutual friendship, research interests, and university policy, we show that, given a temporal co-authorship network, one could predict the quality and quantity of future research publications. We are working on the comparison of existing graph embedding and feature engineering methods, presenting combined approach for constructing co-author recommender system formulated as link prediction problem. We also present a new link embedding operator improving the quality of link prediction base don embedding feature space. We evaluate our research on a single university publication dataset, providing meaningful interpretation of the obtained results.

{{< /ci-details >}}

{{< ci-details summary="Graph Visualization and Navigation in Information Visualization: A Survey (I. Herman et al., 2000)">}}

I. Herman, G. Melançon, M. S. Marshall. (2000)  
**Graph Visualization and Navigation in Information Visualization: A Survey**  
IEEE Trans. Vis. Comput. Graph.  
[Paper Link](https://www.semanticscholar.org/paper/2bbb5387adb3bd725069b1914609dc08c4ed8571)  
Influential Citation Count (76), SS-ID (2bbb5387adb3bd725069b1914609dc08c4ed8571)  

**ABSTRACT**  
This is a survey on graph visualization and navigation techniques, as used in information visualization. Graphs appear in numerous applications such as Web browsing, state-transition diagrams, and data structures. The ability to visualize and to navigate in these potentially large, abstract graphs is often a crucial part of an application. Information visualization has specific requirements, which means that this survey approaches the results of traditional graph drawing from a different perspective.

{{< /ci-details >}}

{{< ci-details summary="Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks (Jiwei Li et al., 2015)">}}

Jiwei Li, Alan Ritter, Dan Jurafsky. (2015)  
**Learning multi-faceted representations of individuals from heterogeneous evidence using neural networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/2dde8f7171522243870fef46a0c93e5bf7be45b9)  
Influential Citation Count (2), SS-ID (2dde8f7171522243870fef46a0c93e5bf7be45b9)  

**ABSTRACT**  
Inferring latent attributes of people online is an important social computing task, but requires integrating the many heterogeneous sources of information available on the web. We propose learning individual representations of people using neural nets to integrate rich linguistic and network evidence gathered from social media. The algorithm is able to combine diverse cues, such as the text a person writes, their attributes (e.g. gender, employer, education, location) and social relations to other people. We show that by integrating both textual and network evidence, these representations offer improved performance at four important tasks in social media inference on Twitter: predicting (1) gender, (2) occupation, (3) location, and (4) friendships for users. Our approach scales to large datasets and the learned representations can be used as general features in and have the potential to benefit a large number of downstream tasks including link prediction, community detection, or probabilistic reasoning over social networks.

{{< /ci-details >}}

{{< ci-details summary="Multi-View Unsupervised Feature Selection with Adaptive Similarity and View Weight (Chenping Hou et al., 2017)">}}

Chenping Hou, F. Nie, Hong Tao, Dong-yun Yi. (2017)  
**Multi-View Unsupervised Feature Selection with Adaptive Similarity and View Weight**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/2ea8d4cfb92d6354caa76a7070a3a5e053e1b066)  
Influential Citation Count (5), SS-ID (2ea8d4cfb92d6354caa76a7070a3a5e053e1b066)  

**ABSTRACT**  
With the advent of multi-view data, multi-view learning has become an important research direction in both machine learning and data mining. Considering the difficulty of obtaining labeled data in many real applications, we focus on the multi-view unsupervised feature selection problem. Traditional approaches all characterize the similarity by fixed and pre-defined graph Laplacian in each view separately and ignore the underlying common structures across different views. In this paper, we propose an algorithm named Multi-view Unsupervised Feature Selection with Adaptive Similarity and View Weight (ASVW) to overcome the above mentioned problems. Specifically, by leveraging the learning mechanism to characterize the common structures adaptively, we formulate the objective function by a common graph Laplacian across different views, together with the sparse <inline-formula><tex-math notation="LaTeX">$\ell _{2,p}$</tex-math> <alternatives><inline-graphic xlink:href="hou-ieq1-2681670.gif"/></alternatives></inline-formula>-norm constraint designed for feature selection. We develop an efficient algorithm to address the non-smooth minimization problem and prove that the algorithm will converge. To validate the effectiveness of ASVW, comparisons are made with some benchmark methods on real-world datasets. We also evaluate our method in the real sports action recognition task. The experimental results demonstrate the effectiveness of our proposed algorithm.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Attacks on Node Embeddings (Aleksandar Bojchevski et al., 2018)">}}

Aleksandar Bojchevski, Stephan Günnemann. (2018)  
**Adversarial Attacks on Node Embeddings**  
ICML 2019  
[Paper Link](https://www.semanticscholar.org/paper/2fae42932811ff9307b6b4a5059c2300f3587b53)  
Influential Citation Count (6), SS-ID (2fae42932811ff9307b6b4a5059c2300f3587b53)  

**ABSTRACT**  
The goal of network representation learning is to learn low-dimensional node embeddings that capture the graph structure and are useful for solving downstream tasks. However, despite the proliferation of such methods there is currently no study of their robustness to adversarial attacks. We provide the first adversarial vulnerability analysis on the widely used family of methods based on random walks. We derive efficient adversarial perturbations that poison the network structure and have a negative effect on both the quality of the embeddings and the downstream tasks. We further show that our attacks are transferable – they generalize to many models – and are successful even when the attacker has restricted actions.

{{< /ci-details >}}

{{< ci-details summary="InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization (Fan-Yun Sun et al., 2019)">}}

Fan-Yun Sun, Jordan Hoffmann, Jian Tang. (2019)  
**InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/2fb59ebe271d6b007bb0429c1701fd1004782d1b)  
Influential Citation Count (50), SS-ID (2fb59ebe271d6b007bb0429c1701fd1004782d1b)  

**ABSTRACT**  
This paper studies learning the representations of whole graphs in both unsupervised and semi-supervised scenarios. Graph-level representations are critical in a variety of real-world applications such as predicting the properties of molecules and community analysis in social networks. Traditional graph kernel based methods are simple, yet effective for obtaining fixed-length representations for graphs but they suffer from poor generalization due to hand-crafted designs. There are also some recent methods based on language models (e.g. graph2vec) but they tend to only consider certain substructures (e.g. subtrees) as graph representatives. Inspired by recent progress of unsupervised representation learning, in this paper we proposed a novel method called InfoGraph for learning graph-level representations. We maximize the mutual information between the graph-level representation and the representations of substructures of different scales (e.g., nodes, edges, triangles). By doing so, the graph-level representations encode aspects of the data that are shared across different scales of substructures. Furthermore, we further propose InfoGraph*, an extension of InfoGraph for semi-supervised scenarios. InfoGraph* maximizes the mutual information between unsupervised graph representations learned by InfoGraph and the representations learned by existing supervised methods. As a result, the supervised encoder learns from unlabeled data while preserving the latent semantic space favored by the current supervised task. Experimental results on the tasks of graph classification and molecular property prediction show that InfoGraph is superior to state-of-the-art baselines and InfoGraph* can achieve performance competitive with state-of-the-art semi-supervised models.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Embedding: A Survey of Approaches and Applications (Quan Wang et al., 2017)">}}

Quan Wang, Zhendong Mao, Bin Wang, Li Guo. (2017)  
**Knowledge Graph Embedding: A Survey of Approaches and Applications**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/30321b036607a7936221235ea8ec7cf7c1627100)  
Influential Citation Count (100), SS-ID (30321b036607a7936221235ea8ec7cf7c1627100)  

**ABSTRACT**  
Knowledge graph (KG) embedding is to embed components of a KG including entities and relations into continuous vector spaces, so as to simplify the manipulation while preserving the inherent structure of the KG. It can benefit a variety of downstream tasks such as KG completion and relation extraction, and hence has quickly gained massive attention. In this article, we provide a systematic review of existing techniques, including not only the state-of-the-arts but also those with latest trends. Particularly, we make the review based on the type of information used in the embedding task. Techniques that conduct embedding using only facts observed in the KG are first introduced. We describe the overall framework, specific model design, typical training procedures, as well as pros and cons of such techniques. After that, we discuss techniques that further incorporate additional information besides facts. We focus specifically on the use of entity types, relation paths, textual descriptions, and logical rules. Finally, we briefly introduce how KG embedding can be applied to and benefit a wide variety of downstream tasks such as KG completion, relation extraction, question answering, and so forth.

{{< /ci-details >}}

{{< ci-details summary="Explainable Recommendation: A Survey and New Perspectives (Yongfeng Zhang et al., 2018)">}}

Yongfeng Zhang, Xu Chen. (2018)  
**Explainable Recommendation: A Survey and New Perspectives**  
Found. Trends Inf. Retr.  
[Paper Link](https://www.semanticscholar.org/paper/303b260d5bef9b5c87c868110ec429fe5ea934ad)  
Influential Citation Count (11), SS-ID (303b260d5bef9b5c87c868110ec429fe5ea934ad)  

**ABSTRACT**  
Explainable recommendation attempts to develop models that generate not only high-quality recommendations but also intuitive explanations. The explanations may either be post-hoc or directly come from an explainable model (also called interpretable or transparent model in some contexts). Explainable recommendation tries to address the problem of why: by providing explanations to users or system designers, it helps humans to understand why certain items are recommended by the algorithm, where the human can either be users or system designers. Explainable recommendation helps to improve the transparency, persuasiveness, effectiveness, trustworthiness, and satisfaction of recommendation systems. It also facilitates system designers for better system debugging. In recent years, a large number of explainable recommendation approaches -- especially model-based methods -- have been proposed and applied in real-world systems.  In this survey, we provide a comprehensive review for the explainable recommendation research. We first highlight the position of explainable recommendation in recommender system research by categorizing recommendation problems into the 5W, i.e., what, when, who, where, and why. We then conduct a comprehensive survey of explainable recommendation on three perspectives: 1) We provide a chronological research timeline of explainable recommendation. 2) We provide a two-dimensional taxonomy to classify existing explainable recommendation research. 3) We summarize how explainable recommendation applies to different recommendation tasks. We also devote a chapter to discuss the explanation perspectives in broader IR and AI/ML research. We end the survey by discussing potential future directions to promote the explainable recommendation research area and beyond.

{{< /ci-details >}}

{{< ci-details summary="Deep Learning on Graphs: A Survey (Ziwei Zhang et al., 2018)">}}

Ziwei Zhang, Peng Cui, Wenwu Zhu. (2018)  
**Deep Learning on Graphs: A Survey**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/30b38ca8151bbd5a5ff45bce94297d1248ff58b5)  
Influential Citation Count (23), SS-ID (30b38ca8151bbd5a5ff45bce94297d1248ff58b5)  

**ABSTRACT**  
Deep learning has been shown to be successful in a number of domains, ranging from acoustics, images, to natural language processing. However, applying deep learning to the ubiquitous graph data is non-trivial because of the unique characteristics of graphs. Recently, substantial research efforts have been devoted to applying deep learning methods to graphs, resulting in beneficial advances in graph analysis techniques. In this survey, we comprehensively review the different types of deep learning methods on graphs. We divide the existing methods into five categories based on their model architectures and training strategies: graph recurrent neural networks, graph convolutional networks, graph autoencoders, graph reinforcement learning, and graph adversarial methods. We then provide a comprehensive overview of these methods in a systematic manner mainly by following their development history. We also analyze the differences and compositions of different methods. Finally, we briefly outline the applications in which they have been used and discuss potential future research directions.

{{< /ci-details >}}

{{< ci-details summary="Visualizing Large-scale and High-dimensional Data (Jian Tang et al., 2016)">}}

Jian Tang, J. Liu, Ming Zhang, Q. Mei. (2016)  
**Visualizing Large-scale and High-dimensional Data**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/30d0f3ec0f80439f561f9912831ea7f3ccf8133c)  
Influential Citation Count (31), SS-ID (30d0f3ec0f80439f561f9912831ea7f3ccf8133c)  

**ABSTRACT**  
We study the problem of visualizing large-scale and high-dimensional data in a low-dimensional (typically 2D or 3D) space. Much success has been reported recently by techniques that first compute a similarity structure of the data points and then project them into a low-dimensional space with the structure preserved. These two steps suffer from considerable computational costs, preventing the state-of-the-art methods such as the t-SNE from scaling to large-scale and high-dimensional data (e.g., millions of data points and hundreds of dimensions). We propose the LargeVis, a technique that first constructs an accurately approximated K-nearest neighbor graph from the data and then layouts the graph in the low-dimensional space. Comparing to t-SNE, LargeVis significantly reduces the computational cost of the graph construction step and employs a principled probabilistic model for the visualization step, the objective of which can be effectively optimized through asynchronous stochastic gradient descent with a linear time complexity. The whole procedure thus easily scales to millions of high-dimensional data points. Experimental results on real-world data sets demonstrate that the LargeVis outperforms the state-of-the-art methods in both efficiency and effectiveness. The hyper-parameters of LargeVis are also much more stable over different data sets.

{{< /ci-details >}}

{{< ci-details summary="Complex network graph embedding method based on shortest path and MOEA/D for community detection (Weitong Zhang et al., 2020)">}}

Weitong Zhang, Ronghua Shang, L. Jiao. (2020)  
**Complex network graph embedding method based on shortest path and MOEA/D for community detection**  
Appl. Soft Comput.  
[Paper Link](https://www.semanticscholar.org/paper/322a2670adda4c595a6cae54801981c4fc9e005f)  
Influential Citation Count (0), SS-ID (322a2670adda4c595a6cae54801981c4fc9e005f)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Discriminative Embeddings of Latent Variable Models for Structured Data (H. Dai et al., 2016)">}}

H. Dai, Bo Dai, Le Song. (2016)  
**Discriminative Embeddings of Latent Variable Models for Structured Data**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/322cf9bcde458a45eaeca989a1eec92f7c6db984)  
Influential Citation Count (51), SS-ID (322cf9bcde458a45eaeca989a1eec92f7c6db984)  

**ABSTRACT**  
Kernel classifiers and regressors designed for structured data, such as sequences, trees and graphs, have significantly advanced a number of interdisciplinary areas such as computational biology and drug design. Typically, kernels are designed beforehand for a data type which either exploit statistics of the structures or make use of probabilistic generative models, and then a discriminative classifier is learned based on the kernels via convex optimization. However, such an elegant two-stage approach also limited kernel methods from scaling up to millions of data points, and exploiting discriminative information to learn feature representations.    We propose, structure2vec, an effective and scalable approach for structured data representation based on the idea of embedding latent variable models into feature spaces, and learning such feature spaces using discriminative information. Interestingly, structure2vec extracts features by performing a sequence of function mappings in a way similar to graphical model inference procedures, such as mean field and belief propagation. In applications involving millions of data points, we showed that structure2vec runs 2 times faster, produces models which are 10, 000 times smaller, while at the same time achieving the state-of-the-art predictive performance.

{{< /ci-details >}}

{{< ci-details summary="Learning to Cluster Faces on an Affinity Graph (Lei Yang et al., 2019)">}}

Lei Yang, Xiaohang Zhan, Dapeng Chen, Junjie Yan, Chen Change Loy, Dahua Lin. (2019)  
**Learning to Cluster Faces on an Affinity Graph**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/32873f6111963607d3f768f4685fe8137fdd1253)  
Influential Citation Count (16), SS-ID (32873f6111963607d3f768f4685fe8137fdd1253)  

**ABSTRACT**  
Face recognition sees remarkable progress in recent years, and its performance has reached a very high level. Taking it to a next level requires substantially larger data, which would involve prohibitive annotation cost. Hence, exploiting unlabeled data becomes an appealing alternative. Recent works have shown that clustering unlabeled faces is a promising approach, often leading to notable performance gains. Yet, how to effectively cluster, especially on a large-scale (i.e. million-level or above) dataset, remains an open question. A key challenge lies in the complex variations of cluster patterns, which make it difficult for conventional clustering methods to meet the needed accuracy. This work explores a novel approach, namely, learning to cluster instead of relying on hand-crafted criteria. Specifically, we propose a framework based on graph convolutional network, which combines a detection and a segmentation module to pinpoint face clusters. Experiments show that our method yields significantly more accurate face clusters, which, as a result, also lead to further performance gain in face recognition.

{{< /ci-details >}}

{{< ci-details summary="Large-scale structural and textual similarity-based mining of knowledge graph to predict drug-drug interactions (I. Abdelaziz et al., 2017)">}}

I. Abdelaziz, Achille Fokoue, O. Hassanzadeh, Ping Zhang, Mohammad Sadoghi. (2017)  
**Large-scale structural and textual similarity-based mining of knowledge graph to predict drug-drug interactions**  
J. Web Semant.  
[Paper Link](https://www.semanticscholar.org/paper/328aa2ad73e4aa715fecc1cd41c82f69c337562a)  
Influential Citation Count (3), SS-ID (328aa2ad73e4aa715fecc1cd41c82f69c337562a)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Adversarial Attack and Defense on Graph Data: A Survey (Lichao Sun et al., 2018)">}}

Lichao Sun, Yingtong Dou, Ji Wang, Philip S. Yu, B. Li. (2018)  
**Adversarial Attack and Defense on Graph Data: A Survey**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/32fc920df7bb39acc34a9284e62a19438934f2d8)  
Influential Citation Count (8), SS-ID (32fc920df7bb39acc34a9284e62a19438934f2d8)  

**ABSTRACT**  
Deep neural networks (DNNs) have been widely applied to various applications including image classification, text generation, audio recognition, and graph data analysis. However, recent studies have shown that DNNs are vulnerable to adversarial attacks. Though there are several works studying adversarial attack and defense strategies on domains such as images and natural language processing, it is still difficult to directly transfer the learned knowledge to graph structure data due to its representation challenges. Given the importance of graph analysis, an increasing number of works start to analyze the robustness of machine learning models on graph data. Nevertheless, current studies considering adversarial behaviors on graph data usually focus on specific types of attacks with certain assumptions. In addition, each work proposes its own mathematical formulation which makes the comparison among different methods difficult. Therefore, in this paper, we aim to survey existing adversarial learning strategies on graph data and first provide a unified formulation for adversarial learning on graph data which covers most adversarial learning studies on graph. Moreover, we also compare different attacks and defenses on graph data and discuss their corresponding contributions and limitations. In this work, we systemically organize the considered works based on the features of each topic. This survey not only serves as a reference for the research community, but also brings a clear image researchers outside this research domain. Besides, we also create an online resource and keep updating the relevant papers during the last two years. More details of the comparisons of various studies based on this survey are open-sourced at this https URL.

{{< /ci-details >}}

{{< ci-details summary="Incorporate Group Information to Enhance Network Embedding (Jifan Chen et al., 2016)">}}

Jifan Chen, Qi Zhang, Xuanjing Huang. (2016)  
**Incorporate Group Information to Enhance Network Embedding**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/332ec914469af4ecbc4ada0631773febc030406e)  
Influential Citation Count (4), SS-ID (332ec914469af4ecbc4ada0631773febc030406e)  

**ABSTRACT**  
The problem of representing large-scale networks with low-dimensional vectors has received considerable attention in recent years. Except the networks that include only vertices and edges, a variety of networks contain information about groups or communities. For example, on Facebook, in addition to users and the follower-followee relations between them, users can also create and join groups. However, previous studies have rarely utilized this valuable information to generate embeddings of vertices. In this paper, we investigate a novel method for learning the network embeddings with valuable group information for large-scale networks. The proposed methods take both the inner structures of the groups and the information across groups into consideration. Experimental results demonstrate that the embeddings generated by the proposed methods significantly outperform state-of-the-art network embedding methods on two different scale real-world network

{{< /ci-details >}}

{{< ci-details summary="Artificial Intelligence Scientific Documentation Dataset for Recommender Systems (F. Ortega et al., 2018)">}}

F. Ortega, J. Bobadilla, A. Gutiérrez, R. Hurtado, Xin Li. (2018)  
**Artificial Intelligence Scientific Documentation Dataset for Recommender Systems**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/333037df391e5e82e67fbd75be9e5a98dd2d73eb)  
Influential Citation Count (1), SS-ID (333037df391e5e82e67fbd75be9e5a98dd2d73eb)  

**ABSTRACT**  
The existing scientific documentation-based recommender systems focus on exploiting the citations and references information included in each research paper and also the lists of co-authors. In this way, it can be addressed the recommendation of related papers and even related authors. The approach we propose is original because instead of using each paper citations and co-authors, we relate each of the papers with their main research topics. This approach provides a semantic level superior to that currently used, which allows us to obtain useful results. We can use collaborative filtering recommender systems to recommend research topics related to each paper and also to recommend papers related to each research topic. In order to face this innovative proposal, we have solved a series of challenges that allow us to offer various resources and results in the paper. Our main contributions are: 1) making a data mining of scientific documentation; 2) creating and publishing an open database containing the data mining results; 3) extracting the research topics from the available scientific documentation; 4) creating and publishing a recommender system data set obtained from the database and the research topics; 5) testing the data set through a complete set of collaborative filtering methods and quality measures; and 6) selecting and showing the best methods and results, obtained using the open data set, in the context of scientific documentation recommendations. Results of the paper show the suitability of the provided data set in collaborative filtering processes, as well as the superiority of the model-based methods to face scientific documentation recommendations.

{{< /ci-details >}}

{{< ci-details summary="Graph Attention Networks (Petar Velickovic et al., 2017)">}}

Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, P. Lio’, Yoshua Bengio. (2017)  
**Graph Attention Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/33998aff64ce51df8dee45989cdca4b6b1329ec4)  
Influential Citation Count (1371), SS-ID (33998aff64ce51df8dee45989cdca4b6b1329ec4)  

**ABSTRACT**  
We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

{{< /ci-details >}}

{{< ci-details summary="Identification of pathways associated with chemosensitivity through network embedding (Sheng Wang et al., 2019)">}}

Sheng Wang, Edward W. Huang, J. Cairns, Jian Peng, Liewei Wang, S. Sinha. (2019)  
**Identification of pathways associated with chemosensitivity through network embedding**  
PLoS Comput. Biol.  
[Paper Link](https://www.semanticscholar.org/paper/33f7bc1fa413a47727d4cd741ffe1e7f95030604)  
Influential Citation Count (0), SS-ID (33f7bc1fa413a47727d4cd741ffe1e7f95030604)  

**ABSTRACT**  
Basal gene expression levels have been shown to be predictive of cellular response to cytotoxic treatments. However, such analyses do not fully reveal complex genotype- phenotype relationships, which are partly encoded in highly interconnected molecular networks. Biological pathways provide a complementary way of understanding drug response variation among individuals. In this study, we integrate chemosensitivity data from a large-scale pharmacogenomics study with basal gene expression data from the CCLE project and prior knowledge of molecular networks to identify specific pathways mediating chemical response. We first develop a computational method called PACER, which ranks pathways for enrichment in a given set of genes using a novel network embedding method. It examines a molecular network that encodes known gene-gene as well as gene-pathway relationships, and determines a vector representation of each gene and pathway in the same low-dimensional vector space. The relevance of a pathway to the given gene set is then captured by the similarity between the pathway vector and gene vectors. To apply this approach to chemosensitivity data, we identify genes whose basal expression levels in a panel of cell lines are correlated with cytotoxic response to a compound, and then rank pathways for relevance to these response-correlated genes using PACER. Extensive evaluation of this approach on benchmarks constructed from databases of compound target genes and large collections of drug response signatures demonstrates its advantages in identifying compound-pathway associations compared to existing statistical methods of pathway enrichment analysis. The associations identified by PACER can serve as testable hypotheses on chemosensitivity pathways and help further study the mechanisms of action of specific cytotoxic drugs. More broadly, PACER represents a novel technique of identifying enriched properties of any gene set of interest while also taking into account networks of known gene-gene relationships and interactions.

{{< /ci-details >}}

{{< ci-details summary="Graph Neural Networks with Generated Parameters for Relation Extraction (Hao Zhu et al., 2019)">}}

Hao Zhu, Yankai Lin, Zhiyuan Liu, Jie Fu, Tat-Seng Chua, Maosong Sun. (2019)  
**Graph Neural Networks with Generated Parameters for Relation Extraction**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/352ac73b7d92afa915c06026a4336927d550cec3)  
Influential Citation Count (4), SS-ID (352ac73b7d92afa915c06026a4336927d550cec3)  

**ABSTRACT**  
In this paper, we propose a novel graph neural network with generated parameters (GP-GNNs). The parameters in the propagation module, i.e. the transition matrices used in message passing procedure, are produced by a generator taking natural language sentences as inputs. We verify GP-GNNs in relation extraction from text, both on bag- and instance-settings. Experimental results on a human-annotated dataset and two distantly supervised datasets show that multi-hop reasoning mechanism yields significant improvements. We also perform a qualitative analysis to demonstrate that our model could discover more accurate relations by multi-hop relational reasoning.

{{< /ci-details >}}

{{< ci-details summary="A global geometric framework for nonlinear dimensionality reduction. (J. Tenenbaum et al., 2000)">}}

J. Tenenbaum, V. De Silva, J. Langford. (2000)  
**A global geometric framework for nonlinear dimensionality reduction.**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/3537fcd0ff99a3b3cb3d279012df826358420556)  
Influential Citation Count (1143), SS-ID (3537fcd0ff99a3b3cb3d279012df826358420556)  

**ABSTRACT**  
Scientists working with large volumes of high-dimensional data, such as global climate patterns, stellar spectra, or human gene distributions, regularly confront the problem of dimensionality reduction: finding meaningful low-dimensional structures hidden in their high-dimensional observations. The human brain confronts the same problem in everyday perception, extracting from its high-dimensional sensory inputs-30,000 auditory nerve fibers or 10(6) optic nerve fibers-a manageably small number of perceptually relevant features. Here we describe an approach to solving dimensionality reduction problems that uses easily measured local metric information to learn the underlying global geometry of a data set. Unlike classical techniques such as principal component analysis (PCA) and multidimensional scaling (MDS), our approach is capable of discovering the nonlinear degrees of freedom that underlie complex natural observations, such as human handwriting or images of a face under different viewing conditions. In contrast to previous algorithms for nonlinear dimensionality reduction, ours efficiently computes a globally optimal solution, and, for an important class of data manifolds, is guaranteed to converge asymptotically to the true structure.

{{< /ci-details >}}

{{< ci-details summary="Heterogeneous Information Network Embedding with Convolutional Graph Attention Networks (Meng Cao et al., 2020)">}}

Meng Cao, Xiying Ma, Kai Zhu, Ming Xu, Chong-Jun Wang. (2020)  
**Heterogeneous Information Network Embedding with Convolutional Graph Attention Networks**  
2020 International Joint Conference on Neural Networks (IJCNN)  
[Paper Link](https://www.semanticscholar.org/paper/3569199f440cc0178d5522644266c4b9b443e8ce)  
Influential Citation Count (0), SS-ID (3569199f440cc0178d5522644266c4b9b443e8ce)  

**ABSTRACT**  
Heterogeneous Information Networks (HINs) are prevalent in our daily life, such as social networks and bibliography networks, which contain multiple types of nodes and links. Heterogeneous information network embedding is an effective HIN analysis method, it aims at projecting network elements into a lower-dimensional vector space for further machine learning related evaluations, such as node classification, node clustering, and so on. However, existing HIN embedding methods mainly focus on extracting the semantic-related information or close neighboring relations, while the high-level proximity of the network is also important but not preserved. To address the problem, in this paper we propose CGAT, a semi-supervised heterogeneous information network embedding method. We optimize the graph attention network by adding additional convolution layers, thereby we can extract multiple types of semantics and preserve high-level information in HIN embedding at the same time. Also, we utilize label information in HINs for semi-supervised training to better obtain the model parameters and HIN embeddings. Experimental results on real-world datasets demonstrate the effectiveness and efficiency of the proposed model.

{{< /ci-details >}}

{{< ci-details summary="Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network (Sunil Kumar Sahu et al., 2019)">}}

Sunil Kumar Sahu, Fenia Christopoulou, Makoto Miwa, S. Ananiadou. (2019)  
**Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/358ca777d9992bdc06fdcc1940e3b18a8da68878)  
Influential Citation Count (13), SS-ID (358ca777d9992bdc06fdcc1940e3b18a8da68878)  

**ABSTRACT**  
Inter-sentence relation extraction deals with a number of complex semantic relationships in documents, which require local, non-local, syntactic and semantic dependencies. Existing methods do not fully exploit such dependencies. We present a novel inter-sentence relation extraction model that builds a labelled edge graph convolutional neural network model on a document-level graph. The graph is constructed using various inter- and intra-sentence dependencies to capture local and non-local dependency information. In order to predict the relation of an entity pair, we utilise multi-instance learning with bi-affine pairwise scoring. Experimental results show that our model achieves comparable performance to the state-of-the-art neural models on two biochemistry datasets. Our analysis shows that all the types in the graph are effective for inter-sentence relation extraction.

{{< /ci-details >}}

{{< ci-details summary="HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival (Huiting Hong et al., 2020)">}}

Huiting Hong, Yucheng Lin, Xiaoqing Yang, Zang Li, Kun Fu, Zheng Wang, X. Qie, Jieping Ye. (2020)  
**HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/364b6a10a827a6ba994d17baab2b2a2f1271dc29)  
Influential Citation Count (1), SS-ID (364b6a10a827a6ba994d17baab2b2a2f1271dc29)  

**ABSTRACT**  
The estimated time of arrival (ETA) is a critical task in the intelligent transportation system, which involves the spatiotemporal data. Despite a significant amount of prior efforts have been made to design efficient and accurate systems for ETA task, few of them take structural graph data into account, much less the heterogeneous information network. In this paper, we propose HetETA to leverage heterogeneous information graph in ETA task. Specifically, we translate the road map into a multi-relational network and introduce a vehicle-trajectories based network to jointly consider the traffic behavior pattern. Moreover, we employ three components to model temporal information from recent periods, daily periods and weekly periods respectively. Each component comprises temporal convolutions and graph convolutions to learn representations of the spatiotemporal heterogeneous information for ETA task. Experiments on large-scale datasets illustrate the effectiveness of the proposed HetETA beyond the state-of-the-art methods, and show the importance of representation learning of heterogeneous information networks for ETA task.

{{< /ci-details >}}

{{< ci-details summary="Graph Signal Processing: Overview, Challenges, and Applications (Antonio Ortega et al., 2017)">}}

Antonio Ortega, P. Frossard, J. Kovacevic, J. Moura, P. Vandergheynst. (2017)  
**Graph Signal Processing: Overview, Challenges, and Applications**  
Proceedings of the IEEE  
[Paper Link](https://www.semanticscholar.org/paper/36d442f59c61ea2912d227c24dee76778c546b0a)  
Influential Citation Count (87), SS-ID (36d442f59c61ea2912d227c24dee76778c546b0a)  

**ABSTRACT**  
Research in graph signal processing (GSP) aims to develop tools for processing data defined on irregular graph domains. In this paper, we first provide an overview of core ideas in GSP and their connection to conventional digital signal processing, along with a brief historical perspective to highlight how concepts recently developed in GSP build on top of prior research in other areas. We then summarize recent advances in developing basic GSP tools, including methods for sampling, filtering, or graph learning. Next, we review progress in several application areas using GSP, including processing and analysis of sensor network data, biological data, and applications to image processing and machine learning.

{{< /ci-details >}}

{{< ci-details summary="Detecting Changes of Functional Connectivity by Dynamic Graph Embedding Learning (Yi Lin et al., 2020)">}}

Yi Lin, J. Hou, P. Laurienti, Guorong Wu. (2020)  
**Detecting Changes of Functional Connectivity by Dynamic Graph Embedding Learning**  
MICCAI  
[Paper Link](https://www.semanticscholar.org/paper/36e12c48cf33d7097386fc93f091dd22d7349535)  
Influential Citation Count (0), SS-ID (36e12c48cf33d7097386fc93f091dd22d7349535)  

**ABSTRACT**  
Our current understandings reach the unanimous consensus that the brain functions and cognitive states are dynamically changing even in the resting state rather than remaining at a single constant state. Due to the low signal-to-noise ratio and high vertex-time dependency in BOLD (blood oxygen level dependent) signals, however, it is challenging to detect the dynamic behavior in connectivity without requiring prior knowledge of the experimental design. Like the Fourier bases in signal processing, each brain network can be summarized by a set of harmonic bases (Eigensystem) which are derived from its latent Laplacian matrix. In this regard, we propose to establish a subject-specific spectrum domain, where the learned orthogonal harmonic-Fourier bases allow us to detect the changes of functional connectivity more accurately than using the BOLD signals in an arbitrary sliding window. To do so, we first present a novel dynamic graph learning method to simultaneously estimate the intrinsic BOLD signals and learn the joint harmonic-Fourier bases for the underlying functional connectivity network. Then, we project the BOLD signals to the spectrum domain spanned by learned network harmonic and Fourier bases, forming the new system-level fluctuation patterns, called dynamic graph embeddings. We employ the classic clustering approach to identify the changes of functional connectivity using the novel dynamic graph embedding vectors. Our method has been evaluated on working memory task-based fMRI dataset and comparisons with state-of-the-art methods, where our joint harmonic-Fourier bases achieves higher accuracy in detecting multiple cognitive states.

{{< /ci-details >}}

{{< ci-details summary="node2vec: Scalable Feature Learning for Networks (Aditya Grover et al., 2016)">}}

Aditya Grover, J. Leskovec. (2016)  
**node2vec: Scalable Feature Learning for Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/36ee2c8bd605afd48035d15fdc6b8c8842363376)  
Influential Citation Count (1119), SS-ID (36ee2c8bd605afd48035d15fdc6b8c8842363376)  

**ABSTRACT**  
Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

{{< /ci-details >}}

{{< ci-details summary="Semi-Supervised Classification with Graph Convolutional Networks (Thomas Kipf et al., 2016)">}}

Thomas Kipf, M. Welling. (2016)  
**Semi-Supervised Classification with Graph Convolutional Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/36eff562f65125511b5dfab68ce7f7a943c27478)  
Influential Citation Count (3139), SS-ID (36eff562f65125511b5dfab68ce7f7a943c27478)  

**ABSTRACT**  
We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs. We motivate the choice of our convolutional architecture via a localized first-order approximation of spectral graph convolutions. Our model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes. In a number of experiments on citation networks and on a knowledge graph dataset we demonstrate that our approach outperforms related methods by a significant margin.

{{< /ci-details >}}

{{< ci-details summary="Graph Embedding Techniques, Applications, and Performance: A Survey (Palash Goyal et al., 2017)">}}

Palash Goyal, Emilio Ferrara. (2017)  
**Graph Embedding Techniques, Applications, and Performance: A Survey**  
Knowl. Based Syst.  
[Paper Link](https://www.semanticscholar.org/paper/374b4409f6a1d2d853af31e329f025da239d375f)  
Influential Citation Count (40), SS-ID (374b4409f6a1d2d853af31e329f025da239d375f)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Walklets: Multiscale Graph Embeddings for Interpretable Network Classification (Bryan Perozzi et al., 2016)">}}

Bryan Perozzi, Vivek Kulkarni, S. Skiena. (2016)  
**Walklets: Multiscale Graph Embeddings for Interpretable Network Classification**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/37cf46e45777e67676f80c9110bed675a9840590)  
Influential Citation Count (11), SS-ID (37cf46e45777e67676f80c9110bed675a9840590)  

**ABSTRACT**  
We present Walklets, a novel approach for learning multiscale representations of vertices in a network. These representations clearly encode multiscale vertex relationships in a continuous vector space suitable for multi-label classification problems. Unlike previous work, the latent features generated using Walklets are analytically derivable, and human interpretable.  Walklets uses the offsets between vertices observed in a random walk to learn a series of latent representations, each which captures successively larger relationships. This variety of dependency information allows the same representation strategy to model phenomenon which occur at different scales.  We demonstrate Walklets' latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that Walklets outperforms new methods based on neural matrix factorization, and can scale to graphs with millions of vertices and edges.

{{< /ci-details >}}

{{< ci-details summary="Graph based Neural Networks for Event Factuality Prediction using Syntactic and Semantic Structures (Amir Pouran Ben Veyseh et al., 2019)">}}

Amir Pouran Ben Veyseh, T. Nguyen, D. Dou. (2019)  
**Graph based Neural Networks for Event Factuality Prediction using Syntactic and Semantic Structures**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/37d8e064df4e921bcb7b80b6d4c3ee7488a027e0)  
Influential Citation Count (1), SS-ID (37d8e064df4e921bcb7b80b6d4c3ee7488a027e0)  

**ABSTRACT**  
Event factuality prediction (EFP) is the task of assessing the degree to which an event mentioned in a sentence has happened. For this task, both syntactic and semantic information are crucial to identify the important context words. The previous work for EFP has only combined these information in a simple way that cannot fully exploit their coordination. In this work, we introduce a novel graph-based neural network for EFP that can integrate the semantic and syntactic information more effectively. Our experiments demonstrate the advantage of the proposed model for EFP.

{{< /ci-details >}}

{{< ci-details summary="Scalable Graph Embedding for Asymmetric Proximity (Chang Zhou et al., 2017)">}}

Chang Zhou, Yuqiong Liu, Xiaofei Liu, Zhongyi Liu, Jun Gao. (2017)  
**Scalable Graph Embedding for Asymmetric Proximity**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/390bc9d41c1169d316accd993fc715b8ed17f269)  
Influential Citation Count (28), SS-ID (390bc9d41c1169d316accd993fc715b8ed17f269)  

**ABSTRACT**  
Graph Embedding methods are aimed at mapping each vertex into a low dimensional vector space, which preserves certain structural relationships among the vertices in the original graph. Recently, several works have been proposed to learn embeddings based on sampled paths from the graph, e.g., DeepWalk, Line, Node2Vec. However, their methods only preserve symmetric proximities, which could be insufficient in many applications, even the underlying graph is undirected. Besides, they lack of theoretical analysis of what exactly the relationships they preserve in their embedding space. In this paper, we propose an asymmetric proximity preserving (APP) graph embedding method via random walk with restart, which captures both asymmetric and high-order similarities between node pairs. We give theoretical analysis that our method implicitly preserves the Rooted PageRank score for any two vertices. We conduct extensive experiments on tasks of link prediction and node recommendation on open source datasets, as well as online recommendation services in Alibaba Group, in which the training graph has over 290 million vertices and 18 billion edges, showing our method to be highly scalable and effective.

{{< /ci-details >}}

{{< ci-details summary="GOMES: A group-aware multi-view fusion approach towards real-world image clustering (Zhe Xue et al., 2015)">}}

Zhe Xue, Guorong Li, Shuhui Wang, Chunjie Zhang, W. Zhang, Qingming Huang. (2015)  
**GOMES: A group-aware multi-view fusion approach towards real-world image clustering**  
2015 IEEE International Conference on Multimedia and Expo (ICME)  
[Paper Link](https://www.semanticscholar.org/paper/394ba1a52e3cd59974f4277ef1ae987bc3500870)  
Influential Citation Count (1), SS-ID (394ba1a52e3cd59974f4277ef1ae987bc3500870)  

**ABSTRACT**  
Different features describe different views of visual appearance, multi-view based methods can integrate the information contained in each view and improve the image clustering performance. Most of the existing methods assume that the importance of one type of feature is the same to all the data. However, the visual appearance of images are different, so the description abilities of different features vary with different images. To solve this problem, we propose a group-aware multi-view fusion approach. Images are partitioned into groups which consist of several images sharing similar visual appearance. We assign different weights to evaluate the pairwise similarity between different groups. Then the clustering results and the fusion weights are learned by an iterative optimization procedure. Experimental results indicate that our approach achieves promising clustering performance compared with the existing methods.

{{< /ci-details >}}

{{< ci-details summary="The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains (D. Shuman et al., 2012)">}}

D. Shuman, S. K. Narang, P. Frossard, Antonio Ortega, P. Vandergheynst. (2012)  
**The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains**  
IEEE Signal Processing Magazine  
[Paper Link](https://www.semanticscholar.org/paper/39e223e6b5a6f8727e9f60b8b7c7720dc40a5dbc)  
Influential Citation Count (361), SS-ID (39e223e6b5a6f8727e9f60b8b7c7720dc40a5dbc)  

**ABSTRACT**  
In applications such as social, energy, transportation, sensor, and neuronal networks, high-dimensional data naturally reside on the vertices of weighted graphs. The emerging field of signal processing on graphs merges algebraic and spectral graph theoretic concepts with computational harmonic analysis to process such signals on graphs. In this tutorial overview, we outline the main challenges of the area, discuss different ways to define graph spectral domains, which are the analogs to the classical frequency domain, and highlight the importance of incorporating the irregular structures of graph data domains when processing signals on graphs. We then review methods to generalize fundamental operations such as filtering, translation, modulation, dilation, and downsampling to the graph setting and survey the localized, multiscale transforms that have been proposed to efficiently extract information from high-dimensional data on graphs. We conclude with a brief discussion of open issues and possible extensions.

{{< /ci-details >}}

{{< ci-details summary="UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction (Leland McInnes et al., 2018)">}}

Leland McInnes, John Healy. (2018)  
**UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/3a288c63576fc385910cb5bc44eaea75b442e62e)  
Influential Citation Count (641), SS-ID (3a288c63576fc385910cb5bc44eaea75b442e62e)  

**ABSTRACT**  
UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning.

{{< /ci-details >}}

{{< ci-details summary="Out-of-sample Node Representation Learning for Heterogeneous Graph in Real-time Android Malware Detection (Yanfang Ye et al., 2019)">}}

Yanfang Ye, Shifu Hou, Lingwei Chen, Jingwei Lei, Wenqiang Wan, Jiabin Wang, Qi Xiong, Fudong Shao. (2019)  
**Out-of-sample Node Representation Learning for Heterogeneous Graph in Real-time Android Malware Detection**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/3adf4438aca17918622483593431c47fc5fd97cb)  
Influential Citation Count (2), SS-ID (3adf4438aca17918622483593431c47fc5fd97cb)  

**ABSTRACT**  
The increasingly sophisticated Android malware calls for new defensive techniques that are capable of protecting mobile users against novel threats. In this paper, we first extract the runtime Application Programming Interface (API) call sequences from Android apps, and then analyze higher-level semantic relations within the ecosystem to comprehensively characterize the apps. To model different types of entities (i.e., app, API, device, signature, affiliation) and rich relations among them, we present a structured heterogeneous graph (HG) for modeling. To efficiently classify nodes (e.g., apps) in the constructed HG, we propose the HG-Learning method to first obtain in-sample node embeddings and then learn representations of out-of-sample nodes without rerunning/adjusting HG embeddings at the first attempt. We later design a deep neural network classifier taking the learned HG representations as inputs for real-time Android malware detection. Comprehensive experiments on large-scale and real sample collections from Tencent Security Lab are performed to compare various baselines. Promising results demonstrate that our developed system AiDroid which integrates our proposed method outperforms others in real-time Android malware detection.

{{< /ci-details >}}

{{< ci-details summary="Context-aware citation recommendation (Qi He et al., 2010)">}}

Qi He, J. Pei, Daniel Kifer, P. Mitra, C. Lee Giles. (2010)  
**Context-aware citation recommendation**  
WWW '10  
[Paper Link](https://www.semanticscholar.org/paper/3c0312918ac9fea614abaa0732d83f3e76c16f7d)  
Influential Citation Count (32), SS-ID (3c0312918ac9fea614abaa0732d83f3e76c16f7d)  

**ABSTRACT**  
When you write papers, how many times do you want to make some citations at a place but you are not sure which papers to cite? Do you wish to have a recommendation system which can recommend a small number of good candidates for every place that you want to make some citations? In this paper, we present our initiative of building a context-aware citation recommendation system. High quality citation recommendation is challenging: not only should the citations recommended be relevant to the paper under composition, but also should match the local contexts of the places citations are made. Moreover, it is far from trivial to model how the topic of the whole paper and the contexts of the citation places should affect the selection and ranking of citations. To tackle the problem, we develop a context-aware approach. The core idea is to design a novel non-parametric probabilistic model which can measure the context-based relevance between a citation context and a document. Our approach can recommend citations for a context effectively. Moreover, it can recommend a set of citations for a paper with high quality. We implement a prototype system in CiteSeerX. An extensive empirical evaluation in the CiteSeerX digital library against many baselines demonstrates the effectiveness and the scalability of our approach.

{{< /ci-details >}}

{{< ci-details summary="LRBM: A Restricted Boltzmann Machine Based Approach for Representation Learning on Linked Data (Kang Li et al., 2014)">}}

Kang Li, Jing Gao, Suxin Guo, Nan Du, Xiaoyi Li, A. Zhang. (2014)  
**LRBM: A Restricted Boltzmann Machine Based Approach for Representation Learning on Linked Data**  
2014 IEEE International Conference on Data Mining  
[Paper Link](https://www.semanticscholar.org/paper/3ce947f68c2c4061736a8b4363ebf84f48c572c9)  
Influential Citation Count (3), SS-ID (3ce947f68c2c4061736a8b4363ebf84f48c572c9)  

**ABSTRACT**  
Linked data consist of both node attributes, e.g., Preferences, posts and degrees, and links which describe the connections between nodes. They have been widely used to represent various network systems, such as social networks, biological networks and etc. Knowledge discovery on linked data is of great importance to many real applications. One of the major challenges of learning linked data is how to effectively and efficiently extract useful information from both node attributes and links in linked data. Current studies on this topic either use selected topological statistics to represent network structures, or linearly map node attributes and network structures to a shared latent feature space. However, while approaches based on statistics may miss critical patterns in network structure, approaches based on linear mappings may not be sufficient to capture the non-linear characteristics of nodes and links. To handle the challenge, we propose, to our knowledge, the first deep learning method to learn from linked data. A restricted Boltzmann machine model named LRBM is developed for representation learning on linked data. In LRBM, we aim to extract the latent feature representation of each node from both node attributes and network structures, non-linearly map each pair of nodes to the links, and use hidden units to control the mapping. The details of how to adapt LRBM for link prediction and node classification on linked data have also been presented. In the experiments, we test the performance of LRBM as well as other baselines on link prediction and node classification. Overall, the extensive experimental evaluations confirm the effectiveness of the proposed LRBM model in mining linked data.

{{< /ci-details >}}

{{< ci-details summary="Revisiting Semi-Supervised Learning with Graph Embeddings (Zhilin Yang et al., 2016)">}}

Zhilin Yang, William W. Cohen, R. Salakhutdinov. (2016)  
**Revisiting Semi-Supervised Learning with Graph Embeddings**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/3d846cb01f6a975554035d2210b578ca61344b22)  
Influential Citation Count (178), SS-ID (3d846cb01f6a975554035d2210b578ca61344b22)  

**ABSTRACT**  
We present a semi-supervised learning framework based on graph embeddings. Given a graph between instances, we train an embedding for each instance to jointly predict the class label and the neighborhood context in the graph. We develop both transductive and inductive variants of our method. In the transductive variant of our method, the class labels are determined by both the learned embeddings and input feature vectors, while in the inductive variant, the embeddings are defined as a parametric function of the feature vectors, so predictions can be made on instances not seen during training. On a large and diverse set of benchmark tasks, including text classification, distantly supervised entity extraction, and entity classification, we show improved performance over many of the existing models.

{{< /ci-details >}}

{{< ci-details summary="Task-Oriented Genetic Activation for Large-Scale Complex Heterogeneous Graph Embedding (Zhuoren Jiang et al., 2020)">}}

Zhuoren Jiang, Zheng Gao, Jinjiong Lan, Hongxia Yang, Yao Lu, Xiaozhong Liu. (2020)  
**Task-Oriented Genetic Activation for Large-Scale Complex Heterogeneous Graph Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/3e98d1b468304ef11b6cceb07d6d6b94d36ef7bc)  
Influential Citation Count (1), SS-ID (3e98d1b468304ef11b6cceb07d6d6b94d36ef7bc)  

**ABSTRACT**  
The recent success of deep graph embedding innovates the graphical information characterization methodologies. However, in real-world applications, such a method still struggles with the challenges of heterogeneity, scalability, and multiplex. To address these challenges, in this study, we propose a novel solution, Genetic hEterogeneous gRaph eMbedding (GERM), which enables flexible and efficient task-driven vertex embedding in a complex heterogeneous graph. Unlike prior efforts for this track of studies, we employ a task-oriented genetic activation strategy to efficiently generate the “Edge Type Activated Vector” (ETAV) over the edge types in the graph. The generated ETAV can not only reduce the incompatible noise and navigate the heterogeneous graph random walk at the graph-schema level, but also activate an optimized subgraph for efficient representation learning. By revealing the correlation between the graph structure and task information, the model interpretability can be enhanced as well. Meanwhile, an activated heterogeneous skip-gram framework is proposed to encapsulate both topological and task-specific information of a given heterogeneous graph. Through extensive experiments on both scholarly and e-commerce datasets, we demonstrate the efficacy and scalability of the proposed methods via various search/recommendation tasks. GERM can significantly reduces the running time and remove expert-intervention without sacrificing the performance (or even modestly improve) by comparing with baselines.

{{< /ci-details >}}

{{< ci-details summary="Link Prediction Adversarial Attack (Jinyin Chen et al., 2018)">}}

Jinyin Chen, Ziqiang Shi, Yangyang Wu, Xuanheng Xu, Haibin Zheng. (2018)  
**Link Prediction Adversarial Attack**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/3ed0205a0c99302455fb15348c99ef0511ffab91)  
Influential Citation Count (2), SS-ID (3ed0205a0c99302455fb15348c99ef0511ffab91)  

**ABSTRACT**  
Deep neural network has shown remarkable performance in solving computer vision and some graph evolved tasks, such as node classification and link prediction. However, the vulnerability of deep model has also been revealed by carefully designed adversarial examples generated by various adversarial attack methods. With the wider application of deep model in complex network analysis, in this paper we define and formulate the link prediction adversarial attack problem and put forward a novel iterative gradient attack (IGA) based on the gradient information in trained graph auto-encoder (GAE). To our best knowledge, it is the first time link prediction adversarial attack problem is defined and attack method is brought up. Not surprisingly, GAE was easily fooled by adversarial network with only a few links perturbed on the clean network. By conducting comprehensive experiments on different real-world data sets, we can conclude that most deep model based and other state-of-art link prediction algorithms cannot escape the adversarial attack just like GAE. We can benefit the attack as an efficient privacy protection tool from link prediction unknown violation, on the other hand, link prediction attack can be a robustness evaluation metric for current link prediction algorithm in attack defensibility.

{{< /ci-details >}}

{{< ci-details summary="The Graph Neural Network Model (F. Scarselli et al., 2009)">}}

F. Scarselli, M. Gori, A. Tsoi, M. Hagenbuchner, G. Monfardini. (2009)  
**The Graph Neural Network Model**  
IEEE Transactions on Neural Networks  
[Paper Link](https://www.semanticscholar.org/paper/3efd851140aa28e95221b55fcc5659eea97b172d)  
Influential Citation Count (265), SS-ID (3efd851140aa28e95221b55fcc5659eea97b172d)  

**ABSTRACT**  
Many underlying relationships among data in several areas of science and engineering, e.g., computer vision, molecular chemistry, molecular biology, pattern recognition, and data mining, can be represented in terms of graphs. In this paper, we propose a new neural network model, called graph neural network (GNN) model, that extends existing neural network methods for processing the data represented in graph domains. This GNN model, which can directly process most of the practically useful types of graphs, e.g., acyclic, cyclic, directed, and undirected, implements a function tau(G,n) isin IRm that maps a graph G and one of its nodes n into an m-dimensional Euclidean space. A supervised learning algorithm is derived to estimate the parameters of the proposed GNN model. The computational cost of the proposed algorithm is also considered. Some experimental results are shown to validate the proposed learning algorithm, and to demonstrate its generalization capabilities.

{{< /ci-details >}}

{{< ci-details summary="Graph Clustering Based on Structural/Attribute Similarities (Yang Zhou et al., 2009)">}}

Yang Zhou, Hong Cheng, J. Yu. (2009)  
**Graph Clustering Based on Structural/Attribute Similarities**  
Proc. VLDB Endow.  
[Paper Link](https://www.semanticscholar.org/paper/3f397e7dab0e253e0859d18bb5711b5471c657fe)  
Influential Citation Count (58), SS-ID (3f397e7dab0e253e0859d18bb5711b5471c657fe)  

**ABSTRACT**  
The goal of graph clustering is to partition vertices in a large graph into different clusters based on various criteria such as vertex connectivity or neighborhood similarity. Graph clustering techniques are very useful for detecting densely connected groups in a large graph. Many existing graph clustering methods mainly focus on the topological structure for clustering, but largely ignore the vertex properties which are often heterogenous. In this paper, we propose a novel graph clustering algorithm, SA-Cluster, based on both structural and attribute similarities through a unified distance measure. Our method partitions a large graph associated with attributes into k clusters so that each cluster contains a densely connected subgraph with homogeneous attribute values. An effective method is proposed to automatically learn the degree of contributions of structural similarity and attribute similarity. Theoretical analysis is provided to show that SA-Cluster is converging. Extensive experimental results demonstrate the effectiveness of SA-Cluster through comparison with the state-of-the-art graph clustering and summarization methods.

{{< /ci-details >}}

{{< ci-details summary="Leveraging social media networks for classification (Lei Tang et al., 2011)">}}

Lei Tang, Huan Liu. (2011)  
**Leveraging social media networks for classification**  
Data Mining and Knowledge Discovery  
[Paper Link](https://www.semanticscholar.org/paper/3f9df5c77af49d5b1b19eac9b82cb430b50f482d)  
Influential Citation Count (26), SS-ID (3f9df5c77af49d5b1b19eac9b82cb430b50f482d)  

**ABSTRACT**  
Social media has reshaped the way in which people interact with each other. The rapid development of participatory web and social networking sites like YouTube, Twitter, and Facebook, also brings about many data mining opportunities and novel challenges. In particular, we focus on classification tasks with user interaction information in a social network. Networks in social media are heterogeneous, consisting of various relations. Since the relation-type information may not be available in social media, most existing approaches treat these inhomogeneous connections homogeneously, leading to an unsatisfactory classification performance. In order to handle the network heterogeneity, we propose the concept of social dimension to represent actors’ latent affiliations, and develop a classification framework based on that. The proposed framework, SocioDim, first extracts social dimensions based on the network structure to accurately capture prominent interaction patterns between actors, then learns a discriminative classifier to select relevant social dimensions. SocioDim, by differentiating different types of network connections, outperforms existing representative methods of classification in social media, and offers a simple yet effective approach to integrating two types of seemingly orthogonal information: the network of actors and their attributes.

{{< /ci-details >}}

{{< ci-details summary="Which way? Direction-Aware Attributed Graph Embedding (Zekarias T. Kefato et al., 2020)">}}

Zekarias T. Kefato, Nasrullah Sheikh, A. Montresor. (2020)  
**Which way? Direction-Aware Attributed Graph Embedding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/403cc71d267960ae325173868180aaafec3b4b9b)  
Influential Citation Count (0), SS-ID (403cc71d267960ae325173868180aaafec3b4b9b)  

**ABSTRACT**  
Graph embedding algorithms are used to efficiently represent (encode) a graph in a low-dimensional continuous vector space that preserves the most important properties of the graph. One aspect that is often overlooked is whether the graph is directed or not. Most studies ignore the directionality, so as to learn high-quality representations optimized for node classification. On the other hand, studies that capture directionality are usually effective on link prediction but do not perform well on other tasks. This preliminary study presents a novel text-enriched, direction-aware algorithm called DIAGRAM , based on a carefully designed multi-objective model to learn embeddings that preserve the direction of edges, textual features and graph context of nodes. As a result, our algorithm does not have to trade one property for another and jointly learns high-quality representations for multiple network analysis tasks. We empirically show that DIAGRAM significantly outperforms six state-of-the-art baselines, both direction-aware and oblivious ones,on link prediction and network reconstruction experiments using two popular datasets. It also achieves a comparable performance on node classification experiments against these baselines using the same datasets.

{{< /ci-details >}}

{{< ci-details summary="Multilevel graph embedding (Benjamin Quiring et al., 2020)">}}

Benjamin Quiring, P. Vassilevski. (2020)  
**Multilevel graph embedding**  
Numer. Linear Algebra Appl.  
[Paper Link](https://www.semanticscholar.org/paper/412b0cf8d4a39634cb02235a1333ddf3bf792732)  
Influential Citation Count (0), SS-ID (412b0cf8d4a39634cb02235a1333ddf3bf792732)  

**ABSTRACT**  
The goal of the present paper is the design of embeddings of a general sparse graph into a set of points in ℝd for appropriate d ≥ 2. The embeddings that we are looking at aim to keep vertices that are grouped in communities together and keep the rest apart. To achieve this property, we utilize coarsening that respects possible community structures of the given graph. We employ a hierarchical multilevel coarsening approach that identifies communities (strongly connected groups of vertices) at every level. The multilevel strategy allows any given (presumably expensive) graph embedding algorithm to be made into a more scalable (and faster) algorithm. We demonstrate the presented approach on a number of given embedding algorithms and large‐scale graphs and achieve speed‐up over the methods in a recent paper.

{{< /ci-details >}}

{{< ci-details summary="Exploiting Cliques for Granular Computing-based Graph Classification (L. Baldini et al., 2020)">}}

L. Baldini, A. Martino, A. Rizzi. (2020)  
**Exploiting Cliques for Granular Computing-based Graph Classification**  
2020 International Joint Conference on Neural Networks (IJCNN)  
[Paper Link](https://www.semanticscholar.org/paper/414c510f5b932d86e68b81044a8a18a46a46e007)  
Influential Citation Count (0), SS-ID (414c510f5b932d86e68b81044a8a18a46a46e007)  

**ABSTRACT**  
The most fascinating aspect of graphs is their ability to encode the information contained in the inner structural organization between its constituting elements. Learning from graphs belong to the so-called Structural Pattern Recognition, from which Graph Embedding emerged as a successful method for processing graphs by evaluating their dissimilarity in a suitable geometric space. In this paper, we investigate the possibility to perform the embedding into a geometric space by leveraging to peculiar constituent graph substructures extracted from training set, namely the maximal cliques, and providing the performances obtained under three main aspects concerning classification capabilities, running times and model complexity. Thanks to a Granular Computing approach, the employed methodology can be seen as a powerful framework able to synthesize models suitable to be interpreted by field-experts, pushing the boundary towards new frontiers in the field of explainable AI and knowledge discovery also in big data contexts.

{{< /ci-details >}}

{{< ci-details summary="On the properties of small-world network models (A. Barrat et al., 1999)">}}

A. Barrat, M. Weigt. (1999)  
**On the properties of small-world network models**  
  
[Paper Link](https://www.semanticscholar.org/paper/421a56ce3c6627da87efa298d90d15f90642272d)  
Influential Citation Count (21), SS-ID (421a56ce3c6627da87efa298d90d15f90642272d)  

**ABSTRACT**  
Abstract:We study the small-world networks recently introduced by Watts and Strogatz [Nature 393, 440 (1998)], using analytical as well as numerical tools. We characterize the geometrical properties resulting from the coexistence of a local structure and random long-range connections, and we examine their evolution with size and disorder strength. We show that any finite value of the disorder is able to trigger a “small-world” behaviour as soon as the initial lattice is big enough, and study the crossover between a regular lattice and a “small-world” one. These results are corroborated by the investigation of an Ising model defined on the network, showing for every finite disorder fraction a crossover from a high-temperature region dominated by the underlying one-dimensional structure to a mean-field like low-temperature region. In particular there exists a finite-temperature ferromagnetic phase transition as soon as the disorder strength is finite. [0.5cm]

{{< /ci-details >}}

{{< ci-details summary="Label Informed Attributed Network Embedding (Xiao Huang et al., 2017)">}}

Xiao Huang, Jundong Li, Xia Hu. (2017)  
**Label Informed Attributed Network Embedding**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/44044556dae0e21cab058c18f704b15d33bd17c5)  
Influential Citation Count (45), SS-ID (44044556dae0e21cab058c18f704b15d33bd17c5)  

**ABSTRACT**  
Attributed network embedding aims to seek low-dimensional vector representations for nodes in a network, such that original network topological structure and node attribute proximity can be preserved in the vectors. These learned representations have been demonstrated to be helpful in many learning tasks such as network clustering and link prediction. While existing algorithms follow an unsupervised manner, nodes in many real-world attributed networks are often associated with abundant label information, which is potentially valuable in seeking more effective joint vector representations. In this paper, we investigate how labels can be modeled and incorporated to improve attributed network embedding. This is a challenging task since label information could be noisy and incomplete. In addition, labels are completely distinct with the geometrical structure and node attributes. The bewildering combination of heterogeneous information makes the joint vector representation learning more difficult. To address these issues, we propose a novel Label informed Attributed Network Embedding (LANE) framework. It can smoothly incorporate label information into the attributed network embedding while preserving their correlations. Experiments on real-world datasets demonstrate that the proposed framework achieves significantly better performance compared with the state-of-the-art embedding algorithms.

{{< /ci-details >}}

{{< ci-details summary="Deep Belief Network-Based Approaches for Link Prediction in Signed Social Networks (Feng Liu et al., 2015)">}}

Feng Liu, Bingquan Liu, Chengjie Sun, Ming Liu, Xiaolong Wang. (2015)  
**Deep Belief Network-Based Approaches for Link Prediction in Signed Social Networks**  
Entropy  
[Paper Link](https://www.semanticscholar.org/paper/45cb1ff9e1221b52ba3c26e33e98550f6117ae5a)  
Influential Citation Count (2), SS-ID (45cb1ff9e1221b52ba3c26e33e98550f6117ae5a)  

**ABSTRACT**  
In some online social network services (SNSs), the members are allowed to label their relationships with others, and such relationships can be represented as the links with signed values (positive or negative). The networks containing such relations are named signed social networks (SSNs), and some real-world complex systems can be also modeled with SSNs. Given the information of the observed structure of an SSN, the link prediction aims to estimate the values of the unobserved links. Noticing that most of the previous approaches for link prediction are based on the members’ similarity and the supervised learning method, however, research work on the investigation of the hidden principles that drive the behaviors of social members are rarely conducted. In this paper, the deep belief network (DBN)-based approaches for link prediction are proposed. Including an unsupervised link prediction model, a feature representation method and a DBN-based link prediction method are introduced. The experiments are done on the datasets from three SNSs (social networking services) in different domains, and the results show that our methods can predict the values of the links with high performance and have a good generalization ability across these datasets.

{{< /ci-details >}}

{{< ci-details summary="Wasserstein Embedding for Graph Learning (S. Kolouri et al., 2020)">}}

S. Kolouri, Navid Naderializadeh, G. Rohde, Heiko Hoffmann. (2020)  
**Wasserstein Embedding for Graph Learning**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/463f490d3bded6e527b0838da8495ed6441da25a)  
Influential Citation Count (1), SS-ID (463f490d3bded6e527b0838da8495ed6441da25a)  

**ABSTRACT**  
We present Wasserstein Embedding for Graph Learning (WEGL), a novel and fast framework for embedding entire graphs in a vector space, in which various machine learning models are applicable for graph-level prediction tasks. We leverage new insights on defining similarity between graphs as a function of the similarity between their node embedding distributions. Specifically, we use the Wasserstein distance to measure the dissimilarity between node embeddings of different graphs. Different from prior work, we avoid pairwise calculation of distances between graphs and reduce the computational complexity from quadratic to linear in the number of graphs. WEGL calculates Monge maps from a reference distribution to each node embedding and, based on these maps, creates a fixed-sized vector representation of the graph. We evaluate our new graph embedding approach on various benchmark graph-property prediction tasks, showing state-of-the-art classification performance, while having superior computational efficiency.

{{< /ci-details >}}

{{< ci-details summary="Representation Learning for Scale-free Networks (Rui Feng et al., 2017)">}}

Rui Feng, Yang Yang, Wenjie Hu, Fei Wu, Yueting Zhuang. (2017)  
**Representation Learning for Scale-free Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/464a60d8e67ab42d360c9be2d29f919d30312315)  
Influential Citation Count (3), SS-ID (464a60d8e67ab42d360c9be2d29f919d30312315)  

**ABSTRACT**  
Network embedding aims to learn the low-dimensional representations of vertexes in a network, while structure and inherent properties of the network is preserved. Existing network embedding works primarily focus on preserving the microscopic structure, such as the first- and second-order proximity of vertexes, while the macroscopic scale-free property is largely ignored. Scale-free property depicts the fact that vertex degrees follow a heavy-tailed distribution (i.e., only a few vertexes have high degrees) and is a critical property of real-world networks, such as social networks. In this paper, we study the problem of learning representations for scale-free networks. We first theoretically analyze the difficulty of embedding and reconstructing a scale-free network in the Euclidean space, by converting our problem to the sphere packing problem. Then, we propose the "degree penalty" principle for designing scale-free property preserving network embedding algorithm: punishing the proximity between high-degree vertexes. We introduce two implementations of our principle by utilizing the spectral techniques and a skip-gram model respectively. Extensive experiments on six datasets show that our algorithms are able to not only reconstruct heavy-tailed distributed degree distribution, but also outperform state-of-the-art embedding models in various network mining tasks, such as vertex classification and link prediction.

{{< /ci-details >}}

{{< ci-details summary="A latent factor model for highly multi-relational data (Rodolphe Jenatton et al., 2012)">}}

Rodolphe Jenatton, Nicolas Le Roux, Antoine Bordes, G. Obozinski. (2012)  
**A latent factor model for highly multi-relational data**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/473b3f2cc2c942c0116d980fe5b36a338f6017de)  
Influential Citation Count (35), SS-ID (473b3f2cc2c942c0116d980fe5b36a338f6017de)  

**ABSTRACT**  
Many data such as social networks, movie preferences or knowledge bases are multi-relational, in that they describe multiple relations between entities. While there is a large body of work focused on modeling these data, modeling these multiple types of relations jointly remains challenging. Further, existing approaches tend to breakdown when the number of these types grows. In this paper, we propose a method for modeling large multi-relational datasets, with possibly thousands of relations. Our model is based on a bilinear structure, which captures various orders of interaction of the data, and also shares sparse latent factors across different relations. We illustrate the performance of our approach on standard tensor-factorization datasets where we attain, or outperform, state-of-the-art results. Finally, a NLP application demonstrates our scalability and the ability of our model to learn efficient and semantically meaningful verb representations.

{{< /ci-details >}}

{{< ci-details summary="Random-Walk Computation of Similarities between Nodes of a Graph with Application to Collaborative Recommendation (François Fouss et al., 2007)">}}

François Fouss, A. Pirotte, J. Renders, Marco Saerens. (2007)  
**Random-Walk Computation of Similarities between Nodes of a Graph with Application to Collaborative Recommendation**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/474db64356d6c9c82fe2a8604cd6c13bc17bae78)  
Influential Citation Count (84), SS-ID (474db64356d6c9c82fe2a8604cd6c13bc17bae78)  

**ABSTRACT**  
This work presents a new perspective on characterizing the similarity between elements of a database or, more generally, nodes of a weighted and undirected graph. It is based on a Markov-chain model of random walk through the database. More precisely, we compute quantities (the average commute time, the pseudoinverse of the Laplacian matrix of the graph, etc.) that provide similarities between any pair of nodes, having the nice property of increasing when the number of paths connecting those elements increases and when the "length" of paths decreases. It turns out that the square root of the average commute time is a Euclidean distance and that the pseudoinverse of the Laplacian matrix is a kernel matrix (its elements are inner products closely related to commute times). A principal component analysis (PCA) of the graph is introduced for computing the subspace projection of the node vectors in a manner that preserves as much variance as possible in terms of the Euclidean commute-time distance. This graph PCA provides a nice interpretation to the "Fiedler vector," widely used for graph partitioning. The model is evaluated on a collaborative-recommendation task where suggestions are made about which movies people should watch based upon what they watched in the past. Experimental results on the MovieLens database show that the Laplacian-based similarities perform well in comparison with other methods. The model, which nicely fits into the so-called "statistical relational learning" framework, could also be used to compute document or word similarities, and, more generally, it could be applied to machine-learning and pattern-recognition tasks involving a relational database

{{< /ci-details >}}

{{< ci-details summary="Memory-Based Graph Networks (Amir Hosein Khas Ahmadi et al., 2020)">}}

Amir Hosein Khas Ahmadi, Kaveh Hassani, Parsa Moradi, Leo Lee, Q. Morris. (2020)  
**Memory-Based Graph Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/47f01fd4f0c9c77058a966d3f17dbc09cf7ef42a)  
Influential Citation Count (6), SS-ID (47f01fd4f0c9c77058a966d3f17dbc09cf7ef42a)  

**ABSTRACT**  
Graph Neural Networks (GNNs) are a class of deep models that operates on data with arbitrary topology and order-invariant structure represented as graphs. We introduce an efficient memory layer for GNNs that can learn to jointly perform graph representation learning and graph pooling. We also introduce two new networks based on our memory layer: Memory-Based Graph Neural Network (MemGNN) and Graph Memory Network (GMN) that can learn hierarchical graph representations by coarsening the graph throughout the layers of memory. The experimental results demonstrate that the proposed models achieve state-of-the-art results in six out of seven graph classification and regression benchmarks. We also show that the learned representations could correspond to chemical features in the molecule data.

{{< /ci-details >}}

{{< ci-details summary="Link Prediction on Multiple Graphs with Graph Embedding and Optimal Transport (Luu Huu Phuc et al., 2020)">}}

Luu Huu Phuc, M. Yamada, H. Kashima. (2020)  
**Link Prediction on Multiple Graphs with Graph Embedding and Optimal Transport**  
  
[Paper Link](https://www.semanticscholar.org/paper/480ff986724acc27e096560f8d433847e86cbdb3)  
Influential Citation Count (0), SS-ID (480ff986724acc27e096560f8d433847e86cbdb3)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier (Maxat Kulmanov et al., 2017)">}}

Maxat Kulmanov, Mohammed Asif Khan, R. Hoehndorf. (2017)  
**DeepGO: predicting protein functions from sequence and interactions using a deep ontology-aware classifier**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/48326bdcbb094fce48b17390f7743c23cd0a1ebc)  
Influential Citation Count (25), SS-ID (48326bdcbb094fce48b17390f7743c23cd0a1ebc)  

**ABSTRACT**  
Abstract Motivation A large number of protein sequences are becoming available through the application of novel high-throughput sequencing technologies. Experimental functional characterization of these proteins is time-consuming and expensive, and is often only done rigorously for few selected model organisms. Computational function prediction approaches have been suggested to fill this gap. The functions of proteins are classified using the Gene Ontology (GO), which contains over 40 000 classes. Additionally, proteins have multiple functions, making function prediction a large-scale, multi-class, multi-label problem. Results We have developed a novel method to predict protein function from sequence. We use deep learning to learn features from protein sequences as well as a cross-species protein–protein interaction network. Our approach specifically outputs information in the structure of the GO and utilizes the dependencies between GO classes as background information to construct a deep learning model. We evaluate our method using the standards established by the Computational Assessment of Function Annotation (CAFA) and demonstrate a significant improvement over baseline methods such as BLAST, in particular for predicting cellular locations. Availability and implementation Web server: http://deepgo.bio2vec.net, Source code: https://github.com/bio-ontology-research-group/deepgo Supplementary information Supplementary data are available at Bioinformatics online.

{{< /ci-details >}}

{{< ci-details summary="Gated Graph Sequence Neural Networks (Yujia Li et al., 2015)">}}

Yujia Li, Daniel Tarlow, Marc Brockschmidt, R. Zemel. (2015)  
**Gated Graph Sequence Neural Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/492f57ee9ceb61fb5a47ad7aebfec1121887a175)  
Influential Citation Count (280), SS-ID (492f57ee9ceb61fb5a47ad7aebfec1121887a175)  

**ABSTRACT**  
Abstract: Graph-structured data appears frequently in domains including chemistry, natural language semantics, social networks, and knowledge bases. In this work, we study feature learning techniques for graph-structured inputs. Our starting point is previous work on Graph Neural Networks (Scarselli et al., 2009), which we modify to use gated recurrent units and modern optimization techniques and then extend to output sequences. The result is a flexible and broadly useful class of neural network models that has favorable inductive biases relative to purely sequence-based models (e.g., LSTMs) when the problem is graph-structured. We demonstrate the capabilities on some simple AI (bAbI) and graph algorithm learning tasks. We then show it achieves state-of-the-art performance on a problem from program verification, in which subgraphs need to be matched to abstract data structures.

{{< /ci-details >}}

{{< ci-details summary="Applying link-based classification to label blogs (Smriti Bhagat et al., 2007)">}}

Smriti Bhagat, Irina Rozenbaum, Graham Cormode. (2007)  
**Applying link-based classification to label blogs**  
WebKDD/SNA-KDD '07  
[Paper Link](https://www.semanticscholar.org/paper/496d1a45eb511893a86dd7a8452a386aa5ce1657)  
Influential Citation Count (5), SS-ID (496d1a45eb511893a86dd7a8452a386aa5ce1657)  

**ABSTRACT**  
In analyzing data from social and communication networks, we encounter the problem of classifying objects where there is an explicit link structure amongst the objects. We study the problem of inferring the classification of all the objects from a labeled subset, using only the link-based information amongst the objects.  We abstract the above as a labeling problem on multigraphs with weighted edges. We present two classes of algorithms, based on local and global similarities. Then we focus on multigraphs induced by blog data, and carefully apply our general algorithms to specifically infer labels such as age, gender and location associated with the blog based only on the link-structure amongst them. We perform a comprehensive set of experiments with real, large-scale blog data sets and show that significant accuracy is possible from little or no non-link information, and our methods scale to millions of nodes and edges.

{{< /ci-details >}}

{{< ci-details summary="Support and Centrality: Learning Weights for Knowledge Graph Embedding Models (Gengchen Mai et al., 2018)">}}

Gengchen Mai, K. Janowicz, Bo Yan. (2018)  
**Support and Centrality: Learning Weights for Knowledge Graph Embedding Models**  
EKAW  
[Paper Link](https://www.semanticscholar.org/paper/49899fd94cd272914f7d1e81b0915058c25bb665)  
Influential Citation Count (0), SS-ID (49899fd94cd272914f7d1e81b0915058c25bb665)  

**ABSTRACT**  
Computing knowledge graph (KG) embeddings is a technique to learn distributional representations for components of a knowledge graph while preserving structural information. The learned embeddings can be used in multiple downstream tasks such as question answering, information extraction, query expansion, semantic similarity, and information retrieval. Over the past years, multiple embedding techniques have been proposed based on different underlying assumptions. The most actively researched models are translation-based which treat relations as translation operations in a shared (or relation-specific) space. Interestingly, almost all KG embedding models treat each triple equally, regardless of the fact that the contribution of each triple to the global information content differs substantially. Many triples can be inferred from others, while some triples are the foundational (basis) statements that constitute a knowledge graph, thereby supporting other triples. Hence, in order to learn a suitable embedding model, each triple should be treated differently with respect to its information content. Here, we propose a data-driven approach to measure the information content of each triple with respect to the whole knowledge graph by using rule mining and PageRank. We show how to compute triple-specific weights to improve the performance of three KG embedding models (TransE, TransR and HolE). Link prediction tasks on two standard datasets, FB15K and WN18, show the effectiveness of our weighted KG embedding model over other more complex models. In fact, for FB15K our TransE-RW embeddings model outperforms models such as TransE, TransM, TransH, and TransR by at least 12.98% for measuring the Mean Rank and at least 1.45% for HIT@10. Our HolE-RW model also outperforms HolE and ComplEx by at least 14.3% for MRR and about 30.4% for HIT@1 on FB15K. Finally, TransR-RW show an improvement over TransR by 3.90% for Mean Rank and 0.87% for HIT@10.

{{< /ci-details >}}

{{< ci-details summary="Watch Your Step: Learning Node Embeddings via Graph Attention (Sami Abu-El-Haija et al., 2017)">}}

Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou, Alexander A. Alemi. (2017)  
**Watch Your Step: Learning Node Embeddings via Graph Attention**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/49a5b5e65078eff512083d9de413d49a8aadc064)  
Influential Citation Count (13), SS-ID (49a5b5e65078eff512083d9de413d49a8aadc064)  

**ABSTRACT**  
Graph embedding methods represent nodes in a continuous vector space, preserving different types of relational information from the graph. There are many hyper-parameters to these methods (e.g. the length of a random walk) which have to be manually tuned for every graph. In this paper, we replace previously fixed hyper-parameters with trainable ones that we automatically learn via backpropagation. In particular, we propose a novel attention model on the power series of the transition matrix, which guides the random walk to optimize an upstream objective. Unlike previous approaches to attention models, the method that we propose utilizes attention parameters exclusively on the data itself (e.g. on the random walk), and are not used by the model for inference. We experiment on link prediction tasks, as we aim to produce embeddings that best-preserve the graph structure, generalizing to unseen information. We improve state-of-the-art results on a comprehensive suite of real-world graph datasets including social, collaboration, and biological networks, where we observe that our graph attention model can reduce the error by up to 20\%-40\%. We show that our automatically-learned attention parameters can vary significantly per graph, and correspond to the optimal choice of hyper-parameter if we manually tune existing methods.

{{< /ci-details >}}

{{< ci-details summary="Deep Graph Contrastive Representation Learning (Yanqiao Zhu et al., 2020)">}}

Yanqiao Zhu, Yichen Xu, Feng Yu, Q. Liu, Shu Wu, Liang Wang. (2020)  
**Deep Graph Contrastive Representation Learning**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/4bf76588122827157c43a59e656dccc6b6a22e90)  
Influential Citation Count (35), SS-ID (4bf76588122827157c43a59e656dccc6b6a22e90)  

**ABSTRACT**  
Graph representation learning nowadays becomes fundamental in analyzing graph-structured data. Inspired by recent success of contrastive methods, in this paper, we propose a novel framework for unsupervised graph representation learning by leveraging a contrastive objective at the node level. Specifically, we generate two graph views by corruption and learn node representations by maximizing the agreement of node representations in these two views. To provide diverse node contexts for the contrastive objective, we propose a hybrid scheme for generating graph views on both structure and attribute levels. Besides, we provide theoretical justification behind our motivation from two perspectives, mutual information and the classical triplet loss. We perform empirical experiments on both transductive and inductive learning tasks using a variety of real-world datasets. Experimental experiments demonstrate that despite its simplicity, our proposed method consistently outperforms existing state-of-the-art methods by large margins. Moreover, our unsupervised method even surpasses its supervised counterparts on transductive tasks, demonstrating its great potential in real-world applications.

{{< /ci-details >}}

{{< ci-details summary="Homophily, Structure, and Content Augmented Network Representation Learning (Daokun Zhang et al., 2016)">}}

Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang. (2016)  
**Homophily, Structure, and Content Augmented Network Representation Learning**  
2016 IEEE 16th International Conference on Data Mining (ICDM)  
[Paper Link](https://www.semanticscholar.org/paper/4cc10f77819ad376ea539074c2de14a8999e3269)  
Influential Citation Count (4), SS-ID (4cc10f77819ad376ea539074c2de14a8999e3269)  

**ABSTRACT**  
Advances in social networking and communication technologies have witnessed an increasing number of applications where data is not only characterized by rich content information, but also connected with complex relationships representing social roles and dependencies between individuals. To enable knowledge discovery from such networked data, network representation learning (NRL) aims to learn vector representations for network nodes, such that off-the-shelf machine learning algorithms can be directly applied. To date, existing NRL methods either primarily focus on network structure or simply combine node content and topology for learning. We argue that in information networks, information is mainly originated from three sources: (1) homophily, (2) topology structure, and (3) node content. Homophily states social phenomenon where individuals sharing similar attributes (content) tend to be directly connected through local relational ties, while topology structure emphasizes more on global connections. To ensure effective network representation learning, we propose to augment three information sources into one learning objective function, so that the interplay roles between three parties are enforced by requiring the learned network representations (1) being consistent with node content and topology structure, and also (2) following the social homophily constraints in the learned space. Experiments on multi-class node classification demonstrate that the representations learned by the proposed method consistently outperform state-of-the-art NRL methods, especially for very sparsely labeled networks.

{{< /ci-details >}}

{{< ci-details summary="The rendezvous algorithm: multiclass semi-supervised learning with Markov random walks (Arik Azran, 2007)">}}

Arik Azran. (2007)  
**The rendezvous algorithm: multiclass semi-supervised learning with Markov random walks**  
ICML '07  
[Paper Link](https://www.semanticscholar.org/paper/4e9585cd65c7e1a19ae16d8fed12c810070c65d3)  
Influential Citation Count (5), SS-ID (4e9585cd65c7e1a19ae16d8fed12c810070c65d3)  

**ABSTRACT**  
We consider the problem of multiclass classification where both labeled and unlabeled data points are given. We introduce and demonstrate a new approach for estimating a distribution over the missing labels where data points are viewed as nodes of a graph, and pairwise similarities are used to derive a transition probability matrix P for a Markov random walk between them. The algorithm associates each point with a particle which moves between points according to P. Labeled points are set to be absorbing states of the Markov random walk, and the probability of each particle to be absorbed by the different labeled points, as the number of steps increases, is then used to derive a distribution over the associated missing label. A computationally efficient algorithm to implement this is derived and demonstrated on both real and artificial data sets, including a numerical comparison with other methods.

{{< /ci-details >}}

{{< ci-details summary="Learning from Collective Intelligence (Hanwang Zhang et al., 2016)">}}

Hanwang Zhang, Xindi Shang, Huanbo Luan, Meng Wang, Tat-Seng Chua. (2016)  
**Learning from Collective Intelligence**  
ACM Trans. Multim. Comput. Commun. Appl.  
[Paper Link](https://www.semanticscholar.org/paper/4f3417e73528025a5429547814e5a2fd91deb818)  
Influential Citation Count (4), SS-ID (4f3417e73528025a5429547814e5a2fd91deb818)  

**ABSTRACT**  
Feature representation for visual content is the key to the progress of many fundamental applications such as annotation and cross-modal retrieval. Although recent advances in deep feature learning offer a promising route towards these tasks, they are limited in application domains where high-quality and large-scale training data are expensive to obtain. In this article, we propose a novel deep feature learning paradigm based on social collective intelligence, which can be acquired from the inexhaustible social multimedia content on the Web, in particular, largely social images and tags. Differing from existing feature learning approaches that rely on high-quality image-label supervision, our weak supervision is acquired by mining the visual-semantic embeddings from noisy, sparse, and diverse social image collections. The resultant image-word embedding space can be used to (1) fine-tune deep visual models for low-level feature extractions and (2) seek sparse representations as high-level cross-modal features for both image and text. We offer an easy-to-use implementation for the proposed paradigm, which is fast and compatible with any state-of-the-art deep architectures. Extensive experiments on several benchmarks demonstrate that the cross-modal features learned by our paradigm significantly outperforms others in various applications such as content-based retrieval, classification, and image captioning.

{{< /ci-details >}}

{{< ci-details summary="K-Core based Temporal Graph Convolutional Network for Dynamic Graphs (Jingxin Liu et al., 2020)">}}

Jingxin Liu, Chang Xu, Chang Yin, Weiqiang Wu, You Song. (2020)  
**K-Core based Temporal Graph Convolutional Network for Dynamic Graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/4fda7672f67a8872902dee92e4c1bcf2c833a2b1)  
Influential Citation Count (1), SS-ID (4fda7672f67a8872902dee92e4c1bcf2c833a2b1)  

**ABSTRACT**  
Graph representation learning is a fundamental task in various applications that strives to learn low-dimensional embeddings for nodes that can preserve graph topology information. However, many existing methods focus on static graphs while ignoring evolving graph patterns. Inspired by the success of graph convolutional networks(GCNs) in static graph embedding, we propose a novel k-core based temporal graph convolutional network, the CTGCN, to learn node representations for dynamic graphs. In contrast to previous dynamic graph embedding methods, CTGCN can preserve both local connective proximity and global structural similarity while simultaneously capturing graph dynamics. In the proposed framework, the traditional graph convolution is generalized into two phases, feature transformation and feature aggregation, which gives the CTGCN more flexibility and enables the CTGCN to learn connective and structural information under the same framework. Experimental results on 7 real-world graphs demonstrate that the CTGCN outperforms existing state-of-the-art graph embedding methods in several tasks, including link prediction and structural role classification. The source code of this work can be obtained from this https URL.

{{< /ci-details >}}

{{< ci-details summary="Reasoning With Neural Tensor Networks for Knowledge Base Completion (R. Socher et al., 2013)">}}

R. Socher, Danqi Chen, Christopher D. Manning, A. Ng. (2013)  
**Reasoning With Neural Tensor Networks for Knowledge Base Completion**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/50d53cc562225549457cbc782546bfbe1ac6f0cf)  
Influential Citation Count (277), SS-ID (50d53cc562225549457cbc782546bfbe1ac6f0cf)  

**ABSTRACT**  
Knowledge bases are an important resource for question answering and other tasks but often suffer from incompleteness and lack of ability to reason over their discrete entities and relationships. In this paper we introduce an expressive neural tensor network suitable for reasoning over relationships between two entities. Previous work represented entities as either discrete atomic units or with a single entity vector representation. We show that performance can be improved when entities are represented as an average of their constituting word vectors. This allows sharing of statistical strength between, for instance, facts involving the "Sumatran tiger" and "Bengal tiger." Lastly, we demonstrate that all models improve when these word vectors are initialized with vectors learned from unsupervised large corpora. We assess the model by considering the problem of predicting additional true relations between entities given a subset of the knowledge base. Our model outperforms previous models and can classify unseen relationships in WordNet and FreeBase with an accuracy of 86.2% and 90.0%, respectively.

{{< /ci-details >}}

{{< ci-details summary="Learning network representations (L. G. Moyano, 2017)">}}

L. G. Moyano. (2017)  
**Learning network representations**  
  
[Paper Link](https://www.semanticscholar.org/paper/51a748c8d7b780c2bb863a2259598a3a216330f1)  
Influential Citation Count (0), SS-ID (51a748c8d7b780c2bb863a2259598a3a216330f1)  

**ABSTRACT**  
Abstract In this review I present several representation learning methods, and discuss the latest advancements with emphasis in applications to network science. Representation learning is a set of techniques that has the goal of efficiently mapping data structures into convenient latent spaces. Either for dimensionality reduction or for gaining semantic content, this type of feature embeddings has demonstrated to be useful, for example, for node classification or link prediction tasks, among many other relevant applications to networks. I provide a description of the state-of-the-art of network representation learning as well as a detailed account of the connections with other fields of study such as continuous word embeddings and deep learning architectures. Finally, I provide a broad view of several applications of these techniques to networks in various domains. 

{{< /ci-details >}}

{{< ci-details summary="Link prediction approach to collaborative filtering (Zan Huang et al., 2005)">}}

Zan Huang, Xin Li, Hsinchun Chen. (2005)  
**Link prediction approach to collaborative filtering**  
Proceedings of the 5th ACM/IEEE-CS Joint Conference on Digital Libraries (JCDL '05)  
[Paper Link](https://www.semanticscholar.org/paper/5257c7ddcf1d0b51596524eea9f54942f124e0ec)  
Influential Citation Count (19), SS-ID (5257c7ddcf1d0b51596524eea9f54942f124e0ec)  

**ABSTRACT**  
Recommender systems can provide valuable services in a digital library environment, as demonstrated by its commercial success in book, movie, and music industries. One of the most commonly-used and successful recommendation algorithms is collaborative filtering, which explores the correlations within user-item interactions to infer user interests and preferences. However, the recommendation quality of collaborative filtering approaches is greatly limited by the data sparsity problem. To alleviate this problem we have previously proposed graph-based algorithms to explore transitive user-item associations. In this paper, we extend the idea of analyzing user-item interactions as graphs and employ link prediction approaches proposed in the recent network modeling literature for making collaborative filtering recommendations. We have adapted a wide range of linkage measures for making recommendations. Our preliminary experimental results based on a book recommendation dataset show that some of these measures achieved significantly better performance than standard collaborative filtering algorithms

{{< /ci-details >}}

{{< ci-details summary="Heterogeneous Information Network Embedding for Meta Path based Proximity (Zhipeng Huang et al., 2017)">}}

Zhipeng Huang, N. Mamoulis. (2017)  
**Heterogeneous Information Network Embedding for Meta Path based Proximity**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/52a150d6a098ef142bece099dadaa613fddbae50)  
Influential Citation Count (14), SS-ID (52a150d6a098ef142bece099dadaa613fddbae50)  

**ABSTRACT**  
A network embedding is a representation of a large graph in a low-dimensional space, where vertices are modeled as vectors. The objective of a good embedding is to preserve the proximity between vertices in the original graph. This way, typical search and mining methods can be applied in the embedded space with the help of off-the-shelf multidimensional indexing approaches. Existing network embedding techniques focus on homogeneous networks, where all vertices are considered to belong to a single class.

{{< /ci-details >}}

{{< ci-details summary="Graph Convolutional Tracking (Junyu Gao et al., 2019)">}}

Junyu Gao, Tianzhu Zhang, Changsheng Xu. (2019)  
**Graph Convolutional Tracking**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/53970ae69a73f547a56661fd25f6711746d277fb)  
Influential Citation Count (13), SS-ID (53970ae69a73f547a56661fd25f6711746d277fb)  

**ABSTRACT**  
Tracking by siamese networks has achieved favorable performance in recent years. However, most of existing siamese methods do not take full advantage of spatial-temporal target appearance modeling under different contextual situations. In fact, the spatial-temporal information can provide diverse features to enhance the target representation, and the context information is important for online adaption of target localization. To comprehensively leverage the spatial-temporal structure of historical target exemplars and get benefit from the context information, in this work, we present a novel Graph Convolutional Tracking (GCT) method for high-performance visual tracking. Specifically, the GCT jointly incorporates two types of Graph Convolutional Networks (GCNs) into a siamese framework for target appearance modeling. Here, we adopt a spatial-temporal GCN to model the structured representation of historical target exemplars. Furthermore, a context GCN is designed to utilize the context of the current frame to learn adaptive features for target localization. Extensive results on 4 challenging benchmarks show that our GCT method performs favorably against state-of-the-art trackers while running around 50 frames per second.

{{< /ci-details >}}

{{< ci-details summary="Improving Neural Entity Disambiguation with Graph Embeddings (Özge Sevgili et al., 2019)">}}

Özge Sevgili, Alexander Panchenko, Chris Biemann. (2019)  
**Improving Neural Entity Disambiguation with Graph Embeddings**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/540a140e4b8576e0b4edaefd5cee9d9c55da0e1d)  
Influential Citation Count (1), SS-ID (540a140e4b8576e0b4edaefd5cee9d9c55da0e1d)  

**ABSTRACT**  
Entity Disambiguation (ED) is the task of linking an ambiguous entity mention to a corresponding entry in a knowledge base. Current methods have mostly focused on unstructured text data to learn representations of entities, however, there is structured information in the knowledge base itself that should be useful to disambiguate entities. In this work, we propose a method that uses graph embeddings for integrating structured information from the knowledge base with unstructured information from text-based representations. Our experiments confirm that graph embeddings trained on a graph of hyperlinks between Wikipedia articles improve the performances of simple feed-forward neural ED model and a state-of-the-art neural ED system.

{{< /ci-details >}}

{{< ci-details summary="Variational Graph Auto-Encoders (Thomas Kipf et al., 2016)">}}

Thomas Kipf, M. Welling. (2016)  
**Variational Graph Auto-Encoders**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/54906484f42e871f7c47bbfe784a358b1448231f)  
Influential Citation Count (347), SS-ID (54906484f42e871f7c47bbfe784a358b1448231f)  

**ABSTRACT**  
We introduce the variational graph auto-encoder (VGAE), a framework for unsupervised learning on graph-structured data based on the variational auto-encoder (VAE). This model makes use of latent variables and is capable of learning interpretable latent representations for undirected graphs. We demonstrate this model using a graph convolutional network (GCN) encoder and a simple inner product decoder. Our model achieves competitive results on a link prediction task in citation networks. In contrast to most existing models for unsupervised learning on graph-structured data and link prediction, our model can naturally incorporate node features, which significantly improves predictive performance on a number of benchmark datasets.

{{< /ci-details >}}

{{< ci-details summary="Efficient embedding of complex networks to hyperbolic space via their Laplacian (Gregorio Alanis-Lobato et al., 2016)">}}

Gregorio Alanis-Lobato, Pablo Mier, Miguel Andrade. (2016)  
**Efficient embedding of complex networks to hyperbolic space via their Laplacian**  
Scientific reports  
[Paper Link](https://www.semanticscholar.org/paper/559897d98f2c006d27fa09d7263c4a959d94eec3)  
Influential Citation Count (4), SS-ID (559897d98f2c006d27fa09d7263c4a959d94eec3)  

**ABSTRACT**  
The different factors involved in the growth process of complex networks imprint valuable information in their observable topologies. How to exploit this information to accurately predict structural network changes is the subject of active research. A recent model of network growth sustains that the emergence of properties common to most complex systems is the result of certain trade-offs between node birth-time and similarity. This model has a geometric interpretation in hyperbolic space, where distances between nodes abstract this optimisation process. Current methods for network hyperbolic embedding search for node coordinates that maximise the likelihood that the network was produced by the afore-mentioned model. Here, a different strategy is followed in the form of the Laplacian-based Network Embedding, a simple yet accurate, efficient and data driven manifold learning approach, which allows for the quick geometric analysis of big networks. Comparisons against existing embedding and prediction techniques highlight its applicability to network evolution and link prediction.

{{< /ci-details >}}

{{< ci-details summary="Molecular graph convolutions: moving beyond fingerprints (S. Kearnes et al., 2016)">}}

S. Kearnes, Kevin McCloskey, M. Berndl, V. Pande, Patrick F. Riley. (2016)  
**Molecular graph convolutions: moving beyond fingerprints**  
Journal of Computer-Aided Molecular Design  
[Paper Link](https://www.semanticscholar.org/paper/561c3fa53d36405186da9cab02bd68635c3738aa)  
Influential Citation Count (41), SS-ID (561c3fa53d36405186da9cab02bd68635c3738aa)  

**ABSTRACT**  
Molecular “fingerprints” encoding structural information are the workhorse of cheminformatics and machine learning in drug discovery applications. However, fingerprint representations necessarily emphasize particular aspects of the molecular structure while ignoring others, rather than allowing the model to make data-driven decisions. We describe molecular graph convolutions, a machine learning architecture for learning from undirected graphs, specifically small molecules. Graph convolutions use a simple encoding of the molecular graph—atoms, bonds, distances, etc.—which allows the model to take greater advantage of information in the graph structure. Although graph convolutions do not outperform all fingerprint-based methods, they (along with other graph-based methods) represent a new paradigm in ligand-based virtual screening with exciting opportunities for future improvement.

{{< /ci-details >}}

{{< ci-details summary="DeepBrowse: Similarity-Based Browsing Through Large Lists (Extended Abstract) (Haochen Chen et al., 2017)">}}

Haochen Chen, Arvind Ram Anantharam, S. Skiena. (2017)  
**DeepBrowse: Similarity-Based Browsing Through Large Lists (Extended Abstract)**  
SISAP  
[Paper Link](https://www.semanticscholar.org/paper/562d08ed73dad28c989b6a04fa1e389b0125ba59)  
Influential Citation Count (0), SS-ID (562d08ed73dad28c989b6a04fa1e389b0125ba59)  

**ABSTRACT**  
We propose a new approach for browsing through large lists in the absence of a predefined hierarchy. DeepBrowse is defined by the interaction of two fixed, globally-defined permutations on the space of objects: one ordering the items by similarity, the second based on magnitude or importance. We demonstrate this paradigm through our WikiBrowse app for discovering interesting Wikipedia pages, which enables the user to scan similar related entities and then increase depth once a region of interest has been found.

{{< /ci-details >}}

{{< ci-details summary="Drug-target interaction prediction using ensemble learning and dimensionality reduction. (Ali Ezzat et al., 2017)">}}

Ali Ezzat, Min Wu, Xiaoli Li, C. Kwoh. (2017)  
**Drug-target interaction prediction using ensemble learning and dimensionality reduction.**  
Methods  
[Paper Link](https://www.semanticscholar.org/paper/57068d32847df76f2d5ba6bea4df0fef8fac0f30)  
Influential Citation Count (5), SS-ID (57068d32847df76f2d5ba6bea4df0fef8fac0f30)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Heterogeneous Dynamic Graph Attention Network (Qiuyan Li et al., 2020)">}}

Qiuyan Li, Yanlei Shang, Xiuquan Qiao, Wei Dai. (2020)  
**Heterogeneous Dynamic Graph Attention Network**  
2020 IEEE International Conference on Knowledge Graph (ICKG)  
[Paper Link](https://www.semanticscholar.org/paper/57512101d4d64e7ec715a50eaba2e3e479239c64)  
Influential Citation Count (1), SS-ID (57512101d4d64e7ec715a50eaba2e3e479239c64)  

**ABSTRACT**  
Network embedding (graph embedding) has become the focus of studying graph structure in recent years. In addition to the research on homogeneous networks and heterogeneous networks, there are also some methods to attempt to solve the problem of dynamic network embedding. However, in dynamic networks, there is no research method specifically for heterogeneous networks. Therefore, this paper proposes a heterogeneous dynamic graph attention network (HDGAN), which attempts to use the attention mechanism to take the heterogeneity and dynamics of the network into account at the same time, so as to better learn network embedding. Our method is based on three levels of attention, namely structural-level attention, semantic-level attention and time-level attention. Structural-level attention pays attention to the network structure itself, and obtains the representation of structural-level nodes by learning the attention coefficients of neighbor nodes. Semantic-level attention integrates semantic information into the representation of nodes by learning the optimal weighted combination of different meta-paths. Time-level attention is based on the time decay effect, and the time feature is introduced into the node representation by neighborhood formation sequence. Through the above three levels of attention mechanism, the final network embedding can be obtained.Through experiments on two real-world heterogeneous dynamic networks, our models have the best results, proving the effectiveness of the HDGAN model.

{{< /ci-details >}}

{{< ci-details summary="Recognizing Mentions of Adverse Drug Reaction in Social Media Using Knowledge-Infused Recurrent Models (Gabriel Stanovsky et al., 2017)">}}

Gabriel Stanovsky, D. Gruhl, Pablo N. Mendes. (2017)  
**Recognizing Mentions of Adverse Drug Reaction in Social Media Using Knowledge-Infused Recurrent Models**  
EACL  
[Paper Link](https://www.semanticscholar.org/paper/583aef90d21360c2a406af2f2323b7cfa86be532)  
Influential Citation Count (4), SS-ID (583aef90d21360c2a406af2f2323b7cfa86be532)  

**ABSTRACT**  
Recognizing mentions of Adverse Drug Reactions (ADR) in social media is challenging: ADR mentions are context-dependent and include long, varied and unconventional descriptions as compared to more formal medical symptom terminology. We use the CADEC corpus to train a recurrent neural network (RNN) transducer, integrated with knowledge graph embeddings of DBpedia, and show the resulting model to be highly accurate (93.4 F1). Furthermore, even when lacking high quality expert annotations, we show that by employing an active learning technique and using purpose built annotation tools, we can train the RNN to perform well (83.9 F1).

{{< /ci-details >}}

{{< ci-details summary="GEMSEC: Graph Embedding with Self Clustering (Benedek Rozemberczki et al., 2018)">}}

Benedek Rozemberczki, Ryan Davies, R. Sarkar, Charles Sutton. (2018)  
**GEMSEC: Graph Embedding with Self Clustering**  
2019 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)  
[Paper Link](https://www.semanticscholar.org/paper/59be148a4b5f7e05cb3cb24afa1f6adad2cdfa29)  
Influential Citation Count (16), SS-ID (59be148a4b5f7e05cb3cb24afa1f6adad2cdfa29)  

**ABSTRACT**  
Modern graph embedding procedures can efficiently process graphs with millions of nodes. In this paper, we propose GEMSEC - a graph embedding algorithm which learns a clustering of the nodes simultaneously with computing their embedding. GEMSEC is a general extension of earlier work in the domain of sequence-based graph embedding. GEMSEC places nodes in an abstract feature space where the vertex features minimize the negative log-likelihood of preserving sampled vertex neighborhoods, and it incorporates known social network properties through a machine learning regularization. We present two new social network datasets and show that by simultaneously considering the embedding and clustering problems with respect to social properties, GEMSEC extracts high-quality clusters competitive with or superior to other community detection algorithms. In experiments, the method is found to be computationally efficient and robust to the choice of hyperparameters.

{{< /ci-details >}}

{{< ci-details summary="Graph Classification using Structural Attention (J. B. Lee et al., 2018)">}}

J. B. Lee, Ryan A. Rossi, Xiangnan Kong. (2018)  
**Graph Classification using Structural Attention**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/59d502851cd20f28af03eef1d15dc83d3a7bb300)  
Influential Citation Count (10), SS-ID (59d502851cd20f28af03eef1d15dc83d3a7bb300)  

**ABSTRACT**  
Graph classification is a problem with practical applications in many different domains. To solve this problem, one usually calculates certain graph statistics (i.e., graph features) that help discriminate between graphs of different classes. When calculating such features, most existing approaches process the entire graph. In a graphlet-based approach, for instance, the entire graph is processed to get the total count of different graphlets or subgraphs. In many real-world applications, however, graphs can be noisy with discriminative patterns confined to certain regions in the graph only. In this work, we study the problem of attention-based graph classification. The use of attention allows us to focus on small but informative parts of the graph, avoiding noise in the rest of the graph. We present a novel RNN model, called the Graph Attention Model (GAM), that processes only a portion of the graph by adaptively selecting a sequence of "informative" nodes. Experimental results on multiple real-world datasets show that the proposed method is competitive against various well-known methods in graph classification even though our method is limited to only a portion of the graph.

{{< /ci-details >}}

{{< ci-details summary="Fast Gradient Attack on Network Embedding (Jinyin Chen et al., 2018)">}}

Jinyin Chen, Yangyang Wu, Xuanheng Xu, Yixian Chen, Haibin Zheng, Qi Xuan. (2018)  
**Fast Gradient Attack on Network Embedding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/5c0fe48ce1530d9757efca49d78709fc77caaf6c)  
Influential Citation Count (14), SS-ID (5c0fe48ce1530d9757efca49d78709fc77caaf6c)  

**ABSTRACT**  
Network embedding maps a network into a low-dimensional Euclidean space, and thus facilitate many network analysis tasks, such as node classification, link prediction and community detection etc, by utilizing machine learning methods. In social networks, we may pay special attention to user privacy, and would like to prevent some target nodes from being identified by such network analysis methods in certain cases. Inspired by successful adversarial attack on deep learning models, we propose a framework to generate adversarial networks based on the gradient information in Graph Convolutional Network (GCN). In particular, we extract the gradient of pairwise nodes based on the adversarial network, and select the pair of nodes with maximum absolute gradient to realize the Fast Gradient Attack (FGA) and update the adversarial network. This process is implemented iteratively and terminated until certain condition is satisfied, i.e., the number of modified links reaches certain predefined value. Comprehensive attacks, including unlimited attack, direct attack and indirect attack, are performed on six well-known network embedding methods. The experiments on real-world networks suggest that our proposed FGA behaves better than some baseline methods, i.e., the network embedding can be easily disturbed using FGA by only rewiring few links, achieving state-of-the-art attack performance.

{{< /ci-details >}}

{{< ci-details summary="Representation Learning for Measuring Entity Relatedness with Rich Information (Yu Zhao et al., 2015)">}}

Yu Zhao, Zhiyuan Liu, Maosong Sun. (2015)  
**Representation Learning for Measuring Entity Relatedness with Rich Information**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/5c560cdfbfa48ed570b9b11e1a2f15e371e635f4)  
Influential Citation Count (3), SS-ID (5c560cdfbfa48ed570b9b11e1a2f15e371e635f4)  

**ABSTRACT**  
Incorporating multiple types of relational information from heterogeneous networks has been proved effective in data mining. Although Wikipedia is one of the most famous heterogeneous network, previous works of semantic analysis on Wikipedia are mostly limited on single type of relations. In this paper, we aim at incorporating multiple types of relations to measure the semantic relatedness between Wikipedia entities. We propose a framework of coordinate matrix factorization to construct lowdimensional continuous representation for entities, categories and words in the same semantic space. We formulate this task as the completion of a sparse entity-entity association matrix, in which each entry quantifies the strength of relatedness between corresponding entities. We evaluate our model on the task of judging pair-wise word similarity. Experiment result shows that our model outperforms both traditional entity relatedness algorithms and other representation learning models.

{{< /ci-details >}}

{{< ci-details summary="Combining content and link for classification using matrix factorization (Shenghuo Zhu et al., 2007)">}}

Shenghuo Zhu, Kai Yu, Yun Chi, Yihong Gong. (2007)  
**Combining content and link for classification using matrix factorization**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/5c58ad9a6c09782814a7d048bebd6ef1609c0fb4)  
Influential Citation Count (19), SS-ID (5c58ad9a6c09782814a7d048bebd6ef1609c0fb4)  

**ABSTRACT**  
The world wide web contains rich textual contents that areinterconnected via complex hyperlinks. This huge database violates the assumption held by most of conventional statistical methods that each web page is considered as an independent and identical sample. It is thus difficult to apply traditional mining or learning methods for solving web mining problems, e.g., web page classification, by exploiting both the content and the link structure. The research in this direction has recently received considerable attention but are still in an early stage. Though a few methods exploit both the link structure or the content information, some of them combine the only authority information with the content information, and the others first decompose the link structure into hub and authority features, then apply them as additional document features. Being practically attractive for its great simplicity, this paper aims to design an algorithm that exploits both the content and linkage information, by carrying out a joint factorization on both the linkage adjacency matrix and the document-term matrix, and derives a new representation for web pages in a low-dimensional factor space, without explicitly separating them as content, hub or authority factors. Further analysis can be performed based on the compact representation of web pages. In the experiments, the proposed method is compared with state-of-the-art methods and demonstrates an excellent accuracy in hypertext classification on the WebKB and Cora benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Convolutional Networks on Graphs for Learning Molecular Fingerprints (D. Duvenaud et al., 2015)">}}

D. Duvenaud, D. Maclaurin, J. Aguilera-Iparraguirre, R. Gómez-Bombarelli, Timothy D. Hirzel, Alán Aspuru-Guzik, Ryan P. Adams. (2015)  
**Convolutional Networks on Graphs for Learning Molecular Fingerprints**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/5d1bfeed240709725c78bc72ea40e55410b373dc)  
Influential Citation Count (147), SS-ID (5d1bfeed240709725c78bc72ea40e55410b373dc)  

**ABSTRACT**  
We introduce a convolutional neural network that operates directly on graphs. These networks allow end-to-end learning of prediction pipelines whose inputs are graphs of arbitrary size and shape. The architecture we present generalizes standard molecular feature extraction methods based on circular fingerprints. We show that these data-driven features are more interpretable, and have better predictive performance on a variety of tasks.

{{< /ci-details >}}

{{< ci-details summary="HGMF: Heterogeneous Graph-based Fusion for Multimodal Data with Incompleteness (Jiayi Chen et al., 2020)">}}

Jiayi Chen, Aidong Zhang. (2020)  
**HGMF: Heterogeneous Graph-based Fusion for Multimodal Data with Incompleteness**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/5d24501a99d05306817171a744878315c31a880b)  
Influential Citation Count (2), SS-ID (5d24501a99d05306817171a744878315c31a880b)  

**ABSTRACT**  
With the advances in data collection techniques, large amounts of multimodal data collected from multiple sources are becoming available. Such multimodal data can provide complementary information that can reveal fundamental characteristics of real-world subjects. Thus, multimodal machine learning has become an active research area. Extensive works have been developed to exploit multimodal interactions and integrate multi-source information. However, multimodal data in the real world usually comes with missing modalities due to various reasons, such as sensor damage, data corruption, and human mistakes in recording. Effectively integrating and analyzing multimodal data with incompleteness remains a challenging problem. We propose a Heterogeneous Graph-based Multimodal Fusion (HGMF) approach to enable multimodal fusion of incomplete data within a heterogeneous graph structure. The proposed approach develops a unique strategy for learning on incomplete multimodal data without data deletion or data imputation. More specifically, we construct a heterogeneous hypernode graph to model the multimodal data having different combinations of missing modalities, and then we formulate a graph neural network based transductive learning framework to project the heterogeneous incomplete data onto a unified embedding space, and multi-modalities are fused along the way. The learning framework captures modality interactions from available data, and leverages the relationships between different incompleteness patterns. Our experimental results demonstrate that the proposed method outperforms existing graph-based as well as non-graph based baselines on three different datasets.

{{< /ci-details >}}

{{< ci-details summary="Max-Margin DeepWalk: Discriminative Learning of Network Representation (Cunchao Tu et al., 2016)">}}

Cunchao Tu, Weicheng Zhang, Zhiyuan Liu, Maosong Sun. (2016)  
**Max-Margin DeepWalk: Discriminative Learning of Network Representation**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/5d66991e1f541a08e81e59060cb0bb7f6931c2d9)  
Influential Citation Count (29), SS-ID (5d66991e1f541a08e81e59060cb0bb7f6931c2d9)  

**ABSTRACT**  
DeepWalk is a typical representation learning method that learns low-dimensional representations for vertices in social networks. Similar to other network representation learning (NRL) models, it encodes the network structure into vertex representations and is learnt in unsupervised form. However, the learnt representations usually lack the ability of discrimination when applied to machine learning tasks, such as vertex classification. In this paper, we overcome this challenge by proposing a novel semi-supervised model, max-margin Deep-Walk (MMDW). MMDW is a unified NRL framework that jointly optimizes the max-margin classifier and the aimed social representation learning model. Influenced by the max-margin classifier, the learnt representations not only contain the network structure, but also have the characteristic of discrimination. The visualizations of learnt representations indicate that our model is more discriminative than unsupervised ones, and the experimental results on vertex classification demonstrate that our method achieves a significant improvement than other state-of-the-art methods. The source code can be obtained from https://github.com/thunlp/MMDW.

{{< /ci-details >}}

{{< ci-details summary="OpenGraphGym: A Parallel Reinforcement Learning Framework for Graph Optimization Problems (Weijian Zheng et al., 2020)">}}

Weijian Zheng, Dali Wang, Fengguang Song. (2020)  
**OpenGraphGym: A Parallel Reinforcement Learning Framework for Graph Optimization Problems**  
ICCS  
[Paper Link](https://www.semanticscholar.org/paper/5e46f4d777670116523ab4c6cb9d58cf20c38e73)  
Influential Citation Count (1), SS-ID (5e46f4d777670116523ab4c6cb9d58cf20c38e73)  

**ABSTRACT**  
This paper presents an open-source, parallel AI environment (named OpenGraphGym) to facilitate the application of reinforcement learning (RL) algorithms to address combinatorial graph optimization problems. This environment incorporates a basic deep reinforcement learning method, and several graph embeddings to capture graph features, it also allows users to rapidly plug in and test new RL algorithms and graph embeddings for graph optimization problems. This new open-source RL framework is targeted at achieving both high performance and high quality of the computed graph solutions. This RL framework forms the foundation of several ongoing research directions, including 1) benchmark works on different RL algorithms and embedding methods for classic graph problems; 2) advanced parallel strategies for extreme-scale graph computations, as well as 3) performance evaluation on real-world graph solutions.

{{< /ci-details >}}

{{< ci-details summary="Spectral Networks and Locally Connected Networks on Graphs (Joan Bruna et al., 2013)">}}

Joan Bruna, Wojciech Zaremba, Arthur D. Szlam, Yann LeCun. (2013)  
**Spectral Networks and Locally Connected Networks on Graphs**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/5e925a9f1e20df61d1e860a7aa71894b35a1c186)  
Influential Citation Count (262), SS-ID (5e925a9f1e20df61d1e860a7aa71894b35a1c186)  

**ABSTRACT**  
Convolutional Neural Networks are extremely efficient architectures in image and audio recognition tasks, thanks to their ability to exploit the local translational invariance of signal classes over their domain. In this paper we consider possible generalizations of CNNs to signals defined on more general domains without the action of a translation group. In particular, we propose two constructions, one based upon a hierarchical clustering of the domain, and another based on the spectrum of the graph Laplacian. We show through experiments that for low-dimensional graphs it is possible to learn convolutional layers with a number of parameters independent of the input size, resulting in efficient deep architectures.

{{< /ci-details >}}

{{< ci-details summary="GraphSeq2Seq: Graph-Sequence-to-Sequence for Neural Machine Translation (Guoshuai Zhao et al., 2018)">}}

Guoshuai Zhao, Jun Yu Li, Lu Wang, Xueming Qian, Y. Fu. (2018)  
**GraphSeq2Seq: Graph-Sequence-to-Sequence for Neural Machine Translation**  
  
[Paper Link](https://www.semanticscholar.org/paper/5f2a156909e2550cdc09b7d3d3d503ec5d52b1d7)  
Influential Citation Count (1), SS-ID (5f2a156909e2550cdc09b7d3d3d503ec5d52b1d7)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Auto-Encoding Variational Bayes (Diederik P. Kingma et al., 2013)">}}

Diederik P. Kingma, M. Welling. (2013)  
**Auto-Encoding Variational Bayes**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/5f5dc5b9a2ba710937e2c413b37b053cd673df02)  
Influential Citation Count (3503), SS-ID (5f5dc5b9a2ba710937e2c413b37b053cd673df02)  

**ABSTRACT**  
Abstract: How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions is two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.

{{< /ci-details >}}

{{< ci-details summary="Distance-Aware DAG Embedding for Proximity Search on Heterogeneous Graphs (Zemin Liu et al., 2018)">}}

Zemin Liu, V. Zheng, Zhou Zhao, Fanwei Zhu, K. Chang, Minghui Wu, Jing Ying. (2018)  
**Distance-Aware DAG Embedding for Proximity Search on Heterogeneous Graphs**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/6065a29041525360665320fab231dd9e5ca82ab8)  
Influential Citation Count (3), SS-ID (6065a29041525360665320fab231dd9e5ca82ab8)  

**ABSTRACT**  
Proximity search on heterogeneous graphs aims to measure the proximity between two nodes on a graph w.r.t. some semantic relation for ranking. Pioneer work often tries to measure such proximity by paths connecting the two nodes. However, paths as linear sequences have limited expressiveness for the complex network connections. In this paper, we explore a more expressive DAG (directed acyclic graph) data structure for modeling the connections between two nodes. Particularly, we are interested in learning a representation for the DAGs to encode the proximity between two nodes. We face two challenges to use DAGs, including how to efficiently generate DAGs and how to effectively learn DAG embedding for proximity search. We find distance-awareness as important for proximity search and the key to solve the above challenges. Thus we develop a novel Distance-aware DAG Embedding (D2AGE) model. We evaluate D2AGE on three benchmark data sets with six semantic relations, and we show that D2AGE outperforms the state-of-the-art baselines. We release the code on https://github.com/shuaiOKshuai.

{{< /ci-details >}}

{{< ci-details summary="Complex and Holographic Embeddings of Knowledge Graphs: A Comparison (Théo Trouillon et al., 2017)">}}

Théo Trouillon, Maximilian Nickel. (2017)  
**Complex and Holographic Embeddings of Knowledge Graphs: A Comparison**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/607e0bafc04bcc93089d25fed6d2ba1a41637ed7)  
Influential Citation Count (6), SS-ID (607e0bafc04bcc93089d25fed6d2ba1a41637ed7)  

**ABSTRACT**  
Embeddings of knowledge graphs have received significant attention due to their excellent performance for tasks like link prediction and entity resolution. In this short paper, we are providing a comparison of two state-of-the-art knowledge graph embeddings for which their equivalence has recently been established, i.e., ComplEx and HolE [Nickel, Rosasco, and Poggio, 2016; Trouillon et al., 2016; Hayashi and Shimbo, 2017]. First, we briefly review both models and discuss how their scoring functions are equivalent. We then analyze the discrepancy of results reported in the original articles, and show experimentally that they are likely due to the use of different loss functions. In further experiments, we evaluate the ability of both models to embed symmetric and antisymmetric patterns. Finally, we discuss advantages and disadvantages of both models and under which conditions one would be preferable to the other.

{{< /ci-details >}}

{{< ci-details summary="Graph Embedding Based on Characteristic of Rooted Subgraph Structure (Yan Liu et al., 2020)">}}

Yan Liu, Xiaokun Zhang, Lian Liu, Gaojian Li. (2020)  
**Graph Embedding Based on Characteristic of Rooted Subgraph Structure**  
KSEM  
[Paper Link](https://www.semanticscholar.org/paper/60e8a34070dbbb8ef1b3ca4e789d20dd7c826ded)  
Influential Citation Count (0), SS-ID (60e8a34070dbbb8ef1b3ca4e789d20dd7c826ded)  

**ABSTRACT**  
Given the problem that currently distributed graph embedding models have not yet been effectively modeled of substructure similarity, biased-graph2vec, a graph embedding model based on structural characteristics of rooted subgraphs is proposed in this paper. This model, based on the distributed representation model of the graph, has modified its original random walk process and converted it to a random walk with weight bias based on structural similarity. The appropriate context is generated for all substructures. Based on preserving the tag features of the nodes and edges in the substructure, the representation of the substructure in the feature space depends more on the structural similarity itself. Biased-graph2vec calculates the graph representations with unsupervised algorithm and could build the model for both graphs and substructures via universal models, leaving complex feature engineering behind and has functional mobility. Meanwhile, this method models similar information among substructures, solving the problem that typical random walk strategies could not capture similarities of substructures with long distance. The experiments of graph classification are carried out on six open benchmark datasets. The comparison among our method, the graph kernel method, and the baseline method without considering the structural similarity of long-distance ions is made. Experiments show that the method this paper proposed has varying degrees inordinately improved the accuracy of classification tasks.

{{< /ci-details >}}

{{< ci-details summary="MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment (Da Zhang et al., 2018)">}}

Da Zhang, Xiyang Dai, Xin Eric Wang, Yuan-fang Wang, L. Davis. (2018)  
**MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/613f59279586bd53aed57bc133246a4eb3c38977)  
Influential Citation Count (42), SS-ID (613f59279586bd53aed57bc133246a4eb3c38977)  

**ABSTRACT**  
This research strives for natural language moment retrieval in long, untrimmed video streams. The problem is not trivial especially when a video contains multiple moments of interests and the language describes complex temporal dependencies, which often happens in real scenarios. We identify two crucial challenges: semantic misalignment and structural misalignment. However, existing approaches treat different moments separately and do not explicitly model complex moment-wise temporal relations. In this paper, we present Moment Alignment Network (MAN), a novel framework that unifies the candidate moment encoding and temporal structural reasoning in a single-shot feed-forward network. MAN naturally assigns candidate moment representations aligned with language semantics over different temporal locations and scales. Most importantly, we propose to explicitly model moment-wise temporal relations as a structured graph and devise an iterative graph adjustment network to jointly learn the best structure in an end-to-end manner. We evaluate the proposed approach on two challenging public benchmarks DiDeMo and Charades-STA, where our MAN significantly outperforms the state-of-the-art by a large margin.

{{< /ci-details >}}

{{< ci-details summary="Multi-View Clustering and Semi-Supervised Classification with Adaptive Neighbours (F. Nie et al., 2017)">}}

F. Nie, Guohao Cai, Xuelong Li. (2017)  
**Multi-View Clustering and Semi-Supervised Classification with Adaptive Neighbours**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/62b5d514ec49173af59cab6f0dfdc7f280d53f36)  
Influential Citation Count (34), SS-ID (62b5d514ec49173af59cab6f0dfdc7f280d53f36)  

**ABSTRACT**  
Due to the efficiency of learning relationships and complex structures hidden in data, graph-oriented methods have been widely investigated and achieve promising performance in multi-view learning. Generally, these learning algorithms construct informative graph for each view or fuse different views to one graph, on which the following procedure are based. However, in many real world dataset, original data always contain noise and outlying entries that result in unreliable and inaccurate graphs, which cannot be ameliorated in the previous methods. In this paper, we propose a novel multi-view learning model which performs clustering/semi-supervised classification and local structure learning simultaneously. The obtained optimal graph can be partitioned into specific clusters directly. Moreover, our model can allocate ideal weight for each view automatically without additional weight and penalty parameters. An efficient algorithm is proposed to optimize this model. Extensive experimental results on different real-world datasets show that the proposed model outperforms other state-of-the-art multi-view algorithms.

{{< /ci-details >}}

{{< ci-details summary="Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks (Diego Marcheggiani et al., 2018)">}}

Diego Marcheggiani, Jasmijn Bastings, Ivan Titov. (2018)  
**Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/6411da05a0e6f3e38bcac0ce57c28038ff08081c)  
Influential Citation Count (8), SS-ID (6411da05a0e6f3e38bcac0ce57c28038ff08081c)  

**ABSTRACT**  
Semantic representations have long been argued as potentially useful for enforcing meaning preservation and improving generalization performance of machine translation methods. In this work, we are the first to incorporate information about predicate-argument structure of source sentences (namely, semantic-role representations) into neural machine translation. We use Graph Convolutional Networks (GCNs) to inject a semantic bias into sentence encoders and achieve improvements in BLEU scores over the linguistic-agnostic and syntax-aware versions on the English–German language pair.

{{< /ci-details >}}

{{< ci-details summary="Tracking network dynamics: A survey using graph distances (C. Donnat et al., 2018)">}}

C. Donnat, S. Holmes. (2018)  
**Tracking network dynamics: A survey using graph distances**  
The Annals of Applied Statistics  
[Paper Link](https://www.semanticscholar.org/paper/64aa05ee62ed2c55e002acdcdeadd29daefe9426)  
Influential Citation Count (2), SS-ID (64aa05ee62ed2c55e002acdcdeadd29daefe9426)  

**ABSTRACT**  
From longitudinal biomedical studies to social networks, graphs have emerged as essential objects for describing evolving interactions between agents in complex systems. In such studies, after pre-processing, the data are encoded by a set of graphs, each representing a system’s state at a different point in time or space. The analysis of the system’s dynamics depends on the selection of the appropriate analytical tools. In particular, after specifying properties characterizing similarities between states, a critical step lies in the choice of a distance between graphs capable of reflecting such similarities. While the literature offers a number of distances to choose from, their properties have been little investigated and no guidelines regarding the choice of such a distance have yet been provided. In particular, most graph distances consider that the nodes are exchangeable—ignoring node “identities.” Alignment of the graphs according to identified nodes enables us to enhance these distances’ sensitivity to perturbations in the network and detect important changes in graph dynamics. Thus the selection of an adequate metric is a decisive—yet delicate—practical matter. In the spirit of Goldenberg et al.’s seminal 2009 review [Found. Trends Mach. Learn. 2 (2010) 129–233], this article provides an overview of commonly-used graph distances and an explicit characterization of the structural changes that they are best able to capture. We show how these choices affect real-life situations, and we use these distances to analyze both a longitudinal microbiome dataset and a brain fMRI study. One contribution of the present study is a coordinated suite of data analytic techniques, displays and statistical tests using “metagraphs”: a graph of graphs based on a chosen metric. Permutation tests can uncover the effects of covariates on the graphs’ variability. Furthermore, synthetic examples provide intuition as to the qualities and drawbacks of the different distances. Above all, we provide some guidance on choosing one distance over another in different contexts. Finally, we extend the scope of our analyses from temporal to spatial dynamics and apply these different distances to a network created from worldwide recipes.

{{< /ci-details >}}

{{< ci-details summary="Linkage Based Face Clustering via Graph Convolution Network (Zhongdao Wang et al., 2019)">}}

Zhongdao Wang, Liang Zheng, Yali Li, Shengjin Wang. (2019)  
**Linkage Based Face Clustering via Graph Convolution Network**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/6834b6a529c969e5feb1fb77713eff8f19704b31)  
Influential Citation Count (18), SS-ID (6834b6a529c969e5feb1fb77713eff8f19704b31)  

**ABSTRACT**  
In this paper, we present an accurate and scalable approach to the face clustering task. We aim at grouping a set of faces by their potential identities. We formulate this task as a link prediction problem: a link exists between two faces if they are of the same identity. The key idea is that we find the local context in the feature space around an instance (face) contains rich information about the linkage relationship between this instance and its neighbors. By constructing sub-graphs around each instance as input data, which depict the local context, we utilize the graph convolution network (GCN) to perform reasoning and infer the likelihood of linkage between pairs in the sub-graphs. Experiments show that our method is more robust to the complex distribution of faces than conventional methods, yielding favorably comparable results to state-of-the-art methods on standard face clustering benchmarks, and is scalable to large datasets. Furthermore, we show that the proposed method does not need the number of clusters as prior, is aware of noises and outliers, and can be extended to a multi-view version for more accurate clustering accuracy.

{{< /ci-details >}}

{{< ci-details summary="Embedding of Embedding (EOE): Joint Embedding for Coupled Heterogeneous Networks (Linchuan Xu et al., 2017)">}}

Linchuan Xu, Xiaokai Wei, Jiannong Cao, Philip S. Yu. (2017)  
**Embedding of Embedding (EOE): Joint Embedding for Coupled Heterogeneous Networks**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/688f937ddeed178802c53963743d1801a778614e)  
Influential Citation Count (14), SS-ID (688f937ddeed178802c53963743d1801a778614e)  

**ABSTRACT**  
Network embedding is increasingly employed to assist network analysis as it is effective to learn latent features that encode linkage information. Various network embedding methods have been proposed, but they are only designed for a single network scenario. In the era of big data, different types of related information can be fused together to form a coupled heterogeneous network, which consists of two different but related sub-networks connected by inter-network edges. In this scenario, the inter-network edges can act as comple- mentary information in the presence of intra-network ones. This complementary information is important because it can make latent features more comprehensive and accurate. And it is more important when the intra-network edges are ab- sent, which can be referred to as the cold-start problem. In this paper, we thus propose a method named embedding of embedding (EOE) for coupled heterogeneous networks. In the EOE, latent features encode not only intra-network edges, but also inter-network ones. To tackle the challenge of heterogeneities of two networks, the EOE incorporates a harmonious embedding matrix to further embed the em- beddings that only encode intra-network edges. Empirical experiments on a variety of real-world datasets demonstrate the EOE outperforms consistently single network embedding methods in applications including visualization, link prediction multi-class classification, and multi-label classification.

{{< /ci-details >}}

{{< ci-details summary="Skeleton-Based Action Recognition With Directed Graph Neural Networks (Lei Shi et al., 2019)">}}

Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu. (2019)  
**Skeleton-Based Action Recognition With Directed Graph Neural Networks**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/68a024d7b70ef3989a6751678f635cbe754440fc)  
Influential Citation Count (34), SS-ID (68a024d7b70ef3989a6751678f635cbe754440fc)  

**ABSTRACT**  
The skeleton data have been widely used for the action recognition tasks since they can robustly accommodate dynamic circumstances and complex backgrounds. In existing methods, both the joint and bone information in skeleton data have been proved to be of great help for action recognition tasks. However, how to incorporate these two types of data to best take advantage of the relationship between joints and bones remains a problem to be solved. In this work, we represent the skeleton data as a directed acyclic graph based on the kinematic dependency between the joints and bones in the natural human body. A novel directed graph neural network is designed specially to extract the information of joints, bones and their relations and make prediction based on the extracted features. In addition, to better fit the action recognition task, the topological structure of the graph is made adaptive based on the training process, which brings notable improvement. Moreover, the motion information of the skeleton sequence is exploited and combined with the spatial information to further enhance the performance in a two-stream framework. Our final model is tested on two large-scale datasets, NTU-RGBD and Skeleton-Kinetics, and exceeds state-of-the-art performance on both of them.

{{< /ci-details >}}

{{< ci-details summary="Multi-Modal Bayesian Embeddings for Learning Social Knowledge Graphs (Zhilin Yang et al., 2015)">}}

Zhilin Yang, Jie Tang, William W. Cohen. (2015)  
**Multi-Modal Bayesian Embeddings for Learning Social Knowledge Graphs**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/695d4c04f6e4f7ba5f771ac7853fdbaa81713ae8)  
Influential Citation Count (3), SS-ID (695d4c04f6e4f7ba5f771ac7853fdbaa81713ae8)  

**ABSTRACT**  
We study the extent to which online social networks can be connected to open knowledge bases. The problem is referred to as learning social knowledge graphs. We propose a multi-modal Bayesian embedding model, GenVector, to learn latent topics that generate word and network embeddings. GenVector leverages large-scale unlabeled data with embeddings and represents data of two modalities---i.e., social network users and knowledge concepts---in a shared latent topic space. Experiments on three datasets show that the proposed method clearly outperforms state-of-the-art methods. We then deploy the method on AMiner, a large-scale online academic search system with a network of 38,049,189 researchers with a knowledge base with 35,415,011 concepts. Our method significantly decreases the error rate in an online A/B test with live users.

{{< /ci-details >}}

{{< ci-details summary="Task-Guided and Path-Augmented Heterogeneous Network Embedding for Author Identification (Ting Chen et al., 2016)">}}

Ting Chen, Yizhou Sun. (2016)  
**Task-Guided and Path-Augmented Heterogeneous Network Embedding for Author Identification**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/6b183d2297cb493a57dbc875689ab2430d870043)  
Influential Citation Count (13), SS-ID (6b183d2297cb493a57dbc875689ab2430d870043)  

**ABSTRACT**  
In this paper, we study the problem of author identification under double-blind review setting, which is to identify potential authors given information of an anonymized paper. Different from existing approaches that rely heavily on feature engineering, we propose to use network embedding approach to address the problem, which can automatically represent nodes into lower dimensional feature vectors. However, there are two major limitations in recent studies on network embedding: (1) they are usually general-purpose embedding methods, which are independent of the specific tasks; and (2) most of these approaches can only deal with homogeneous networks, where the heterogeneity of the network is ignored. Hence, challenges faced here are two folds: (1) how to embed the network under the guidance of the author identification task, and (2) how to select the best type of information due to the heterogeneity of the network. To address the challenges, we propose a task-guided and path-augmented heterogeneous network embedding model. In our model, nodes are first embedded as vectors in latent feature space. Embeddings are then shared and jointly trained according to task-specific and network-general objectives. We extend the existing unsupervised network embedding to incorporate meta paths in heterogeneous networks, and select paths according to the specific task. The guidance from author identification task for network embedding is provided both explicitly in joint training and implicitly during meta path selection. Our experiments demonstrate that by using path-augmented network embedding with task guidance, our model can obtain significantly better accuracy at identifying the true authors comparing to existing methods.

{{< /ci-details >}}

{{< ci-details summary="Inductive Representation Learning on Large Graphs (William L. Hamilton et al., 2017)">}}

William L. Hamilton, Z. Ying, J. Leskovec. (2017)  
**Inductive Representation Learning on Large Graphs**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/6b7d6e6416343b2a122f8416e69059ce919026ef)  
Influential Citation Count (1254), SS-ID (6b7d6e6416343b2a122f8416e69059ce919026ef)  

**ABSTRACT**  
Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

{{< /ci-details >}}

{{< ci-details summary="Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs (Anton Tsitsulin et al., 2020)">}}

Anton Tsitsulin, Marina Munkhoeva, Bryan Perozzi. (2020)  
**Just SLaQ When You Approximate: Accurate Spectral Distances for Web-Scale Graphs**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/6bcea47afc6fcdada957e8d72b9b27b7866bf535)  
Influential Citation Count (3), SS-ID (6bcea47afc6fcdada957e8d72b9b27b7866bf535)  

**ABSTRACT**  
Graph comparison is a fundamental operation in data mining and information retrieval. Due to the combinatorial nature of graphs, it is hard to balance the expressiveness of the similarity measure and its scalability. Spectral analysis provides quintessential tools for studying the multi-scale structure of graphs and is a well-suited foundation for reasoning about differences between graphs. However, computing full spectrum of large graphs is computationally prohibitive; thus, spectral graph comparison methods often rely on rough approximation techniques with weak error guarantees. In this work, we propose SLaQ , an efficient and effective approximation technique for computing spectral distances between graphs with billions of nodes and edges. We derive the corresponding error bounds and demonstrate that accurate computation is possible in time linear in the number of graph edges. In a thorough experimental evaluation, we show that SLaQ outperforms existing methods, oftentimes by several orders of magnitude in approximation accuracy, and maintains comparable performance, allowing to compare million-scale graphs in a matter of minutes on a single machine.

{{< /ci-details >}}

{{< ci-details summary="StructPool: Structured Graph Pooling via Conditional Random Fields (Hao Yuan et al., 2020)">}}

Hao Yuan, Shuiwang Ji. (2020)  
**StructPool: Structured Graph Pooling via Conditional Random Fields**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/6c252187647a437b32b163a295d62b65cda6e0fe)  
Influential Citation Count (9), SS-ID (6c252187647a437b32b163a295d62b65cda6e0fe)  

**ABSTRACT**  
Learning high-level representations for graphs is of great importance for graph analysis tasks. In addition to graph convolution, graph pooling is an important but less explored research area. In particular, most of existing graph pooling techniques do not consider the graph structural information explicitly. We argue that such information is important and develop a novel graph pooling technique, know as the StructPool, in this work. We consider the graph pooling as a node clustering problem, which requires the learning of a cluster assignment matrix. We propose to formulate it as a structured prediction problem and employ conditional random fields to capture the relationships among assignments of different nodes. We also generalize our method to incorporate graph topological information in designing the Gibbs energy function. Experimental results on multiple datasets demonstrate the effectiveness of our proposed StructPool.

{{< /ci-details >}}

{{< ci-details summary="Graph Convolutional Neural Networks for Web-Scale Recommender Systems (Rex Ying et al., 2018)">}}

Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, J. Leskovec. (2018)  
**Graph Convolutional Neural Networks for Web-Scale Recommender Systems**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/6c96c2d4a3fbd572fef2d59cb856521ee1746789)  
Influential Citation Count (128), SS-ID (6c96c2d4a3fbd572fef2d59cb856521ee1746789)  

**ABSTRACT**  
Recent advancements in deep neural networks for graph-structured data have led to state-of-the-art performance on recommender system benchmarks. However, making these methods practical and scalable to web-scale recommendation tasks with billions of items and hundreds of millions of users remains an unsolved challenge. Here we describe a large-scale deep recommendation engine that we developed and deployed at Pinterest. We develop a data-efficient Graph Convolutional Network (GCN) algorithm, which combines efficient random walks and graph convolutions to generate embeddings of nodes (i.e., items) that incorporate both graph structure as well as node feature information. Compared to prior GCN approaches, we develop a novel method based on highly efficient random walks to structure the convolutions and design a novel training strategy that relies on harder-and-harder training examples to improve robustness and convergence of the model. We also develop an efficient MapReduce model inference algorithm to generate embeddings using a trained model. Overall, we can train on and embed graphs that are four orders of magnitude larger than typical GCN implementations. We show how GCN embeddings can be used to make high-quality recommendations in various settings at Pinterest, which has a massive underlying graph with 3 billion nodes representing pins and boards, and 17 billion edges. According to offline metrics, user studies, as well as A/B tests, our approach generates higher-quality recommendations than comparable deep learning based systems. To our knowledge, this is by far the largest application of deep graph embeddings to date and paves the way for a new generation of web-scale recommender systems based on graph convolutional architectures.

{{< /ci-details >}}

{{< ci-details summary="A General View for Network Embedding as Matrix Factorization (Xin Liu et al., 2019)">}}

Xin Liu, T. Murata, Kyoung-Sook Kim, Chatchawan Kotarasu, Chenyi Zhuang. (2019)  
**A General View for Network Embedding as Matrix Factorization**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/6cb5a18edade451793d6958bc3e2a16b938dc48d)  
Influential Citation Count (1), SS-ID (6cb5a18edade451793d6958bc3e2a16b938dc48d)  

**ABSTRACT**  
We propose a general view that demonstrates the relationship between network embedding approaches and matrix factorization. Unlike previous works that present the equivalence for the approaches from a skip-gram model perspective, we provide a more fundamental connection from an optimization (objective function) perspective. We demonstrate that matrix factorization is equivalent to optimizing two objectives: one is for bringing together the embeddings of similar nodes; the other is for separating the embeddings of distant nodes. The matrix to be factorized has a general form: S-β. The elements of $\mathbfS $ indicate pairwise node similarities. They can be based on any user-defined similarity/distance measure or learned from random walks on networks. The shift number β is related to a parameter that balances the two objectives. More importantly, the resulting embeddings are sensitive to β and we can improve the embeddings by tuning β. Experiments show that matrix factorization based on a new proposed similarity measure and β-tuning strategy significantly outperforms existing matrix factorization approaches on a range of benchmark networks.

{{< /ci-details >}}

{{< ci-details summary="Discovering missing links in Wikipedia (S. F. Adafre et al., 2005)">}}

S. F. Adafre, M. de Rijke. (2005)  
**Discovering missing links in Wikipedia**  
LinkKDD '05  
[Paper Link](https://www.semanticscholar.org/paper/6d9064ff94c5186e12c39ea2e9f3815004066e51)  
Influential Citation Count (12), SS-ID (6d9064ff94c5186e12c39ea2e9f3815004066e51)  

**ABSTRACT**  
In this paper we address the problem of discovering missing hypertext links in Wikipedia. The method we propose consists of two steps: first, we compute a cluster of highly similar pages around a given page, and then we identify candidate links from those similar pages that might be missing on the given page. The main innovation is in the algorithm that we use for identifying similar pages, LTRank, which ranks pages using co-citation and page title information. Both LTRank and the link discovery method are manually evaluated and show acceptable results, especially given the simplicity of the methods and conservativeness of the evaluation criteria.

{{< /ci-details >}}

{{< ci-details summary="Node Classification in Social Networks (Smriti Bhagat et al., 2011)">}}

Smriti Bhagat, Graham Cormode, S. Muthukrishnan. (2011)  
**Node Classification in Social Networks**  
Social Network Data Analytics  
[Paper Link](https://www.semanticscholar.org/paper/6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  
Influential Citation Count (11), SS-ID (6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  

**ABSTRACT**  
When dealing with large graphs, such as those that arise in the context of online social networks, a subset of nodes may be labeled. These labels can indicate demographic values, interest, beliefs or other characteristics of the nodes (users). A core problem is to use this information to extend the labeling so that all nodes are assigned a label (or labels).

{{< /ci-details >}}

{{< ci-details summary="Efficient aggregation for graph summarization (Yuanyuan Tian et al., 2008)">}}

Yuanyuan Tian, R. Hankins, J. Patel. (2008)  
**Efficient aggregation for graph summarization**  
SIGMOD Conference  
[Paper Link](https://www.semanticscholar.org/paper/6e9cf091dd709b557b32e7239647753680f0645b)  
Influential Citation Count (28), SS-ID (6e9cf091dd709b557b32e7239647753680f0645b)  

**ABSTRACT**  
Graphs are widely used to model real world objects and their relationships, and large graph datasets are common in many application domains. To understand the underlying characteristics of large graphs, graph summarization techniques are critical. However, existing graph summarization methods are mostly statistical (studying statistics such as degree distributions, hop-plots and clustering coefficients). These statistical methods are very useful, but the resolutions of the summaries are hard to control.  In this paper, we introduce two database-style operations to summarize graphs. Like the OLAP-style aggregation methods that allow users to drill-down or roll-up to control the resolution of summarization, our methods provide an analogous functionality for large graph datasets. The first operation, called SNAP, produces a summary graph by grouping nodes based on user-selected node attributes and relationships. The second operation, called k-SNAP, further allows users to control the resolutions of summaries and provides the "drill-down" and "roll-up" abilities to navigate through summaries with different resolutions. We propose an efficient algorithm to evaluate the SNAP operation. In addition, we prove that the k-SNAP computation is NP-complete. We propose two heuristic methods to approximate the k-SNAP results. Through extensive experiments on a variety of real and synthetic datasets, we demonstrate the effectiveness and efficiency of the proposed methods.

{{< /ci-details >}}

{{< ci-details summary="A min-max cut algorithm for graph partitioning and data clustering (C. Ding et al., 2001)">}}

C. Ding, Xiaofeng He, H. Zha, Ming Gu, H. Simon. (2001)  
**A min-max cut algorithm for graph partitioning and data clustering**  
Proceedings 2001 IEEE International Conference on Data Mining  
[Paper Link](https://www.semanticscholar.org/paper/6ebb015ac7f7872ecadd75b837e859621abd0751)  
Influential Citation Count (43), SS-ID (6ebb015ac7f7872ecadd75b837e859621abd0751)  

**ABSTRACT**  
An important application of graph partitioning is data clustering using a graph model - the pairwise similarities between all data objects form a weighted graph adjacency matrix that contains all necessary information for clustering. In this paper, we propose a new algorithm for graph partitioning with an objective function that follows the min-max clustering principle. The relaxed version of the optimization of the min-max cut objective function leads to the Fiedler vector in spectral graph partitioning. Theoretical analyses of min-max cut indicate that it leads to balanced partitions, and lower bounds are derived. The min-max cut algorithm is tested on newsgroup data sets and is found to out-perform other current popular partitioning/clustering methods. The linkage-based refinements to the algorithm further improve the quality of clustering substantially. We also demonstrate that a linearized search order based on linkage differential is better than that based on the Fiedler vector, providing another effective partitioning method.

{{< /ci-details >}}

{{< ci-details summary="miRNA-Disease Association Prediction with Collaborative Matrix Factorization (Zhen Shen et al., 2017)">}}

Zhen Shen, You-Hua Zhang, Kyungsook Han, A. Nandi, B. Honig, De-shuang Huang. (2017)  
**miRNA-Disease Association Prediction with Collaborative Matrix Factorization**  
Complex.  
[Paper Link](https://www.semanticscholar.org/paper/6ececc7f15b281f1116f8197fc174f1c6f8b0e51)  
Influential Citation Count (3), SS-ID (6ececc7f15b281f1116f8197fc174f1c6f8b0e51)  

**ABSTRACT**  
As one of the factors in the noncoding RNA family, microRNAs (miRNAs) are involved in the development and progression of various complex diseases. Experimental identification of miRNA-disease association is expensive and time-consuming. Therefore, it is necessary to design efficient algorithms to identify novel miRNA-disease association. In this paper, we developed the computational method of Collaborative Matrix Factorization for miRNA-Disease Association prediction (CMFMDA) to identify potential miRNA-disease associations by integrating miRNA functional similarity, disease semantic similarity, and experimentally verified miRNA-disease associations. Experiments verified that CMFMDA achieves intended purpose and application values with its short consuming-time and high prediction accuracy. In addition, we used CMFMDA on Esophageal Neoplasms and Kidney Neoplasms to reveal their potential related miRNAs. As a result, 84% and 82% of top 50 predicted miRNA-disease pairs for these two diseases were confirmed by experiment. Not only this, but also CMFMDA could be applied to new diseases and new miRNAs without any known associations, which overcome the defects of many previous computational methods.

{{< /ci-details >}}

{{< ci-details summary="Can Adversarial Network Attack be Defended? (Jinyin Chen et al., 2019)">}}

Jinyin Chen, Yangyang Wu, Xiang Lin, Qi Xuan. (2019)  
**Can Adversarial Network Attack be Defended?**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/6ee2ac6e7179fe30065fbf60f8eff329624b9e85)  
Influential Citation Count (2), SS-ID (6ee2ac6e7179fe30065fbf60f8eff329624b9e85)  

**ABSTRACT**  
Machine learning has been successfully applied to complex network analysis in various areas, and graph neural networks (GNNs) based methods outperform others. Recently, adversarial attack on networks has attracted special attention since carefully crafted adversarial networks with slight perturbations on clean network may invalid lots of network applications, such as node classification, link prediction, and community detection etc. Such attacks are easily constructed with serious security threat to various analyze methods, including traditional methods and deep models. To the best of our knowledge, it is the first time that defense method against network adversarial attack is discussed. In this paper, we are interested in the possibility of defense against adversarial attack on network, and propose defense strategies for GNNs against attacks. First, we propose novel adversarial training strategies to improve GNNs' defensibility against attacks. Then, we analytically investigate the robustness properties for GNNs granted by the use of smooth defense, and propose two special smooth defense strategies: smoothing distillation and smoothing cross-entropy loss function. Both of them are capable of smoothing gradient of GNNs, and consequently reduce the amplitude of adversarial gradients, which benefits gradient masking from attackers. The comprehensive experiments show that our proposed strategies have great defensibility against different adversarial attacks on four real-world networks in different network analyze tasks.

{{< /ci-details >}}

{{< ci-details summary="Cauchy Graph Embedding (Dijun Luo et al., 2011)">}}

Dijun Luo, C. Ding, F. Nie, Heng Huang. (2011)  
**Cauchy Graph Embedding**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/6f390eee4c9a082e02843fb34046f653624e9b76)  
Influential Citation Count (3), SS-ID (6f390eee4c9a082e02843fb34046f653624e9b76)  

**ABSTRACT**  
Laplacian embedding provides a low-dimensional representation for the nodes of a graph where the edge weights denote pair-wise similarity among the node objects. It is commonly assumed that the Laplacian embedding results preserve the local topology of the original data on the low-dimensional projected subspaces, i.e., for any pair of graph nodes with large similarity, they should be embedded closely in the embedded space. However, in this paper, we will show that the Laplacian embedding often cannot preserve local topology well as we expected. To enhance the local topology preserving property in graph embedding, we propose a novel Cauchy graph embedding which preserves the similarity relationships of the original data in the embedded space via a new objective. Consequentially the machine learning tasks (such as k Nearest Neighbor type classifications) can be easily conducted on the embedded data with better performance. The experimental results on both synthetic and real world benchmark data sets demonstrate the usefulness of this new type of embedding.

{{< /ci-details >}}

{{< ci-details summary="Learning Edge Representations via Low-Rank Asymmetric Projections (Sami Abu-El-Haija et al., 2017)">}}

Sami Abu-El-Haija, Bryan Perozzi, Rami Al-Rfou. (2017)  
**Learning Edge Representations via Low-Rank Asymmetric Projections**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/6fcd6f350571ae102fea5315ecd3e9ca18814de6)  
Influential Citation Count (6), SS-ID (6fcd6f350571ae102fea5315ecd3e9ca18814de6)  

**ABSTRACT**  
We propose a new method for embedding graphs while preserving directed edge information. Learning such continuous-space vector representations (or embeddings) of nodes in a graph is an important first step for using network information (from social networks, user-item graphs, knowledge bases, etc.) in many machine learning tasks. Unlike previous work, we (1) explicitly model an edge as a function of node embeddings, and we (2) propose a novel objective, the graph likelihood, which contrasts information from sampled random walks with non-existent edges. Individually, both of these contributions improve the learned representations, especially when there are memory constraints on the total size of the embeddings. When combined, our contributions enable us to significantly improve the state-of-the-art by learning more concise representations that better preserve the graph structure. We evaluate our method on a variety of link-prediction task including social networks, collaboration networks, and protein interactions, showing that our proposed method learn representations with error reductions of up to 76% and 55%, on directed and undirected graphs. In addition, we show that the representations learned by our method are quite space efficient, producing embeddings which have higher structure-preserving accuracy but are 10 times smaller.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Learning on Heterogeneous Information Networks (Binbin Hu et al., 2019)">}}

Binbin Hu, Yuan Fang, C. Shi. (2019)  
**Adversarial Learning on Heterogeneous Information Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/721902635d4480b4e4a64e36441a0cf527f2dd02)  
Influential Citation Count (8), SS-ID (721902635d4480b4e4a64e36441a0cf527f2dd02)  

**ABSTRACT**  
Network embedding, which aims to represent network data in a low-dimensional space, has been commonly adopted for analyzing heterogeneous information networks (HIN). Although exiting HIN embedding methods have achieved performance improvement to some extent, they still face a few major weaknesses. Most importantly, they usually adopt negative sampling to randomly select nodes from the network, and they do not learn the underlying distribution for more robust embedding. Inspired by generative adversarial networks (GAN), we develop a novel framework HeGAN for HIN embedding, which trains both a discriminator and a generator in a minimax game. Compared to existing HIN embedding methods, our generator would learn the node distribution to generate better negative samples. Compared to GANs on homogeneous networks, our discriminator and generator are designed to be relation-aware in order to capture the rich semantics on HINs. Furthermore, towards more effective and efficient sampling, we propose a generalized generator, which samples "latent" nodes directly from a continuous distribution, not confined to the nodes in the original network as existing methods are. Finally, we conduct extensive experiments on four real-world datasets. Results show that we consistently and significantly outperform state-of-the-art baselines across all datasets and tasks.

{{< /ci-details >}}

{{< ci-details summary="VOPRec: Vector Representation Learning of Papers with Text Information and Structural Identity for Recommendation (Xiangjie Kong et al., 2021)">}}

Xiangjie Kong, Mengyi Mao, Wei Wang, Jiaying Liu, Bo Xu. (2021)  
**VOPRec: Vector Representation Learning of Papers with Text Information and Structural Identity for Recommendation**  
IEEE Transactions on Emerging Topics in Computing  
[Paper Link](https://www.semanticscholar.org/paper/722b747700f5f878af41570ee8b2a1c57baced45)  
Influential Citation Count (1), SS-ID (722b747700f5f878af41570ee8b2a1c57baced45)  

**ABSTRACT**  
Finding relevant papers is a non-trivial problem for scholars due to the tremendous amount of academic information in the era of scholarly big data. Scientific paper recommendation systems have been developed to solve such problem by recommending relevant papers to scholars. However, previous paper recommendations calculate paper similarity based on hand-engineered features which are inflexible. To address this problem, we develop a scientific paper recommendation system, namely VOPRec, by vector representation learning of paper in citation networks. VOPRec takes advantages of recent research in both text and network representation learning for unsupervised feature design. In VOPRec, the text information is represented with word embedding to find papers of similar research interest. Then, the structural identity is converted into vectors to find papers of similar network topology. After bridging text information and structural identity with the citation network, vector representation of paper can be learned with network embedding. Finally, top-<inline-formula><tex-math notation="LaTeX">$Q$</tex-math><alternatives><mml:math><mml:mi>Q</mml:mi></mml:math><inline-graphic xlink:href="xu-ieq1-2830698.gif"/></alternatives></inline-formula> recommendation list is generated based on the similarity calculated with paper vectors. Through the APS data set, we show that VOPRec outperforms state-of-the-art paper recommendation baselines measured by precision, recall, F1, and NDCG.

{{< /ci-details >}}

{{< ci-details summary="Query-based Music Recommendations via Preference Embedding (Chih-Ming Chen et al., 2016)">}}

Chih-Ming Chen, Ming-Feng Tsai, Yu-Ching Lin, Yi-Hsuan Yang. (2016)  
**Query-based Music Recommendations via Preference Embedding**  
RecSys  
[Paper Link](https://www.semanticscholar.org/paper/72728023c99d35fa884062841fd86661d296758b)  
Influential Citation Count (1), SS-ID (72728023c99d35fa884062841fd86661d296758b)  

**ABSTRACT**  
A common scenario considered in recommender systems is to predict a user's preferences on unseen items based on his/her preferences on observed items. A major limitation of this scenario is that a user might be interested in different things each time when using the system, but there is no way to allow the user to actively alter or adjust the recommended results. To address this issue, we propose the idea of "query-based recommendation" that allows a user to specify his/her search intention while exploring new items, thereby incorporating the concept of information retrieval into recommendation systems. Moreover, the idea is more desirable when the user intention can be expressed in different ways. Take music recommendation as an example: the proposed system allows a user to explore new song tracks by specifying either a track, an album, or an artist. To enable such heterogeneous queries in a recommender system, we present a novel technique called "Heterogeneous Preference Embedding" to encode user preference and query intention into low-dimensional vector spaces. Then, with simple search methods or similarity calculations, we can use the encoded representation of queries to generate recommendations. This method is fairly flexible and it is easy to add other types of information when available. Evaluations on three music listening datasets confirm the effectiveness of the proposed method over the state-of-the-art matrix factorization and network embedding methods.

{{< /ci-details >}}

{{< ci-details summary="Semantic manifold learning for image retrieval (Yen-Yu Lin et al., 2005)">}}

Yen-Yu Lin, Tyng-Luh Liu, Hwann-Tzong Chen. (2005)  
**Semantic manifold learning for image retrieval**  
MULTIMEDIA '05  
[Paper Link](https://www.semanticscholar.org/paper/7438604d467c64156fcb3e86556d80f0ca72342d)  
Influential Citation Count (12), SS-ID (7438604d467c64156fcb3e86556d80f0ca72342d)  

**ABSTRACT**  
Learning the user's semantics for CBIR involves two different sources of information: the similarity relations entailed by the content-based features, and the relevance relations specified in the feedback. Given that, we propose an augmented relation embedding (ARE) to map the image space into a semantic manifold that faithfully grasps the user's preferences. Besides ARE, we also look into the issues of selecting a good feature set for improving the retrieval performance. With these two aspects of efforts we have established a system that yields far better results than those previously reported. Overall, our approach can be characterized by three key properties: 1) The framework uses one relational graph to describe the similarity relations, and the other two to encode the relevant/irrelevant relations indicated in the feedback. 2) With the relational graphs so defined, learning a semantic manifold can be transformed into solving a constrained optimization problem, and is reduced to the ARE algorithm accounting for both the representation and the classification points of views. 3) An image representation based on augmented features is introduced to couple with the ARE learning. The use of these features is significant in capturing the semantics concerning different scales of image regions. We conclude with experimental results and comparisons to demonstrate the effectiveness of our method.

{{< /ci-details >}}

{{< ci-details summary="Learning Features from Large-Scale, Noisy and Social Image-Tag Collection (Hanwang Zhang et al., 2015)">}}

Hanwang Zhang, Xindi Shang, Huanbo Luan, Yang Yang, Tat-Seng Chua. (2015)  
**Learning Features from Large-Scale, Noisy and Social Image-Tag Collection**  
ACM Multimedia  
[Paper Link](https://www.semanticscholar.org/paper/7449864adc5d491fd0b2abf83a218429ce7834d4)  
Influential Citation Count (1), SS-ID (7449864adc5d491fd0b2abf83a218429ce7834d4)  

**ABSTRACT**  
Feature representation for multimedia content is the key to the progress of many fundamental multimedia tasks. Although recent advances in deep feature learning offer a promising route towards these tasks, they are limited in application to domains where high-quality and large-scale training data are hard to obtain. In this paper, we propose a novel deep feature learning paradigm based on large, noisy and social image-tag collections, which can be acquired from the inexhaustible social multimedia content on the Web. Instead of learning features from high-quality image-label supervision, we propose to learn from the image-word semantic relations, in a way of seeking a unified image-word embedding space, where the pairwise feature similarities preserve the semantic relations in the original image-word pairs. We offer an easy-to-use implementation for the proposed paradigm, which is fast and compatible for integrating into any state-of-the-art deep architectures. Experiments on NUSWIDE benchmark demonstrate that the features learned by our method significantly outperforms other state-of-the-art ones.

{{< /ci-details >}}

{{< ci-details summary="Locality Preserving Projections (Xiaofei He et al., 2003)">}}

Xiaofei He, P. Niyogi. (2003)  
**Locality Preserving Projections**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/75335244b49f4d1bb27aa51f1690bbefbbe1c3d1)  
Influential Citation Count (791), SS-ID (75335244b49f4d1bb27aa51f1690bbefbbe1c3d1)  

**ABSTRACT**  
Many problems in information processing involve some form of dimensionality reduction. In this paper, we introduce Locality Preserving Projections (LPP). These are linear projective maps that arise by solving a variational problem that optimally preserves the neighborhood structure of the data set. LPP should be seen as an alternative to Principal Component Analysis (PCA) – a classical linear technique that projects the data along the directions of maximal variance. When the high dimensional data lies on a low dimensional manifold embedded in the ambient space, the Locality Preserving Projections are obtained by finding the optimal linear approximations to the eigenfunctions of the Laplace Beltrami operator on the manifold. As a result, LPP shares many of the data representation properties of nonlinear techniques such as Laplacian Eigenmaps or Locally Linear Embedding. Yet LPP is linear and more crucially is defined everywhere in ambient space rather than just on the training data points. This is borne out by illustrative examples on some high dimensional data sets.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Embedding Based Question Answering (Xiao Huang et al., 2019)">}}

Xiao Huang, Jingyuan Zhang, Dingcheng Li, Ping Li. (2019)  
**Knowledge Graph Embedding Based Question Answering**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/7572aefcd241ec76341addcb2e2e417587cb2e4c)  
Influential Citation Count (12), SS-ID (7572aefcd241ec76341addcb2e2e417587cb2e4c)  

**ABSTRACT**  
Question answering over knowledge graph (QA-KG) aims to use facts in the knowledge graph (KG) to answer natural language questions. It helps end users more efficiently and more easily access the substantial and valuable knowledge in the KG, without knowing its data structures. QA-KG is a nontrivial problem since capturing the semantic meaning of natural language is difficult for a machine. Meanwhile, many knowledge graph embedding methods have been proposed. The key idea is to represent each predicate/entity as a low-dimensional vector, such that the relation information in the KG could be preserved. The learned vectors could benefit various applications such as KG completion and recommender systems. In this paper, we explore to use them to handle the QA-KG problem. However, this remains a challenging task since a predicate could be expressed in different ways in natural language questions. Also, the ambiguity of entity names and partial names makes the number of possible answers large. To bridge the gap, we propose an effective Knowledge Embedding based Question Answering (KEQA) framework. We focus on answering the most common types of questions, i.e., simple questions, in which each question could be answered by the machine straightforwardly if its single head entity and single predicate are correctly identified. To answer a simple question, instead of inferring its head entity and predicate directly, KEQA targets at jointly recovering the question's head entity, predicate, and tail entity representations in the KG embedding spaces. Based on a carefully-designed joint distance metric, the three learned vectors' closest fact in the KG is returned as the answer. Experiments on a widely-adopted benchmark demonstrate that the proposed KEQA outperforms the state-of-the-art QA-KG methods.

{{< /ci-details >}}

{{< ci-details summary="Flexible and robust co-regularized multi-domain graph clustering (Wei Cheng et al., 2013)">}}

Wei Cheng, X. Zhang, Zhishan Guo, Yubao Wu, P. Sullivan, Wei Wang. (2013)  
**Flexible and robust co-regularized multi-domain graph clustering**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/75938c4300e464f901a15332540d6500d957db91)  
Influential Citation Count (8), SS-ID (75938c4300e464f901a15332540d6500d957db91)  

**ABSTRACT**  
Multi-view graph clustering aims to enhance clustering performance by integrating heterogeneous information collected in different domains. Each domain provides a different view of the data instances. Leveraging cross-domain information has been demonstrated an effective way to achieve better clustering results. Despite the previous success, existing multi-view graph clustering methods usually assume that different views are available for the same set of instances. Thus instances in different domains can be treated as having strict one-to-one relationship. In many real-life applications, however, data instances in one domain may correspond to multiple instances in another domain. Moreover, relationships between instances in different domains may be associated with weights based on prior (partial) knowledge. In this paper, we propose a flexible and robust framework, CGC (Co-regularized Graph Clustering), based on non-negative matrix factorization (NMF), to tackle these challenges. CGC has several advantages over the existing methods. First, it supports many-to-many cross-domain instance relationship. Second, it incorporates weight on cross-domain relationship. Third, it allows partial cross-domain mapping so that graphs in different domains may have different sizes. Finally, it provides users with the extent to which the cross-domain instance relationship violates the in-domain clustering structure, and thus enables users to re-evaluate the consistency of the relationship. Extensive experimental results on UCI benchmark data sets, newsgroup data sets and biological interaction networks demonstrate the effectiveness of our approach.

{{< /ci-details >}}

{{< ci-details summary="Neural Embeddings of Graphs in Hyperbolic Space (B. Chamberlain et al., 2017)">}}

B. Chamberlain, J. Clough, M. Deisenroth. (2017)  
**Neural Embeddings of Graphs in Hyperbolic Space**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/76ce8d30c6f72aa56ae02cd42a064e41b9ab9391)  
Influential Citation Count (9), SS-ID (76ce8d30c6f72aa56ae02cd42a064e41b9ab9391)  

**ABSTRACT**  
Neural embeddings have been used with great success in Natural Language Processing (NLP). They provide compact representations that encapsulate word similarity and attain state-of-the-art performance in a range of linguistic tasks. The success of neural embeddings has prompted significant amounts of research into applications in domains other than language. One such domain is graph-structured data, where embeddings of vertices can be learned that encapsulate vertex similarity and improve performance on tasks including edge prediction and vertex labelling. For both NLP and graph based tasks, embeddings have been learned in high-dimensional Euclidean spaces. However, recent work has shown that the appropriate isometric space for embedding complex networks is not the flat Euclidean space, but negatively curved, hyperbolic space. We present a new concept that exploits these recent insights and propose learning neural embeddings of graphs in hyperbolic space. We provide experimental evidence that embedding graphs in their natural geometry significantly improves performance on downstream tasks for several real-world public datasets.

{{< /ci-details >}}

{{< ci-details summary="Neural networks for link prediction in realistic biomedical graphs: a multi-dimensional evaluation of graph embedding-based approaches (Gamal K. O. Crichton et al., 2018)">}}

Gamal K. O. Crichton, Yufan Guo, Sampo Pyysalo, A. Korhonen. (2018)  
**Neural networks for link prediction in realistic biomedical graphs: a multi-dimensional evaluation of graph embedding-based approaches**  
BMC Bioinformatics  
[Paper Link](https://www.semanticscholar.org/paper/7b64c1527ed63b57c0c9fde327bba1529775c5d3)  
Influential Citation Count (1), SS-ID (7b64c1527ed63b57c0c9fde327bba1529775c5d3)  

**ABSTRACT**  
BackgroundLink prediction in biomedical graphs has several important applications including predicting Drug-Target Interactions (DTI), Protein-Protein Interaction (PPI) prediction and Literature-Based Discovery (LBD). It can be done using a classifier to output the probability of link formation between nodes. Recently several works have used neural networks to create node representations which allow rich inputs to neural classifiers. Preliminary works were done on this and report promising results. However they did not use realistic settings like time-slicing, evaluate performances with comprehensive metrics or explain when or why neural network methods outperform. We investigated how inputs from four node representation algorithms affect performance of a neural link predictor on random- and time-sliced biomedical graphs of real-world sizes (∼ 6 million edges) containing information relevant to DTI, PPI and LBD. We compared the performance of the neural link predictor to those of established baselines and report performance across five metrics.ResultsIn random- and time-sliced experiments when the neural network methods were able to learn good node representations and there was a negligible amount of disconnected nodes, those approaches outperformed the baselines. In the smallest graph (∼ 15,000 edges) and in larger graphs with approximately 14% disconnected nodes, baselines such as Common Neighbours proved a justifiable choice for link prediction. At low recall levels (∼ 0.3) the approaches were mostly equal, but at higher recall levels across all nodes and average performance at individual nodes, neural network approaches were superior. Analysis showed that neural network methods performed well on links between nodes with no previous common neighbours; potentially the most interesting links. Additionally, while neural network methods benefit from large amounts of data, they require considerable amounts of computational resources to utilise them.ConclusionsOur results indicate that when there is enough data for the neural network methods to use and there are a negligible amount of disconnected nodes, those approaches outperform the baselines. At low recall levels the approaches are mostly equal but at higher recall levels and average performance at individual nodes, neural network approaches are superior. Performance at nodes without common neighbours which indicate more unexpected and perhaps more useful links account for this.

{{< /ci-details >}}

{{< ci-details summary="Social Attentional Memory Network: Modeling Aspect- and Friend-Level Differences in Recommendation (Chong Chen et al., 2019)">}}

Chong Chen, Min Zhang, Yiqun Liu, Shaoping Ma. (2019)  
**Social Attentional Memory Network: Modeling Aspect- and Friend-Level Differences in Recommendation**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/7c0afa4c7196474ff3b8c360bd6d4888f2417eed)  
Influential Citation Count (12), SS-ID (7c0afa4c7196474ff3b8c360bd6d4888f2417eed)  

**ABSTRACT**  
Social connections are known to be helpful for modeling users' potential preferences and improving the performance of recommender systems. However, in social-aware recommendations, there are two issues which influence the inference of users' preferences, and haven't been well-studied in most existing methods: First, the preferences of a user may only partially match that of his friends in certain aspects, especially when considering a user with diverse interests. Second, for an individual, the influence strength of his friends might be different, as not all friends are equally helpful for modeling his preferences in the system. To address the above issues, in this paper, we propose a novel Social Attentional Memory Network (SAMN) for social-aware recommendation. Specifically, we first design an attention-based memory module to learn user-friend relation vectors, which can capture the varying aspect attentions that a user share with his different friends. Then we build a friend-level attention component to adaptively select informative friends for user modeling. The two components are fused together to mutually enhance each other and lead to a finer extended model. Experimental results on three publicly available datasets show that the proposed SAMN model consistently and significantly outperforms the state-of-the-art recommendation methods. Furthermore, qualitative studies have been made to explore what the proposed attention-based memory module and friend-level attention have learnt, which provide insights into the model's learning process.

{{< /ci-details >}}

{{< ci-details summary="Multi-Layered Network Embedding (Jundong Li et al., 2018)">}}

Jundong Li, C. Chen, Hanghang Tong, Huan Liu. (2018)  
**Multi-Layered Network Embedding**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/7c28b81dff1899e5a148ff57888faacc9945ab22)  
Influential Citation Count (5), SS-ID (7c28b81dff1899e5a148ff57888faacc9945ab22)  

**ABSTRACT**  
Network embedding has gained more attentions in recent years. It has been shown that the learned lowdimensional node vector representations could advance a myriad of graph mining tasks such as node classification, community detection, and link prediction. A vast majority of the existing efforts are overwhelmingly devoted to single-layered networks or homogeneous networks with a single type of nodes and node interactions. However, in many real-world applications, a variety of networks could be abstracted and presented in a multilayered fashion. Typical multi-layered networks include critical infrastructure systems, collaboration platforms, social recommender systems, to name a few. Despite the widespread use of multi-layered networks, it remains a daunting task to learn vector representations of different types of nodes due to the bewildering combination of both within-layer connections and cross-layer network dependencies. In this paper, we study a novel problem of multi-layered network embedding. In particular, we propose a principled framework MANE to model both within-layer connections and cross-layer network dependencies simultaneously in a unified optimization framework for embedding representation learning. Experiments on real-world multi-layered networks corroborate the effectiveness of the proposed framework.

{{< /ci-details >}}

{{< ci-details summary="Learning Convolutional Neural Networks for Graphs (Mathias Niepert et al., 2016)">}}

Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov. (2016)  
**Learning Convolutional Neural Networks for Graphs**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/7c6de5a9e02a779e24504619050c6118f4eac181)  
Influential Citation Count (136), SS-ID (7c6de5a9e02a779e24504619050c6118f4eac181)  

**ABSTRACT**  
Numerous important problems can be framed as learning from graph data. We propose a framework for learning convolutional neural networks for arbitrary graphs. These graphs may be undirected, directed, and with both discrete and continuous node and edge attributes. Analogous to image-based convolutional networks that operate on locally connected regions of the input, we present a general approach to extracting locally connected regions from graphs. Using established benchmark data sets, we demonstrate that the learned feature representations are competitive with state of the art graph kernels and that their computation is highly efficient.

{{< /ci-details >}}

{{< ci-details summary="Robust Multi-Network Clustering via Joint Cross-Domain Cluster Alignment (R. Liu et al., 2015)">}}

R. Liu, Wei Cheng, Hanghang Tong, Wei Wang, X. Zhang. (2015)  
**Robust Multi-Network Clustering via Joint Cross-Domain Cluster Alignment**  
2015 IEEE International Conference on Data Mining  
[Paper Link](https://www.semanticscholar.org/paper/7dd2ad1f992808d04356e8d6e7e5614166f34a85)  
Influential Citation Count (1), SS-ID (7dd2ad1f992808d04356e8d6e7e5614166f34a85)  

**ABSTRACT**  
Network clustering is an important problem thathas recently drawn a lot of attentions. Most existing workfocuses on clustering nodes within a single network. In manyapplications, however, there exist multiple related networks, inwhich each network may be constructed from a different domainand instances in one domain may be related to instances in otherdomains. In this paper, we propose a robust algorithm, MCA, formulti-network clustering that takes into account cross-domain relationshipsbetween instances. MCA has several advantages overthe existing single network clustering methods. First, it is ableto detect associations between clusters from different domains, which, however, is not addressed by any existing methods. Second, it achieves more consistent clustering results on multiple networksby leveraging the duality between clustering individual networksand inferring cross-network cluster alignment. Finally, it providesa multi-network clustering solution that is more robust to noiseand errors. We perform extensive experiments on a variety ofreal and synthetic networks to demonstrate the effectiveness andefficiency of MCA.

{{< /ci-details >}}

{{< ci-details summary="GraphAE: Adaptive Embedding across Graphs (Bencheng Yan et al., 2020)">}}

Bencheng Yan, Chaokun Wang. (2020)  
**GraphAE: Adaptive Embedding across Graphs**  
2020 IEEE 36th International Conference on Data Engineering (ICDE)  
[Paper Link](https://www.semanticscholar.org/paper/7ddb8bdbab7f5aa644569c71d09d0a669af3615e)  
Influential Citation Count (0), SS-ID (7ddb8bdbab7f5aa644569c71d09d0a669af3615e)  

**ABSTRACT**  
Recently, learning embedding of nodes in graphs has attracted increasing research attention. There are two main kinds of graph embedding methods, i.e., the transductive embedding methods and the inductive embedding methods. The former focuses on directly optimizing the embedding vectors, and the latter tries to learn a mapping function for the given nodes and features. However, few works focus on applying the learned model from one graph to another, which is a pervasive idea in Computer Version or Natural Language Processing. Although some of the graph neural networks (GNNs) present similar motivation, none of them considers the graph bias among graphs. In this paper, we present an interesting graph embedding problem called Adaptive Task (AT), and propose a unified framework for this adaptive task, which introduces two types of alignment to learn adaptive node embedding across graphs. Then, based on the proposed framework, a novel graph adaptive embedding network is designed to address the adaptive task. Extensive experimental results demonstrate that our model significantly outperforms the state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Weisfeiler-Lehman Graph Kernels (N. Shervashidze et al., 2011)">}}

N. Shervashidze, Pascal Schweitzer, E. J. V. Leeuwen, K. Mehlhorn, K. Borgwardt. (2011)  
**Weisfeiler-Lehman Graph Kernels**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/7e1874986cf6433fabf96fff93ef42b60bdc49f8)  
Influential Citation Count (183), SS-ID (7e1874986cf6433fabf96fff93ef42b60bdc49f8)  

**ABSTRACT**  
In this article, we propose a family of efficient kernels for large graphs with discrete node labels. Key to our method is a rapid feature extraction scheme based on the Weisfeiler-Lehman test of isomorphism on graphs. It maps the original graph to a sequence of graphs, whose node attributes capture topological and label information. A family of kernels can be defined based on this Weisfeiler-Lehman sequence of graphs, including a highly efficient kernel comparing subtree-like patterns. Its runtime scales only linearly in the number of edges of the graphs and the length of the Weisfeiler-Lehman graph sequence. In our experimental evaluation, our kernels outperform state-of-the-art graph kernels on several graph classification benchmark data sets in terms of accuracy and runtime. Our kernels open the door to large-scale applications of graph kernels in various disciplines such as computational biology and social network analysis.

{{< /ci-details >}}

{{< ci-details summary="On the Embeddability of Random Walk Distances (Xiaohan Zhao et al., 2013)">}}

Xiaohan Zhao, Adelbert Chang, Atish Das Sarma, Haitao Zheng, Ben Y. Zhao. (2013)  
**On the Embeddability of Random Walk Distances**  
Proc. VLDB Endow.  
[Paper Link](https://www.semanticscholar.org/paper/7f2f0d84da4194f048827c9aafd24e2efab8f09f)  
Influential Citation Count (2), SS-ID (7f2f0d84da4194f048827c9aafd24e2efab8f09f)  

**ABSTRACT**  
Analysis of large graphs is critical to the ongoing growth of search engines and social networks. One class of queries centers around node affinity, often quantified by random-walk distances between node pairs, including hitting time, commute time, and personalized PageRank (PPR). Despite the potential of these "metrics," they are rarely, if ever, used in practice, largely due to extremely high computational costs.    In this paper, we investigate methods to scalably and efficiently compute random-walk distances, by "embedding" graphs and distances into points and distances in geometric coordinate spaces. We show that while existing graph coordinate systems (GCS) can accurately estimate shortest path distances, they produce significant errors when embedding random-walk distances. Based on our observations, we propose a new graph embedding system that explicitly accounts for per-node graph properties that affect random walk. Extensive experiments on a range of graphs show that our new approach can accurately estimate both symmetric and asymmetric random-walk distances. Once a graph is embedded, our system can answer queries between any two nodes in 8 microseconds, orders of magnitude faster than existing methods. Finally, we show that our system produces estimates that can replace ground truth in applications with minimal impact on application output.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Attack on Graph Structured Data (H. Dai et al., 2018)">}}

H. Dai, Hui Li, Tian Tian, Xin Huang, L. Wang, Jun Zhu, Le Song. (2018)  
**Adversarial Attack on Graph Structured Data**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/7f77058976e2fe75e98280371962c43d98c98321)  
Influential Citation Count (75), SS-ID (7f77058976e2fe75e98280371962c43d98c98321)  

**ABSTRACT**  
Deep learning on graph structures has shown exciting results in various applications. However, few attentions have been paid to the robustness of such models, in contrast to numerous research work for image or text adversarial attack and defense. In this paper, we focus on the adversarial attacks that fool the model by modifying the combinatorial structure of data. We first propose a reinforcement learning based attack method that learns the generalizable attack policy, while only requiring prediction labels from the target classifier. Also, variants of genetic algorithms and gradient methods are presented in the scenario where prediction confidence or gradients are available. We use both synthetic and real-world data to show that, a family of Graph Neural Network models are vulnerable to these attacks, in both graph-level and node-level classification tasks. We also show such attacks can be used to diagnose the learned classifiers.

{{< /ci-details >}}

{{< ci-details summary="Actional-Structural Graph Convolutional Networks for Skeleton-Based Action Recognition (Maosen Li et al., 2019)">}}

Maosen Li, Siheng Chen, Xu Chen, Ya Zhang, Yanfeng Wang, Qi Tian. (2019)  
**Actional-Structural Graph Convolutional Networks for Skeleton-Based Action Recognition**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/814b70cd133f97ef039bcc44124d9344dd8b3f64)  
Influential Citation Count (40), SS-ID (814b70cd133f97ef039bcc44124d9344dd8b3f64)  

**ABSTRACT**  
Action recognition with skeleton data has recently attracted much attention in computer vision. Previous studies are mostly based on fixed skeleton graphs, only capturing local physical dependencies among joints, which may miss implicit joint correlations. To capture richer dependencies, we introduce an encoder-decoder structure, called A-link inference module, to capture action-specific latent dependencies, i.e. actional links, directly from actions. We also extend the existing skeleton graphs to represent higher-order dependencies, i.e. structural links. Combing the two types of links into a generalized skeleton graph, We further propose the actional-structural graph convolution network (AS-GCN), which stacks actional-structural graph convolution and temporal convolution as a basic building block, to learn both spatial and temporal features for action recognition. A future pose prediction head is added in parallel to the recognition head to help capture more detailed action patterns through self-supervision. We validate AS-GCN in action recognition using two skeleton data sets, NTU-RGB+D and Kinetics. The proposed AS-GCN achieves consistently large improvement compared to the state-of-the-art methods. As a side product, AS-GCN also shows promising results for future pose prediction.

{{< /ci-details >}}

{{< ci-details summary="A Comprehensive Survey on Graph Neural Networks (Zonghan Wu et al., 2019)">}}

Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu. (2019)  
**A Comprehensive Survey on Graph Neural Networks**  
IEEE Transactions on Neural Networks and Learning Systems  
[Paper Link](https://www.semanticscholar.org/paper/81a4fd3004df0eb05d6c1cef96ad33d5407820df)  
Influential Citation Count (197), SS-ID (81a4fd3004df0eb05d6c1cef96ad33d5407820df)  

**ABSTRACT**  
Deep learning has revolutionized many machine learning tasks in recent years, ranging from image classification and video processing to speech recognition and natural language understanding. The data in these tasks are typically represented in the Euclidean space. However, there is an increasing number of applications, where data are generated from non-Euclidean domains and are represented as graphs with complex relationships and interdependency between objects. The complexity of graph data has imposed significant challenges on the existing machine learning algorithms. Recently, many studies on extending deep learning approaches for graph data have emerged. In this article, we provide a comprehensive overview of graph neural networks (GNNs) in data mining and machine learning fields. We propose a new taxonomy to divide the state-of-the-art GNNs into four categories, namely, recurrent GNNs, convolutional GNNs, graph autoencoders, and spatial–temporal GNNs. We further discuss the applications of GNNs across various domains and summarize the open-source codes, benchmark data sets, and model evaluation of GNNs. Finally, we propose potential research directions in this rapidly growing field.

{{< /ci-details >}}

{{< ci-details summary="Deep Reinforcement Learning with Graph-based State Representations (Vikram Waradpande et al., 2020)">}}

Vikram Waradpande, D. Kudenko, Megha Khosla. (2020)  
**Deep Reinforcement Learning with Graph-based State Representations**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/82b794ae070245a7264a13c0a2006685670bc551)  
Influential Citation Count (0), SS-ID (82b794ae070245a7264a13c0a2006685670bc551)  

**ABSTRACT**  
Deep RL approaches build much of their success on the ability of the deep neural network to generate useful internal representations. Nevertheless, they suffer from a high sample-complexity and starting with a good input representation can have a significant impact on the performance. In this paper, we exploit the fact that the underlying Markov decision process (MDP) represents a graph, which enables us to incorporate the topological information for effective state representation learning.  Motivated by the recent success of node representations for several graph analytical tasks we specifically investigate the capability of node representation learning methods to effectively encode the topology of the underlying MDP in Deep RL. To this end we perform a comparative analysis of several models chosen from 4 different classes of representation learning algorithms for policy learning in grid-world navigation tasks, which are representative of a large class of RL problems. We find that all embedding methods outperform the commonly used matrix representation of grid-world environments in all of the studied cases. Moreoever, graph convolution based methods are outperformed by simpler random walk based methods and graph linear autoencoders.

{{< /ci-details >}}

{{< ci-details summary="HEAM: Heterogeneous Network Embedding with Automatic Meta-path Construction (Ruicong Shi et al., 2020)">}}

Ruicong Shi, Tao Liang, Huailiang Peng, Lei Jiang, Qiong Dai. (2020)  
**HEAM: Heterogeneous Network Embedding with Automatic Meta-path Construction**  
KSEM  
[Paper Link](https://www.semanticscholar.org/paper/82eb1da1515172ab7d814320d9666b14b17e2467)  
Influential Citation Count (0), SS-ID (82eb1da1515172ab7d814320d9666b14b17e2467)  

**ABSTRACT**  
Heterogeneous information network (HIN) embedding is widely used in many real-world applications. Meta-path used in HINs can effectively extract semantic information among objects. However, the meta-path faces challenges on the construction and selection. Most of the current works construct dataset-specific meta-paths manually, which rely on the prior knowledge from domain experts. In addition, existing approaches select a few explicit meta-paths, which lack of much subtle semantic information among objects. To tackle the problems, we propose a model with automatic meta-path construction. We develop a hierarchical aggregation to learn effective heterogeneous embeddings with meta-path based proximity. We employ a multi-layer network framework to mine long meta-paths based information implicitly. To demonstrate the effectiveness of our model, we apply it to two real-world datasets and show the performance improvements over state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="The maximum clique problem (P. Pardalos et al., 1994)">}}

P. Pardalos, J. Xue. (1994)  
**The maximum clique problem**  
J. Glob. Optim.  
[Paper Link](https://www.semanticscholar.org/paper/8306882ebcb8066cfbfe984bfe804c3f78d40559)  
Influential Citation Count (38), SS-ID (8306882ebcb8066cfbfe984bfe804c3f78d40559)  

**ABSTRACT**  
In this paper we present a survey of results concerning algorithms, complexity, and applications of the maximum clique problem. We discuss enumerative and exact algorithms, heuristics, and a variety of other proposed methods. An up to date bibliography on the maximum clique and related problems is also provided.

{{< /ci-details >}}

{{< ci-details summary="Network embedding-based representation learning for single cell RNA-seq data (Xiangyu Li et al., 2017)">}}

Xiangyu Li, Weizheng Chen, Yang Chen, Xuegong Zhang, Jin Gu, Michael Q. Zhang. (2017)  
**Network embedding-based representation learning for single cell RNA-seq data**  
Nucleic acids research  
[Paper Link](https://www.semanticscholar.org/paper/83416dea7fd1d3b91ec854d8c4b634926fea77e1)  
Influential Citation Count (2), SS-ID (83416dea7fd1d3b91ec854d8c4b634926fea77e1)  

**ABSTRACT**  
Abstract Single cell RNA-seq (scRNA-seq) techniques can reveal valuable insights of cell-to-cell heterogeneities. Projection of high-dimensional data into a low-dimensional subspace is a powerful strategy in general for mining such big data. However, scRNA-seq suffers from higher noise and lower coverage than traditional bulk RNA-seq, hence bringing in new computational difficulties. One major challenge is how to deal with the frequent drop-out events. The events, usually caused by the stochastic burst effect in gene transcription and the technical failure of RNA transcript capture, often render traditional dimension reduction methods work inefficiently. To overcome this problem, we have developed a novel Single Cell Representation Learning (SCRL) method based on network embedding. This method can efficiently implement data-driven non-linear projection and incorporate prior biological knowledge (such as pathway information) to learn more meaningful low-dimensional representations for both cells and genes. Benchmark results show that SCRL outperforms other dimensional reduction methods on several recent scRNA-seq datasets.

{{< /ci-details >}}

{{< ci-details summary="Simple and Effective Graph Autoencoders with One-Hop Linear Models (Guillaume Salha-Galvan et al., 2020)">}}

Guillaume Salha-Galvan, Romain Hennequin, M. Vazirgiannis. (2020)  
**Simple and Effective Graph Autoencoders with One-Hop Linear Models**  
ECML/PKDD  
[Paper Link](https://www.semanticscholar.org/paper/8380ce56e614d047cf2c5c6106dcfc00beed5b6f)  
Influential Citation Count (2), SS-ID (8380ce56e614d047cf2c5c6106dcfc00beed5b6f)  

**ABSTRACT**  
Over the last few years, graph autoencoders (AE) and variational autoencoders (VAE) emerged as powerful node embedding methods, with promising performances on challenging tasks such as link prediction and node clustering. Graph AE, VAE and most of their extensions rely on multi-layer graph convolutional networks (GCN) encoders to learn vector space representations of nodes. In this paper, we show that GCN encoders are actually unnecessarily complex for many applications. We propose to replace them by significantly simpler and more interpretable linear models w.r.t. the direct neighborhood (one-hop) adjacency matrix of the graph, involving fewer operations, fewer parameters and no activation function. For the two aforementioned tasks, we show that this simpler approach consistently reaches competitive performances w.r.t. GCN-based graph AE and VAE for numerous real-world graphs, including all benchmark datasets commonly used to evaluate graph AE and VAE. Based on these results, we also question the relevance of repeatedly using these datasets to compare complex graph AE and VAE.

{{< /ci-details >}}

{{< ci-details summary="Embedding Entities and Relations for Learning and Inference in Knowledge Bases (Bishan Yang et al., 2014)">}}

Bishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, L. Deng. (2014)  
**Embedding Entities and Relations for Learning and Inference in Knowledge Bases**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/86412306b777ee35aba71d4795b02915cb8a04c3)  
Influential Citation Count (417), SS-ID (86412306b777ee35aba71d4795b02915cb8a04c3)  

**ABSTRACT**  
Abstract: We consider learning representations of entities and relations in KBs using the neural-embedding approach. We show that most existing models, including NTN (Socher et al., 2013) and TransE (Bordes et al., 2013b), can be generalized under a unified learning framework, where entities are low-dimensional vectors learned from a neural network and relations are bilinear and/or linear mapping functions. Under this framework, we compare a variety of embedding models on the link prediction task. We show that a simple bilinear formulation achieves new state-of-the-art results for the task (achieving a top-10 accuracy of 73.2% vs. 54.7% by TransE on Freebase). Furthermore, we introduce a novel approach that utilizes the learned relation embeddings to mine logical rules such as "BornInCity(a,b) and CityInCountry(b,c) => Nationality(a,c)". We find that embeddings learned from the bilinear objective are particularly good at capturing relational semantics and that the composition of relations is characterized by matrix multiplication. More interestingly, we demonstrate that our embedding-based rule extraction approach successfully outperforms a state-of-the-art confidence-based rule mining approach in mining Horn rules that involve compositional reasoning.

{{< /ci-details >}}

{{< ci-details summary="Clustering and Community Detection in Directed Networks: A Survey (Fragkiskos D. Malliaros et al., 2013)">}}

Fragkiskos D. Malliaros, M. Vazirgiannis. (2013)  
**Clustering and Community Detection in Directed Networks: A Survey**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/86be7f7c5888013068ccac9095ed7da6282216b7)  
Influential Citation Count (23), SS-ID (86be7f7c5888013068ccac9095ed7da6282216b7)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Distributed Representations of Words and Phrases and their Compositionality (Tomas Mikolov et al., 2013)">}}

Tomas Mikolov, Ilya Sutskever, Kai Chen, G. Corrado, J. Dean. (2013)  
**Distributed Representations of Words and Phrases and their Compositionality**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/87f40e6f3022adbc1f1905e3e506abad05a9964f)  
Influential Citation Count (3587), SS-ID (87f40e6f3022adbc1f1905e3e506abad05a9964f)  

**ABSTRACT**  
The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.    An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible.

{{< /ci-details >}}

{{< ci-details summary="From Node Embedding To Community Embedding (V. Zheng et al., 2016)">}}

V. Zheng, Sandro Cavallari, Hongyun Cai, K. Chang, E. Cambria. (2016)  
**From Node Embedding To Community Embedding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/88dabd8d295ba9f727baccd73c214e094c6d134f)  
Influential Citation Count (0), SS-ID (88dabd8d295ba9f727baccd73c214e094c6d134f)  

**ABSTRACT**  
Most of the existing graph embedding methods focus on nodes, which aim to output a vector representation for each node in the graph such that two nodes being "close" on the graph are close too in the low-dimensional space. Despite the success of embedding individual nodes for graph analytics, we notice that an important concept of embedding communities (i.e., groups of nodes) is missing. Embedding communities is useful, not only for supporting various community-level applications, but also to help preserve community structure in graph embedding. In fact, we see community embedding as providing a higher-order proximity to define the node closeness, whereas most of the popular graph embedding methods focus on first-order and/or second-order proximities. To learn the community embedding, we hinge upon the insight that community embedding and node embedding reinforce with each other. As a result, we propose ComEmbed, the first community embedding method, which jointly optimizes the community embedding and node embedding together. We evaluate ComEmbed on real-world data sets. We show it outperforms the state-of-the-art baselines in both tasks of node classification and community prediction.

{{< /ci-details >}}

{{< ci-details summary="Dual network embedding for representing research interests in the link prediction problem on co-authorship networks (Ilya Makarov et al., 2019)">}}

Ilya Makarov, Olga Gerasimova, Pavel Sulimov, L. Zhukov. (2019)  
**Dual network embedding for representing research interests in the link prediction problem on co-authorship networks**  
PeerJ Comput. Sci.  
[Paper Link](https://www.semanticscholar.org/paper/894e35751609e1cd88265b055348ced09c8c9acd)  
Influential Citation Count (0), SS-ID (894e35751609e1cd88265b055348ced09c8c9acd)  

**ABSTRACT**  
We present a study on co-authorship network representation based on network embedding together with additional information on topic modeling of research papers and new edge embedding operator. We use the link prediction (LP) model for constructing a recommender system for searching collaborators with similar research interests. Extracting topics for each paper, we construct keywords co-occurrence network and use its embedding for further generalizing author attributes. Standard graph feature engineering and network embedding methods were combined for constructing co-author recommender system formulated as LP problem and prediction of future graph structure. We evaluate our survey on the dataset containing temporal information on National Research University Higher School of Economics over 25 years of research articles indexed in Russian Science Citation Index and Scopus. Our model of network representation shows better performance for stated binary classification tasks on several co-authorship networks.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Large Graph Embedding Based on Balanced and Hierarchical K-Means (F. Nie et al., 2020)">}}

F. Nie, Wei Zhu, Xuelong Li. (2020)  
**Unsupervised Large Graph Embedding Based on Balanced and Hierarchical K-Means**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/894e66e457482d8b658dfc1d1f4d6f532357b400)  
Influential Citation Count (0), SS-ID (894e66e457482d8b658dfc1d1f4d6f532357b400)  

**ABSTRACT**  
There are many successful spectral based unsupervised dimensionality reduction methods, including Laplacian Eigenmap (LE), Locality Preserving Projection (LPP), Spectral Regression (SR), etc. We find that LPP and SR are equivalent if the symmetric similarity matrix is doubly stochastic, Positive Semi-Definite (PSD) and with rank <inline-formula><tex-math notation="LaTeX">$p$</tex-math><alternatives><mml:math><mml:mi>p</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq1-3000226.gif"/></alternatives></inline-formula>, where <inline-formula><tex-math notation="LaTeX">$p$</tex-math><alternatives><mml:math><mml:mi>p</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq2-3000226.gif"/></alternatives></inline-formula> is the reduced dimension. Since solving SR is believed faster than solving LPP based on some related literature, the discovery promotes us to seek to construct such specific similarity matrix to speed up LPP solving procedures. We then propose an unsupervised linear method called Unsupervised Large Graph Embedding (ULGE). ULGE starts with a similar idea as LPP but adopts an efficient approach to construct anchor-based similarity matrix and then performs spectral analysis on it. Moreover, since conventional anchor generation strategies suffer kinds of problems, we propose an efficient and effective anchor generation strategy, called Balanced <inline-formula><tex-math notation="LaTeX">$K$</tex-math><alternatives><mml:math><mml:mi>K</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq3-3000226.gif"/></alternatives></inline-formula>-means based Hierarchical <inline-formula><tex-math notation="LaTeX">$K$</tex-math><alternatives><mml:math><mml:mi>K</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq4-3000226.gif"/></alternatives></inline-formula>-means (BHKH). The computational complexity of ULGE can reduce to <inline-formula><tex-math notation="LaTeX">$O(ndm)$</tex-math><alternatives><mml:math><mml:mrow><mml:mi>O</mml:mi><mml:mo>(</mml:mo><mml:mi>n</mml:mi><mml:mi>d</mml:mi><mml:mi>m</mml:mi><mml:mo>)</mml:mo></mml:mrow></mml:math><inline-graphic xlink:href="zhu-ieq5-3000226.gif"/></alternatives></inline-formula>, which is a significant improvement compared to conventional methods need <inline-formula><tex-math notation="LaTeX">$O(n^2d)$</tex-math><alternatives><mml:math><mml:mrow><mml:mi>O</mml:mi><mml:mo>(</mml:mo><mml:msup><mml:mi>n</mml:mi><mml:mn>2</mml:mn></mml:msup><mml:mi>d</mml:mi><mml:mo>)</mml:mo></mml:mrow></mml:math><inline-graphic xlink:href="zhu-ieq6-3000226.gif"/></alternatives></inline-formula> at least, where <inline-formula><tex-math notation="LaTeX">$n$</tex-math><alternatives><mml:math><mml:mi>n</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq7-3000226.gif"/></alternatives></inline-formula>, <inline-formula><tex-math notation="LaTeX">$d$</tex-math><alternatives><mml:math><mml:mi>d</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq8-3000226.gif"/></alternatives></inline-formula> and <inline-formula><tex-math notation="LaTeX">$m$</tex-math><alternatives><mml:math><mml:mi>m</mml:mi></mml:math><inline-graphic xlink:href="zhu-ieq9-3000226.gif"/></alternatives></inline-formula> are the number of samples, dimensions, and anchors, respectively. Extensive experiments on several publicly available datasets demonstrate the efficiency and effectiveness of the proposed method.

{{< /ci-details >}}

{{< ci-details summary="Graph HyperNetworks for Neural Architecture Search (Chris Zhang et al., 2018)">}}

Chris Zhang, Mengye Ren, R. Urtasun. (2018)  
**Graph HyperNetworks for Neural Architecture Search**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/89c10e08902cb90abbe1276a3042b93c2f9c78b4)  
Influential Citation Count (17), SS-ID (89c10e08902cb90abbe1276a3042b93c2f9c78b4)  

**ABSTRACT**  
Neural architecture search (NAS) automatically finds the best task-specific neural network topology, outperforming many manual architecture designs. However, it can be prohibitively expensive as the search requires training thousands of different networks, while each can last for hours. In this work, we propose the Graph HyperNetwork (GHN) to amortize the search cost: given an architecture, it directly generates the weights by running inference on a graph neural network. GHNs model the topology of an architecture and therefore can predict network performance more accurately than regular hypernetworks and premature early stopping. To perform NAS, we randomly sample architectures and use the validation accuracy of networks with GHN generated weights as the surrogate search signal. GHNs are fast -- they can search nearly 10 times faster than other random search methods on CIFAR-10 and ImageNet. GHNs can be further extended to the anytime prediction setting, where they have found networks with better speed-accuracy tradeoff than the state-of-the-art manual designs.

{{< /ci-details >}}

{{< ci-details summary="Temporal link prediction by integrating content and structure information (Sheng Gao et al., 2011)">}}

Sheng Gao, Ludovic Denoyer, P. Gallinari. (2011)  
**Temporal link prediction by integrating content and structure information**  
CIKM '11  
[Paper Link](https://www.semanticscholar.org/paper/8a634a82681897822b14de28849c6548346206a0)  
Influential Citation Count (4), SS-ID (8a634a82681897822b14de28849c6548346206a0)  

**ABSTRACT**  
In this paper we address the problem of temporal link prediction, i.e., predicting the apparition of new links, in time-evolving networks. This problem appears in applications such as recommender systems, social network analysis or citation analysis. Link prediction in time-evolving networks is usually based on the topological structure of the network only. We propose here a model which exploits multiple information sources in the network in order to predict link occurrence probabilities as a function of time. The model integrates three types of information: the global network structure, the content of nodes in the network if any, and the local or proximity information of a given vertex. The proposed model is based on a matrix factorization formulation of the problem with graph regularization. We derive an efficient optimization method to learn the latent factors of this model. Extensive experiments on several real world datasets suggest that our unified framework outperforms state-of-the-art methods for temporal link prediction tasks.

{{< /ci-details >}}

{{< ci-details summary="Singular value decomposition and least squares solutions (G. Golub et al., 1970)">}}

G. Golub, C. Reinsch. (1970)  
**Singular value decomposition and least squares solutions**  
Milestones in Matrix Computation  
[Paper Link](https://www.semanticscholar.org/paper/8ae0cbae42a5fb9b340adaed9ed39569eb96b42d)  
Influential Citation Count (202), SS-ID (8ae0cbae42a5fb9b340adaed9ed39569eb96b42d)  

**ABSTRACT**  
Let A be a real m×n matrix with m≧n. It is well known (cf. [4]) that    $$A = U\sum {V^T}$$    (1)    where    $${U^T}U = {V^T}V = V{V^T} = {I_n}{\text{ and }}\sum {\text{ = diag(}}{\sigma _{\text{1}}}{\text{,}} \ldots {\text{,}}{\sigma _n}{\text{)}}{\text{.}}$$    The matrix U consists of n orthonormalized eigenvectors associated with the n largest eigenvalues of AA T , and the matrix V consists of the orthonormalized eigenvectors of A T A. The diagonal elements of ∑ are the non-negative square roots of the eigenvalues of A T A; they are called singular values. We shall assume that    $${\sigma _1} \geqq {\sigma _2} \geqq \cdots \geqq {\sigma _n} \geqq 0.$$    Thus if rank(A)=r, σ r+1 = σ r+2=⋯=σ n = 0. The decomposition (1) is called the singular value decomposition (SVD).

{{< /ci-details >}}

{{< ci-details summary="Tri-Party Deep Network Representation (Shirui Pan et al., 2016)">}}

Shirui Pan, Jia Wu, Xingquan Zhu, Chengqi Zhang, Yang Wang. (2016)  
**Tri-Party Deep Network Representation**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/8ba7631515d5e7e0c451af1c4772507f41540a5e)  
Influential Citation Count (32), SS-ID (8ba7631515d5e7e0c451af1c4772507f41540a5e)  

**ABSTRACT**  
Information network mining often requires examination of linkage relationships between nodes for analysis. Recently, network representation has emerged to represent each node in a vector format, embedding network structure, so off-the-shelf machine learning methods can be directly applied for analysis. To date, existing methods only focus on one aspect of node information and cannot leverage node labels. In this paper, we propose TriDNR, a tri-party deep network representation model, using information from three parties: node structure, node content, and node labels (if available) to jointly learn optimal node representation. TriDNR is based on our new coupled deep natural language module, whose learning is enforced at three levels: (1) at the network structure level, TriDNR exploits inter-node relationship by maximizing the probability of observing surrounding nodes given a node in random walks; (2) at the node content level, TriDNR captures node-word correlation by maximizing the co-occurrence of word sequence given a node; and (3) at the node label level, TriDNR models label-word correspondence by maximizing the probability of word sequence given a class label. The tri-party information is jointly fed into the neural network model to mutually enhance each other to learn optimal representation, and results in up to 79% classification accuracy gain, compared to state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Neighbourhood Watch: Referring Expression Comprehension via Language-Guided Graph Attention Networks (Peng Wang et al., 2018)">}}

Peng Wang, Qi Wu, Jiewei Cao, Chunhua Shen, Lianli Gao, A. V. Hengel. (2018)  
**Neighbourhood Watch: Referring Expression Comprehension via Language-Guided Graph Attention Networks**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/8ca91ad7763be4da05238aa17a9e5628f619dc0b)  
Influential Citation Count (12), SS-ID (8ca91ad7763be4da05238aa17a9e5628f619dc0b)  

**ABSTRACT**  
The task in referring expression comprehension is to localize the object instance in an image described by a referring expression phrased in natural language. As a language-to-vision matching task, the key to this problem is to learn a discriminative object feature that can adapt to the expression used. To avoid ambiguity, the expression normally tends to describe not only the properties of the referent itself, but also its relationships to its neighbourhood. To capture and exploit this important information we propose a graph-based, language-guided attention mechanism. Being composed of node attention component and edge attention component, the proposed graph attention mechanism explicitly represents inter-object relationships, and properties with a flexibility and power impossible with competing approaches. Furthermore, the proposed graph attention mechanism enables the comprehension decision to be visualizable and explainable. Experiments on three referring expression comprehension datasets show the advantage of the proposed approach.

{{< /ci-details >}}

{{< ci-details summary="Link Prediction in Complex Networks: A Survey (Linyuan Lu et al., 2010)">}}

Linyuan Lu, T. Zhou. (2010)  
**Link Prediction in Complex Networks: A Survey**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/8cd9aa720a3a2f9dcb52ad9eb1bf258a80ce0648)  
Influential Citation Count (191), SS-ID (8cd9aa720a3a2f9dcb52ad9eb1bf258a80ce0648)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Design of multi-view graph embedding using multiple kernel learning (Asif Salim et al., 2020)">}}

Asif Salim, S. Shiju, S. Sumitra. (2020)  
**Design of multi-view graph embedding using multiple kernel learning**  
Eng. Appl. Artif. Intell.  
[Paper Link](https://www.semanticscholar.org/paper/8ce7c67095d76da23897cc379c063986e9843a7c)  
Influential Citation Count (0), SS-ID (8ce7c67095d76da23897cc379c063986e9843a7c)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Friends and neighbors on the Web (Lada A. Adamic et al., 2003)">}}

Lada A. Adamic, Eytan Adar. (2003)  
**Friends and neighbors on the Web**  
Soc. Networks  
[Paper Link](https://www.semanticscholar.org/paper/8dc9d11e3fc229a1b70bb00de72dc15d55848174)  
Influential Citation Count (180), SS-ID (8dc9d11e3fc229a1b70bb00de72dc15d55848174)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Not All Links Are Created Equal: An Adaptive Embedding Approach for Social Personalized Ranking (Qing Zhang et al., 2016)">}}

Qing Zhang, Houfeng Wang. (2016)  
**Not All Links Are Created Equal: An Adaptive Embedding Approach for Social Personalized Ranking**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/8e1d196f5e7d4b8c24a5970a9052fa57d7f805a7)  
Influential Citation Count (2), SS-ID (8e1d196f5e7d4b8c24a5970a9052fa57d7f805a7)  

**ABSTRACT**  
With a large amount of complex network data available, most existing recommendation models consider exploiting rich user social relations for better interest targeting. In these approaches, the underlying assumption is that similar users in social networks would prefer similar items. However, in practical scenarios, social link may not be formed by common interest. For example, one general collected social network might be used for various specific recommendation scenarios. The problem of noisy social relations without interest relevance will arise to hurt the performance. Moreover, the sparsity problem of social network makes it much more challenging, due to the two-fold problem needed to be solved simultaneously, for effectively incorporating social information to benefit recommendation. To address this challenge, we propose an adaptive embedding approach to solve the both jointly for better recommendation in real world setting. Experiments conducted on real world datasets show that our approach outperforms current methods.

{{< /ci-details >}}

{{< ci-details summary="An Overview of Microsoft Academic Service (MAS) and Applications (Arnab Sinha et al., 2015)">}}

Arnab Sinha, Zhihong Shen, Yang Song, Hao Ma, Darrin Eide, B. Hsu, Kuansan Wang. (2015)  
**An Overview of Microsoft Academic Service (MAS) and Applications**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/8ebc4145aef6a575cbaffcfeec56b20586db573a)  
Influential Citation Count (119), SS-ID (8ebc4145aef6a575cbaffcfeec56b20586db573a)  

**ABSTRACT**  
In this paper we describe a new release of a Web scale entity graph that serves as the backbone of Microsoft Academic Service (MAS), a major production effort with a broadened scope to the namesake vertical search engine that has been publicly available since 2008 as a research prototype. At the core of MAS is a heterogeneous entity graph comprised of six types of entities that model the scholarly activities: field of study, author, institution, paper, venue, and event. In addition to obtaining these entities from the publisher feeds as in the previous effort, we in this version include data mining results from the Web index and an in-house knowledge base from Bing, a major commercial search engine. As a result of the Bing integration, the new MAS graph sees significant increase in size, with fresh information streaming in automatically following their discoveries by the search engine. In addition, the rich entity relations included in the knowledge base provide additional signals to disambiguate and enrich the entities within and beyond the academic domain. The number of papers indexed by MAS, for instance, has grown from low tens of millions to 83 million while maintaining an above 95% accuracy based on test data sets derived from academic activities at Microsoft Research. Based on the data set, we demonstrate two scenarios in this work: a knowledge driven, highly interactive dialog that seamlessly combines reactive search and proactive suggestion experience, and a proactive heterogeneous entity recommendation.

{{< /ci-details >}}

{{< ci-details summary="Predicting Drug–Target Interactions Using Probabilistic Matrix Factorization (M. Çobanoğlu et al., 2013)">}}

M. Çobanoğlu, Chang Liu, F. Hu, Z. Oltvai, I. Bahar. (2013)  
**Predicting Drug–Target Interactions Using Probabilistic Matrix Factorization**  
J. Chem. Inf. Model.  
[Paper Link](https://www.semanticscholar.org/paper/8ebdcddbb9c8b3140c3d88cb479ac8468109ce21)  
Influential Citation Count (8), SS-ID (8ebdcddbb9c8b3140c3d88cb479ac8468109ce21)  

**ABSTRACT**  
Quantitative analysis of known drug–target interactions emerged in recent years as a useful approach for drug repurposing and assessing side effects. In the present study, we present a method that uses probabilistic matrix factorization (PMF) for this purpose, which is particularly useful for analyzing large interaction networks. DrugBank drugs clustered based on PMF latent variables show phenotypic similarity even in the absence of 3D shape similarity. Benchmarking computations show that the method outperforms those recently introduced provided that the input data set of known interactions is sufficiently large—which is the case for enzymes and ion channels, but not for G-protein coupled receptors (GPCRs) and nuclear receptors. Runs performed on DrugBank after hiding 70% of known interactions show that, on average, 88 of the top 100 predictions hit the hidden interactions. De novo predictions permit us to identify new potential interactions. Drug–target pairs implicated in neurobiological disorders are overrepresented among de novo predictions.

{{< /ci-details >}}

{{< ci-details summary="Deep mining heterogeneous networks of biomedical linked data to predict novel drug‐target associations (Nansu Zong et al., 2017)">}}

Nansu Zong, Hyeon-eui Kim, Victoria Ngo, O. Harismendy. (2017)  
**Deep mining heterogeneous networks of biomedical linked data to predict novel drug‐target associations**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/8ef131f8e043a47aebe50dae449ed42843a9749d)  
Influential Citation Count (6), SS-ID (8ef131f8e043a47aebe50dae449ed42843a9749d)  

**ABSTRACT**  
Motivation: A heterogeneous network topology possessing abundant interactions between biomedical entities has yet to be utilized in similarity‐based methods for predicting drug‐target associations based on the array of varying features of drugs and their targets. Deep learning reveals features of vertices of a large network that can be adapted in accommodating the similarity‐based solutions to provide a flexible method of drug‐target prediction. Results: We propose a similarity‐based drug‐target prediction method that enhances existing association discovery methods by using a topology‐based similarity measure. DeepWalk, a deep learning method, is adopted in this study to calculate the similarities within Linked Tripartite Network (LTN), a heterogeneous network generated from biomedical linked datasets. This proposed method shows promising results for drug‐target association prediction: 98.96% AUC ROC score with a 10‐fold cross‐validation and 99.25% AUC ROC score with a Monte Carlo cross‐validation with LTN. By utilizing DeepWalk, we demonstrate that: (i) this method outperforms other existing topology‐based similarity computation methods, (ii) the performance is better for tripartite than with bipartite networks and (iii) the measure of similarity using network topology outperforms the ones derived from chemical structure (drugs) or genomic sequence (targets). Our proposed methodology proves to be capable of providing a promising solution for drug‐target prediction based on topological similarity with a heterogeneous network, and may be readily re‐purposed and adapted in the existing of similarity‐based methodologies. Availability and Implementation: The proposed method has been developed in JAVA and it is available, along with the data at the following URL: https://github.com/zongnansu1982/drug‐target‐prediction. Contact: nazong@ucsd.edu Supplementary information: Supplementary data are available at Bioinformatics online.

{{< /ci-details >}}

{{< ci-details summary="Session-Based Social Recommendation via Dynamic Graph Attention Networks (Weiping Song et al., 2019)">}}

Weiping Song, Zhiping Xiao, Yifan Wang, Laurent Charlin, Ming Zhang, Jian Tang. (2019)  
**Session-Based Social Recommendation via Dynamic Graph Attention Networks**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/901a6ad54f3bfc0ca6671f4e492703c671475288)  
Influential Citation Count (13), SS-ID (901a6ad54f3bfc0ca6671f4e492703c671475288)  

**ABSTRACT**  
Online communities such as Facebook and Twitter are enormously popular and have become an essential part of the daily life of many of their users. Through these platforms, users can discover and create information that others will then consume. In that context, recommending relevant information to users becomes critical for viability. However, recommendation in online communities is a challenging problem: 1) users' interests are dynamic, and 2) users are influenced by their friends. Moreover, the influencers may be context-dependent. That is, different friends may be relied upon for different topics. Modeling both signals is therefore essential for recommendations. We propose a recommender system for online communities based on a dynamic-graph-attention neural network. We model dynamic user behaviors with a recurrent neural network, and context-dependent social influence with a graph-attention neural network, which dynamically infers the influencers based on users' current interests. The whole model can be efficiently fit on large-scale data. Experimental results on several real-world data sets demonstrate the effectiveness of our proposed approach over several competitive baselines including state-of-the-art models.

{{< /ci-details >}}

{{< ci-details summary="Learning Graph Embedding With Adversarial Training Methods (Shirui Pan et al., 2019)">}}

Shirui Pan, Ruiqi Hu, S. Fung, Guodong Long, Jing Jiang, Chengqi Zhang. (2019)  
**Learning Graph Embedding With Adversarial Training Methods**  
IEEE Transactions on Cybernetics  
[Paper Link](https://www.semanticscholar.org/paper/914fbae74420475d54c8099c5921b5f799c1c6c7)  
Influential Citation Count (11), SS-ID (914fbae74420475d54c8099c5921b5f799c1c6c7)  

**ABSTRACT**  
Graph embedding aims to transfer a graph into vectors to facilitate subsequent graph-analytics tasks like link prediction and graph clustering. Most approaches on graph embedding focus on preserving the graph structure or minimizing the reconstruction errors for graph data. They have mostly overlooked the embedding distribution of the latent codes, which unfortunately may lead to inferior representation in many cases. In this article, we present a novel adversarially regularized framework for graph embedding. By employing the graph convolutional network as an encoder, our framework embeds the topological information and node content into a vector representation, from which a graph decoder is further built to reconstruct the input graph. The adversarial training principle is applied to enforce our latent codes to match a prior Gaussian or uniform distribution. Based on this framework, we derive two variants of the adversarial models, the adversarially regularized graph autoencoder (ARGA) and its variational version, and adversarially regularized variational graph autoencoder (ARVGA), to learn the graph embedding effectively. We also exploit other potential variations of ARGA and ARVGA to get a deeper understanding of our designs. Experimental results that compared 12 algorithms for link prediction and 20 algorithms for graph clustering validate our solutions.

{{< /ci-details >}}

{{< ci-details summary="Link-Based Classification (, 2014)">}}

. (2014)  
**Link-Based Classification**  
Encyclopedia of Social Network Analysis and Mining  
[Paper Link](https://www.semanticscholar.org/paper/91bf323aec3270df0ec8082389119d536c395655)  
Influential Citation Count (48), SS-ID (91bf323aec3270df0ec8082389119d536c395655)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Predicting who rated what in large-scale datasets (Yan Liu et al., 2007)">}}

Yan Liu, Zhenzhen Kou. (2007)  
**Predicting who rated what in large-scale datasets**  
SKDD  
[Paper Link](https://www.semanticscholar.org/paper/92b5c2a83183c7d2c7c6e2bea6994730e42f975e)  
Influential Citation Count (1), SS-ID (92b5c2a83183c7d2c7c6e2bea6994730e42f975e)  

**ABSTRACT**  
KDD Cup 2007 focuses on movie rating behaviors. The goal of the task "Who Rated What" is to predict whether "existing" users will review "existing" movies in the future. We cast the task as a link prediction problem and address it via a simple classification approach. Compared with other applications for link prediction, there are two major challenges in our task: (1) the huge size of the Netflix data; (2) the prediction target is complicated by many factors, such as a general decrease of interest in old movies and more tendency to review more movies by Netflix users due to the success of the internet DVD rental industries. We address the first challenge by "selective" subsampling and the second by combining information from the review scores, movie contents and graph topology effectively.

{{< /ci-details >}}

{{< ci-details summary="Collaborative matrix factorization with multiple similarities for predicting drug-target interactions (Xiaodong Zheng et al., 2013)">}}

Xiaodong Zheng, Hao Ding, Hiroshi Mamitsuka, Shanfeng Zhu. (2013)  
**Collaborative matrix factorization with multiple similarities for predicting drug-target interactions**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/92e8f21bcf8437b6e6b7e43d4c2e9bd717e41673)  
Influential Citation Count (30), SS-ID (92e8f21bcf8437b6e6b7e43d4c2e9bd717e41673)  

**ABSTRACT**  
We address the problem of predicting new drug-target interactions from three inputs: known interactions, similarities over drugs and those over targets. This setting has been considered by many methods, which however have a common problem of allowing to have only one similarity matrix over drugs and that over targets. The key idea of our approach is to use more than one similarity matrices over drugs as well as those over targets, where weights over the multiple similarity matrices are estimated from data to automatically select similarities, which are effective for improving the performance of predicting drug-target interactions. We propose a factor model, named Multiple Similarities Collaborative Matrix Factorization(MSCMF), which projects drugs and targets into a common low-rank feature space, which is further consistent with weighted similarity matrices over drugs and those over targets. These two low-rank matrices and weights over similarity matrices are estimated by an alternating least squares algorithm. Our approach allows to predict drug-target interactions by the two low-rank matrices collaboratively and to detect similarities which are important for predicting drug-target interactions. This approach is general and applicable to any binary relations with similarities over elements, being found in many applications, such as recommender systems. In fact, MSCMF is an extension of weighted low-rank approximation for one-class collaborative filtering. We extensively evaluated the performance of MSCMF by using both synthetic and real datasets. Experimental results showed nice properties of MSCMF on selecting similarities useful in improving the predictive performance and the performance advantage of MSCMF over six state-of-the-art methods for predicting drug-target interactions.

{{< /ci-details >}}

{{< ci-details summary="Large-Scale Multi-View Spectral Clustering via Bipartite Graph (Yeqing Li et al., 2015)">}}

Yeqing Li, F. Nie, Heng Huang, Junzhou Huang. (2015)  
**Large-Scale Multi-View Spectral Clustering via Bipartite Graph**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/9383f08c697b8aa43782e16c9a57e089911584d8)  
Influential Citation Count (33), SS-ID (9383f08c697b8aa43782e16c9a57e089911584d8)  

**ABSTRACT**  
In this paper, we address the problem of large-scale multi-view spectral clustering. In many real-world applications, data can be represented in various heterogeneous features or views. Different views often provide different aspects of information that are complementary to each other. Several previous methods of clustering have demonstrated that better accuracy can be achieved using integrated information of all the views than just using each view individually. One important class of such methods is multi-view spectral clustering, which is based on graph Laplacian. However, existing methods are not applicable to large-scale problem for their high computational complexity. To this end, we propose a novel large-scale multi-view spectral clustering approach based on the bipartite graph. Our method uses local manifold fusion to integrate heterogeneous features. To improve efficiency, we approximate the similarity graphs using bipartite graphs. Furthermore, we show that our method can be easily extended to handle the out-of-sample problem. Extensive experimental results on five benchmark datasets demonstrate the effectiveness and efficiency of the proposed method, where our method runs up to nearly 3000 times faster than the state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Multi-view clustering: A survey (Yan Yang et al., 2018)">}}

Yan Yang, Hao Wang. (2018)  
**Multi-view clustering: A survey**  
Big Data Min. Anal.  
[Paper Link](https://www.semanticscholar.org/paper/9404cc4488ae98babf17b286a20e7baa6ef5d398)  
Influential Citation Count (3), SS-ID (9404cc4488ae98babf17b286a20e7baa6ef5d398)  

**ABSTRACT**  
In the big data era, the data are generated from different sources or observed from different views. These data are referred to as multi-view data. Unleashing the power of knowledge in multi-view data is very important in big data mining and analysis. This calls for advanced techniques that consider the diversity of different views, while fusing these data. Multi-view Clustering (MvC) has attracted increasing attention in recent years by aiming to exploit complementary and consensus information across multiple views. This paper summarizes a large number of multi-view clustering algorithms, provides a taxonomy according to the mechanisms and principles involved, and classifies these algorithms into five categories, namely, co-training style algorithms, multi-kernel learning, multiview graph clustering, multi-view subspace clustering, and multi-task multi-view clustering. Therein, multi-view graph clustering is further categorized as graph-based, network-based, and spectral-based methods. Multi-view subspace clustering is further divided into subspace learning-based, and non-negative matrix factorization-based methods. This paper does not only introduce the mechanisms for each category of methods, but also gives a few examples for how these techniques are used. In addition, it lists some publically available multi-view datasets. Overall, this paper serves as an introductory text and survey for multi-view clustering.

{{< /ci-details >}}

{{< ci-details summary="DINIES: drug–target interaction network inference engine based on supervised analysis (Yoshihiro Yamanishi et al., 2014)">}}

Yoshihiro Yamanishi, Masaaki Kotera, Yuki Moriya, Ryusuke Sawada, M. Kanehisa, S. Goto. (2014)  
**DINIES: drug–target interaction network inference engine based on supervised analysis**  
Nucleic Acids Res.  
[Paper Link](https://www.semanticscholar.org/paper/952719f09792f25e66079cc0edd370ffb166caf8)  
Influential Citation Count (2), SS-ID (952719f09792f25e66079cc0edd370ffb166caf8)  

**ABSTRACT**  
DINIES (drug–target interaction network inference engine based on supervised analysis) is a web server for predicting unknown drug–target interaction networks from various types of biological data (e.g. chemical structures, drug side effects, amino acid sequences and protein domains) in the framework of supervised network inference. The originality of DINIES lies in prediction with state-of-the-art machine learning methods, in the integration of heterogeneous biological data and in compatibility with the KEGG database. The DINIES server accepts any ‘profiles’ or precalculated similarity matrices (or ‘kernels’) of drugs and target proteins in tab-delimited file format. When a training data set is submitted to learn a predictive model, users can select either known interaction information in the KEGG DRUG database or their own interaction data. The user can also select an algorithm for supervised network inference, select various parameters in the method and specify weights for heterogeneous data integration. The server can provide integrative analyses with useful components in KEGG, such as biological pathways, functional hierarchy and human diseases. DINIES (http://www.genome.jp/tools/dinies/) is publicly available as one of the genome analysis tools in GenomeNet.

{{< /ci-details >}}

{{< ci-details summary="Distributed large-scale natural graph factorization (Amr Ahmed et al., 2013)">}}

Amr Ahmed, N. Shervashidze, Shravan M. Narayanamurthy, V. Josifovski, Alex Smola. (2013)  
**Distributed large-scale natural graph factorization**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/952bc3bc999be86d4b03a9c4af94c555c822aa11)  
Influential Citation Count (30), SS-ID (952bc3bc999be86d4b03a9c4af94c555c822aa11)  

**ABSTRACT**  
Natural graphs, such as social networks, email graphs, or instant messaging patterns, have become pervasive through the internet. These graphs are massive, often containing hundreds of millions of nodes and billions of edges. While some theoretical models have been proposed to study such graphs, their analysis is still difficult due to the scale and nature of the data. We propose a framework for large-scale graph decomposition and inference. To resolve the scale, our framework is distributed so that the data are partitioned over a shared-nothing set of machines. We propose a novel factorization technique that relies on partitioning a graph so as to minimize the number of neighboring vertices rather than edges across partitions. Our decomposition is based on a streaming algorithm. It is network-aware as it adapts to the network topology of the underlying computational hardware. We use local copies of the variables and an efficient asynchronous communication protocol to synchronize the replicated values in order to perform most of the computation without having to incur the cost of network communication. On a graph of 200 million vertices and 10 billion edges, derived from an email communication network, our algorithm retains convergence properties while allowing for almost linear scalability in the number of computers.

{{< /ci-details >}}

{{< ci-details summary="Learning by Sampling and Compressing: Efficient Graph Representation Learning with Extremely Limited Annotations (Xiaoming Liu et al., 2020)">}}

Xiaoming Liu, Qirui Li, Chao Shen, Xi Peng, Yadong Zhou, Guan Xiaohong. (2020)  
**Learning by Sampling and Compressing: Efficient Graph Representation Learning with Extremely Limited Annotations**  
  
[Paper Link](https://www.semanticscholar.org/paper/95555b454fc93ff6cab8ad8ee9bb3096d32e2b9c)  
Influential Citation Count (0), SS-ID (95555b454fc93ff6cab8ad8ee9bb3096d32e2b9c)  

**ABSTRACT**  
Graph convolution network (GCN) attracts intensive research interest with broad applications. While existing work mainly focused on designing novel GCN architectures for better performance, few of them studied a practical yet challenging problem: How to learn GCNs from data with extremely limited annotation? In this paper, we propose a new learning method by sampling strategy and model compression to overcome this challenge. Our approach has multifold advantages: 1) the adaptive sampling strategy largely suppresses the GCN training deviation over uniform sampling; 2) compressed GCN-based methods with a smaller scale of parameters need fewer labeled data to train; 3) the smaller scale of training data is beneficial to reduce the human resource cost to label them. We choose six popular GCN baselines and conduct extensive experiments on three real-world datasets. The results show that by applying our method, all GCN baselines cut down the annotation requirement by as much as 90$\%$ and compress the scale of parameters more than 6$\times$ without sacrificing their strong performance. It verifies that the training method could extend the existing semi-supervised GCN-based methods to the scenarios with the extremely small scale of labeled data.

{{< /ci-details >}}

{{< ci-details summary="Holographic Embeddings of Knowledge Graphs (Maximilian Nickel et al., 2015)">}}

Maximilian Nickel, L. Rosasco, T. Poggio. (2015)  
**Holographic Embeddings of Knowledge Graphs**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/955fe2ee26d888ae22749b0853981b8b581b133d)  
Influential Citation Count (139), SS-ID (955fe2ee26d888ae22749b0853981b8b581b133d)  

**ABSTRACT**  
Learning embeddings of entities and relations is an efficient and versatile method to perform machine learning on relational data such as knowledge graphs. In this work, we propose holographic embeddings (HOLE) to learn compositional vector space representations of entire knowledge graphs. The proposed method is related to holographic models of associative memory in that it employs circular correlation to create compositional representations. By using correlation as the compositional operator, HOLE can capture rich interactions but simultaneously remains efficient to compute, easy to train, and scalable to very large datasets. Experimentally, we show that holographic embeddings are able to outperform state-of-the-art methods for link prediction on knowledge graphs and relational learning benchmark datasets.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised and Scalable Algorithm for Learning Node Representations (Tiago Pimentel et al., 2017)">}}

Tiago Pimentel, Adriano Veloso, N. Ziviani. (2017)  
**Unsupervised and Scalable Algorithm for Learning Node Representations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/95a32bda5a743da698e062a5f5806ce5f22aef29)  
Influential Citation Count (2), SS-ID (95a32bda5a743da698e062a5f5806ce5f22aef29)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Dual-Primal Graph Convolutional Networks (Federico Monti et al., 2018)">}}

Federico Monti, Oleksandr Shchur, Aleksandar Bojchevski, O. Litany, Stephan Günnemann, M. Bronstein. (2018)  
**Dual-Primal Graph Convolutional Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/980a4959ad4c81a61f4166b549157ddad1f7ddce)  
Influential Citation Count (1), SS-ID (980a4959ad4c81a61f4166b549157ddad1f7ddce)  

**ABSTRACT**  
In recent years, there has been a surge of interest in developing deep learning methods for non-Euclidean structured data such as graphs. In this paper, we propose Dual-Primal Graph CNN, a graph convolutional architecture that alternates convolution-like operations on the graph and its dual. Our approach allows to learn both vertex- and edge features and generalizes the previous graph attention (GAT) model. We provide extensive experimental validation showing state-of-the-art results on a variety of tasks tested on established graph benchmarks, including CORA and Citeseer citation networks as well as MovieLens, Flixter, Douban and Yahoo Music graph-guided recommender systems.

{{< /ci-details >}}

{{< ci-details summary="Collective Pairwise Classification for Multi-Way Analysis of Disease and Drug Data (M. Zitnik et al., 2016)">}}

M. Zitnik, B. Zupan. (2016)  
**Collective Pairwise Classification for Multi-Way Analysis of Disease and Drug Data**  
PSB  
[Paper Link](https://www.semanticscholar.org/paper/982d4c371d4ea37f9438eb58d788c1d183300f99)  
Influential Citation Count (1), SS-ID (982d4c371d4ea37f9438eb58d788c1d183300f99)  

**ABSTRACT**  
Interactions between drugs, drug targets or diseases can be predicted on the basis of molecular, clinical and genomic features by, for example, exploiting similarity of disease pathways, chemical structures, activities across cell lines or clinical manifestations of diseases. A successful way to better understand complex interactions in biomedical systems is to employ collective relational learning approaches that can jointly model diverse relationships present in multiplex data. We propose a novel collective pairwise classification approach for multi-way data analysis. Our model leverages the superiority of latent factor models and classifies relationships in a large relational data domain using a pairwise ranking loss. In contrast to current approaches, our method estimates probabilities, such that probabilities for existing relationships are higher than for assumed-to-be-negative relationships. Although our method bears correspondence with the maximization of non-differentiable area under the ROC curve, we were able to design a learning algorithm that scales well on multi-relational data encoding interactions between thousands of entities.We use the new method to infer relationships from multiplex drug data and to predict connections between clinical manifestations of diseases and their underlying molecular signatures. Our method achieves promising predictive performance when compared to state-of-the-art alternative approaches and can make "category-jumping" predictions about diseases from genomic and clinical data generated far outside the molecular context.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Attention-Based Variational Graph Autoencoder (Ziqiang Weng et al., 2020)">}}

Ziqiang Weng, Weiyu Zhang, Wei Dou. (2020)  
**Adversarial Attention-Based Variational Graph Autoencoder**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/989afdf64a0df329004de086cbc9057a9d8634f4)  
Influential Citation Count (0), SS-ID (989afdf64a0df329004de086cbc9057a9d8634f4)  

**ABSTRACT**  
Autoencoders have been successfully used for graph embedding, and many variants have been proven to effectively express graph data and conduct graph analysis in low-dimensional space. However, previous methods ignore the structure and properties of the reconstructed graph, or they do not consider the potential data distribution in the graph, which typically leads to unsatisfactory graph embedding performance. In this paper, we propose the adversarial attention variational graph autoencoder (AAVGA), which is a novel framework that incorporates attention networks into the encoder part and uses an adversarial mechanism in embedded training. The encoder involves node neighbors in the representation of nodes by stacking attention layers, which can further improve the graph embedding performance of the encoder. At the same time, due to the adversarial mechanism, the distribution of the potential features that are generated by the encoder are closer to the actual distribution of the original graph data; thus, the decoder generates a graph that is closer to the original graph. Experimental results prove that AAVGA performs competitively with state-of-the-art popular graph encoders on three citation datasets.

{{< /ci-details >}}

{{< ci-details summary="SIGN: Scalable Inception Graph Neural Networks (Emanuele Rossi et al., 2020)">}}

Emanuele Rossi, F. Frasca, B. Chamberlain, D. Eynard, M. Bronstein, Federico Monti. (2020)  
**SIGN: Scalable Inception Graph Neural Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/993377a3fc8334558463b82053904e3d684f29c0)  
Influential Citation Count (23), SS-ID (993377a3fc8334558463b82053904e3d684f29c0)  

**ABSTRACT**  
Graph representation learning has recently been applied to a broad spectrum of problems ranging from computer graphics and chemistry to high energy physics and social media. The popularity of graph neural networks has sparked interest, both in academia and in industry, in developing methods that scale to very large graphs such as Facebook or Twitter social networks. In most of these approaches, the computational cost is alleviated by a sampling strategy retaining a subset of node neighbors or subgraphs at training time. In this paper we propose a new, efficient and scalable graph deep learning architecture which sidesteps the need for graph sampling by using graph convolutional filters of different size that are amenable to efficient precomputation, allowing extremely fast training and inference. Our architecture allows using different local graph operators (e.g. motif-induced adjacency matrices or Personalized Page Rank diffusion matrix) to best suit the task at hand. We conduct extensive experimental evaluation on various open benchmarks and show that our approach is competitive with other state-of-the-art architectures, while requiring a fraction of the training and inference time.

{{< /ci-details >}}

{{< ci-details summary="Learning Entity and Relation Embeddings for Knowledge Graph Completion (Yankai Lin et al., 2015)">}}

Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. (2015)  
**Learning Entity and Relation Embeddings for Knowledge Graph Completion**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/994afdf0db0cb0456f4f76468380822c2f532726)  
Influential Citation Count (363), SS-ID (994afdf0db0cb0456f4f76468380822c2f532726)  

**ABSTRACT**  
Knowledge graph completion aims to perform link prediction between entities. In this paper, we consider the approach of knowledge graph embeddings. Recently, models such as TransE and TransH build entity and relation embeddings by regarding a relation as translation from head entity to tail entity. We note that these models simply put both entities and relations within the same semantic space. In fact, an entity may have multiple aspects and various relations may focus on different aspects of entities, which makes a common space insufficient for modeling. In this paper, we propose TransR to build entity and relation embeddings in separate entity space and relation spaces. Afterwards, we learn embeddings by first projecting entities from entity space to corresponding relation space and then building translations between projected entities. In experiments, we evaluate our models on three tasks including link prediction, triple classification and relational fact extraction. Experimental results show significant and consistent improvements compared to state-of-the-art baselines including TransE and TransH. The source code of this paper can be obtained from https://github.com/mrlyk423/relation_extraction.

{{< /ci-details >}}

{{< ci-details summary="The link-prediction problem for social networks (D. Liben-Nowell et al., 2007)">}}

D. Liben-Nowell, J. Kleinberg. (2007)  
**The link-prediction problem for social networks**  
J. Assoc. Inf. Sci. Technol.  
[Paper Link](https://www.semanticscholar.org/paper/996dfa43f6982bcbff862276ef80cbca7515985a)  
Influential Citation Count (240), SS-ID (996dfa43f6982bcbff862276ef80cbca7515985a)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Continuous nonlinear dimensionality reduction by kernel Eigenmaps (M. Brand, 2003)">}}

M. Brand. (2003)  
**Continuous nonlinear dimensionality reduction by kernel Eigenmaps**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/99cd988b104202887ad9657b8a61baa7ff0581c1)  
Influential Citation Count (5), SS-ID (99cd988b104202887ad9657b8a61baa7ff0581c1)  

**ABSTRACT**  
We equate nonlinear dimensionality reduction (NLDR) to graph embedding with side information about the vertices, and derive a solution to either problem in the form of a kernel-based mixture of affine maps from the ambient space to the target space. Unlike most spectral NLDR methods, the central eigenproblem can be made relatively small, and the result is a continuous mapping defined over the entire space, not just the datapoints. A demonstration is made to visualizing the distribution of word usages (as a proxy to word meanings) in a sample of the machine learning literature.

{{< /ci-details >}}

{{< ci-details summary="HeteGCN: Heterogeneous Graph Convolutional Networks for Text Classification (Rahul Ragesh et al., 2020)">}}

Rahul Ragesh, Sundararajan Sellamanickam, Arun Iyer, Ramakrishna Bairi, Vijay Lingam. (2020)  
**HeteGCN: Heterogeneous Graph Convolutional Networks for Text Classification**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/9a6935328336b05fb95a47916eccf7b3c50b2f97)  
Influential Citation Count (3), SS-ID (9a6935328336b05fb95a47916eccf7b3c50b2f97)  

**ABSTRACT**  
We consider the problem of learning efficient and inductive graph convolutional networks for text classification with a large number of examples and features. Existing state-of-the-art graph embedding based methods such as predictive text embedding (PTE) and TextGCN have shortcomings in terms of predictive performance, scalability and inductive capability. To address these limitations, we propose a heterogeneous graph convolutional network (HeteGCN) modeling approach that unites the best aspects of PTE and TextGCN together. The main idea is to learn feature embeddings and derive document embeddings using a HeteGCN architecture with different graphs used across layers. We simplify TextGCN by dissecting into several HeteGCN models which (a) helps to study the usefulness of individual models and (b) offers flexibility in fusing learned embeddings from different models. In effect, the number of model parameters is reduced significantly, enabling faster training and improving performance in small labeled training set scenario. Our detailed experimental studies demonstrate the efficacy of the proposed approach.

{{< /ci-details >}}

{{< ci-details summary="A Spectral Clustering Approach To Finding Communities in Graph (Scott White et al., 2005)">}}

Scott White, Padhraic Smyth. (2005)  
**A Spectral Clustering Approach To Finding Communities in Graph**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/9a92300b0ecc33f8e55a8ac945a51aaade549013)  
Influential Citation Count (22), SS-ID (9a92300b0ecc33f8e55a8ac945a51aaade549013)  

**ABSTRACT**  
Clustering nodes in a graph is a useful general technique in data mining of large network data sets. In this context, Newman and Girvan [9] recently proposed an objective function for graph clustering called the Q function which allows automatic selection of the number of clusters. Empirically, higher values of the Q function have been shown to correlate well with good graph clusterings. In this paper we show how optimizing the Q function can be reformulated as a spectral relaxation problem and propose two new spectral clustering algorithms that seek to maximize Q. Experimental results indicate that the new algorithms are efficient and effective at finding both good clusterings and the appropriate number of clusters across a variety of real-world graph data sets. In addition, the spectral algorithms are much faster for large sparse graphs, scaling roughly linearly with the number of nodes n in the graph, compared to O(n) for previous clustering algorithms using the Q function.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Large Graph Embedding (F. Nie et al., 2017)">}}

F. Nie, Wei Zhu, Xuelong Li. (2017)  
**Unsupervised Large Graph Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/9ad503ff70a2b3a1ddebc96683ed73c7fcd0840b)  
Influential Citation Count (5), SS-ID (9ad503ff70a2b3a1ddebc96683ed73c7fcd0840b)  

**ABSTRACT**  
There are many successful spectral based unsupervised dimensionality reduction methods, including Laplacian Eigenmap (LE), Locality Preserving Projection (LPP), Spectral Regression (SR), etc. LPP and SR are two different linear spectral based methods, however, we discover that LPP and SR are equivalent, if the symmetric similarity matrix is doubly stochastic, Positive Semi-Definite (PSD) and with rank p, where p is the reduced dimension. The discovery promotes us to seek low-rank and doubly stochastic similarity matrix, we then propose an unsupervised linear dimensionality reduction method, called Unsupervised Large Graph Embedding (ULGE). ULGE starts with similar idea as LPP, it adopts an efficient approach to construct similarity matrix and then performs spectral analysis efficiently, the computational complexity can reduce to O(ndm), which is a significant improvement compared to conventional spectral based methods which need O(nd) at least, where n, d and m are the number of samples, dimensions and anchors, respectively. Extensive experiments on several public available data sets demonstrate the efficiency and effectiveness of the proposed method.

{{< /ci-details >}}

{{< ci-details summary="Recommendation as link prediction: a graph kernel-based machine learning approach (Xin Li et al., 2009)">}}

Xin Li, Hsinchun Chen. (2009)  
**Recommendation as link prediction: a graph kernel-based machine learning approach**  
JCDL '09  
[Paper Link](https://www.semanticscholar.org/paper/9b194ee4c71eb526078627bf7da9b9275b0e421b)  
Influential Citation Count (4), SS-ID (9b194ee4c71eb526078627bf7da9b9275b0e421b)  

**ABSTRACT**  
Recommender systems have demonstrated commercial success in multiple industries. In digital libraries they have the potential to be used as a support tool for traditional information retrieval functions. Among the major recommendation algorithms, the successful collaborative filtering (CF) methods explore the use of user-item interactions to infer user interests. Based on the finding that transitive user-item associations can alleviate the data sparsity problem in CF, multiple heuristic algorithms were designed to take advantage of the user-item interaction networks with both direct and indirect interactions. However, the use of such graph representation was still limited in learning-based algorithms. In this paper, we propose a graph kernel-based recommendation framework. For each user-item pair, we inspect its associative interaction graph (AIG) that contains the users, items, and interactions n steps away from the pair. We design a novel graph kernel to capture the AIG structures and use them to predict possible user-item interactions. The framework demonstrates improved performance on an online bookstore dataset, especially when a large number of suggestions are needed.

{{< /ci-details >}}

{{< ci-details summary="Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting (Yaguang Li et al., 2017)">}}

Yaguang Li, Rose Yu, C. Shahabi, Yan Liu. (2017)  
**Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/9ba0186ed40656329c421f55ada7313293e13f17)  
Influential Citation Count (211), SS-ID (9ba0186ed40656329c421f55ada7313293e13f17)  

**ABSTRACT**  
Spatiotemporal forecasting has various applications in neuroscience, climate and transportation domain. Traffic forecasting is one canonical example of such learning task. The task is challenging due to (1) complex spatial dependency on road networks, (2) non-linear temporal dynamics with changing road conditions and (3) inherent difficulty of long-term forecasting. To address these challenges, we propose to model the traffic flow as a diffusion process on a directed graph and introduce Diffusion Convolutional Recurrent Neural Network (DCRNN), a deep learning framework for traffic forecasting that incorporates both spatial and temporal dependency in the traffic flow. Specifically, DCRNN captures the spatial dependency using bidirectional random walks on the graph, and the temporal dependency using the encoder-decoder architecture with scheduled sampling. We evaluate the framework on two real-world large scale road network traffic datasets and observe consistent improvement of 12% - 15% over state-of-the-art baselines.

{{< /ci-details >}}

{{< ci-details summary="Anonymized GCN: A Novel Robust Graph Embedding Method via Hiding Node Position in Noise (Ao Liu, 2020)">}}

Ao Liu. (2020)  
**Anonymized GCN: A Novel Robust Graph Embedding Method via Hiding Node Position in Noise**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/9bad9deef8ebd658417d038849f3ca25b94f136b)  
Influential Citation Count (0), SS-ID (9bad9deef8ebd658417d038849f3ca25b94f136b)  

**ABSTRACT**  
Graph convolution network (GCN) have achieved state-of-the-art performance in the task of node prediction in the graph structure. However, with the gradual various of graph attack methods, there are lack of research on the robustness of GCN. At this paper, we will design a robust GCN method for node prediction tasks. Considering the graph structure contains two types of information: node information and connection information, and attackers usually modify the connection information to complete the interference with the prediction results of the node, we first proposed a method to hide the connection information in the generator, named Anonymized GCN (AN-GCN). By hiding the connection information in the graph structure in the generator through adversarial training, the accurate node prediction can be completed only by the node number rather than its specific position in the graph. Specifically, we first demonstrated the key to determine the embedding of a specific node: the row corresponding to the node of the eigenmatrix of the Laplace matrix, by target it as the output of the generator, we designed a method to hide the node number in the noise. Take the corresponding noise as input, we will obtain the connection structure of the node instead of directly obtaining. Then the encoder and decoder are spliced both in discriminator, so that after adversarial training, the generator and discriminator can cooperate to complete the encoding and decoding of the graph, then complete the node prediction. Finally, All node positions can generated by noise at the same time, that is to say, the generator will hides all the connection information of the graph structure. The evaluation shows that we only need to obtain the initial features and node numbers of the nodes to complete the node prediction, and the accuracy did not decrease, but increased by 0.0293.

{{< /ci-details >}}

{{< ci-details summary="Community detection in graphs (S. Fortunato, 2009)">}}

S. Fortunato. (2009)  
**Community detection in graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/9be428c9383d47b86570b1b9fc20faf006346c5d)  
Influential Citation Count (717), SS-ID (9be428c9383d47b86570b1b9fc20faf006346c5d)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="GOSH: Embedding Big Graphs on Small Hardware (Taha Atahan Akyildiz et al., 2020)">}}

Taha Atahan Akyildiz, Amro Alabsi Aljundi, K. Kaya. (2020)  
**GOSH: Embedding Big Graphs on Small Hardware**  
ICPP  
[Paper Link](https://www.semanticscholar.org/paper/9c1d6253cf83028e00a3d120777263ac2882ad29)  
Influential Citation Count (1), SS-ID (9c1d6253cf83028e00a3d120777263ac2882ad29)  

**ABSTRACT**  
In graph embedding, the connectivity information of a graph is used to represent each vertex as a point in a d-dimensional space. Unlike the original, irregular structural information, such a representation can be used for a multitude of machine learning tasks. Although the process is extremely useful in practice, it is indeed expensive and unfortunately, the graphs are becoming larger and harder to embed. Attempts at scaling up the process to larger graphs have been successful but often at a steep price in hardware requirements. We present Gosh, an approach for embedding graphs of arbitrary sizes on a single GPU with minimum constraints. Gosh utilizes a novel graph coarsening approach to compress the graph and minimize the work required for embedding, delivering high-quality embeddings at a fraction of the time compared to the state-of-the-art. In addition to this, it incorporates a decomposition schema that enables any arbitrarily large graph to be embedded using a single GPU with minimum constraints on the memory size. With these techniques, Gosh is able to embed a graph with over 65 million vertices and 1.8 billion edges in less than an hour on a single GPU and obtains a 93% AUCROC for link-prediction which can be increased to 95% by running the tool for 80 minutes.

{{< /ci-details >}}

{{< ci-details summary="Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering (Mikhail Belkin et al., 2001)">}}

Mikhail Belkin, P. Niyogi. (2001)  
**Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/9d16c547d15a08091e68c86a99731b14366e3f0d)  
Influential Citation Count (356), SS-ID (9d16c547d15a08091e68c86a99731b14366e3f0d)  

**ABSTRACT**  
Drawing on the correspondence between the graph Laplacian, the Laplace-Beltrami operator on a manifold, and the connections to the heat equation, we propose a geometrically motivated algorithm for constructing a representation for data sampled from a low dimensional manifold embedded in a higher dimensional space. The algorithm provides a computationally efficient approach to nonlinear dimensionality reduction that has locality preserving properties and a natural connection to clustering. Several applications are considered.

{{< /ci-details >}}

{{< ci-details summary="Relational Topic Models for Document Networks (Jonathan Chang et al., 2009)">}}

Jonathan Chang, D. Blei. (2009)  
**Relational Topic Models for Document Networks**  
AISTATS  
[Paper Link](https://www.semanticscholar.org/paper/9f68d27df3a4c4be8636f376cb15f77e55a2f496)  
Influential Citation Count (82), SS-ID (9f68d27df3a4c4be8636f376cb15f77e55a2f496)  

**ABSTRACT**  
We develop the relational topic model (RTM), a model of documents and the links between them. For each pair of documents, the RTM models their link as a binary random variable that is conditioned on their contents. The model can be used to summarize a network of documents, predict links between them, and predict words within them. We derive efficient inference and learning algorithms based on variational methods and evaluate the predictive performance of the RTM for large networks of scientific abstracts and web documents.

{{< /ci-details >}}

{{< ci-details summary="Predicting Rich Drug-Drug Interactions via Biomedical Knowledge Graphs and Text Jointly Embedding (M. Wang et al., 2017)">}}

M. Wang, Yihe Chen, B. Qian, Jun Liu, Sen Wang, Guodong Long, Fei Wang. (2017)  
**Predicting Rich Drug-Drug Interactions via Biomedical Knowledge Graphs and Text Jointly Embedding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/a0344c4d7720757a24932b2d8a1486ff9fddf5d3)  
Influential Citation Count (0), SS-ID (a0344c4d7720757a24932b2d8a1486ff9fddf5d3)  

**ABSTRACT**  
Minimizing adverse reactions caused by drug-drug interactions has always been a momentous research topic in clinical pharmacology. Detecting all possible interactions through clinical studies before a drug is released to the market is a demanding task. The power of big data is opening up new approaches to discover various drug-drug interactions. However, these discoveries contain a huge amount of noise and provide knowledge bases far from complete and trustworthy ones to be utilized. Most existing studies focus on predicting binary drug-drug interactions between drug pairs but ignore other interactions. In this paper, we propose a novel framework, called PRD, to predict drug-drug interactions. The framework uses the graph embedding that can overcome data incompleteness and sparsity issues to achieve multiple DDI label prediction. First, a large-scale drug knowledge graph is generated from different sources. Then, the knowledge graph is embedded with comprehensive biomedical text into a common low dimensional space. Finally, the learned embeddings are used to efficiently compute rich DDI information through a link prediction process. To validate the effectiveness of the proposed framework, extensive experiments were conducted on real-world datasets. The results demonstrate that our model outperforms several state-of-the-art baseline methods in terms of capability and accuracy.

{{< /ci-details >}}

{{< ci-details summary="Graph Convolutional Label Noise Cleaner: Train a Plug-And-Play Action Classifier for Anomaly Detection (Jia-Xing Zhong et al., 2019)">}}

Jia-Xing Zhong, Nannan Li, Weijie Kong, Shan Liu, Thomas H. Li, Ge Li. (2019)  
**Graph Convolutional Label Noise Cleaner: Train a Plug-And-Play Action Classifier for Anomaly Detection**  
2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/a03bda078490e8ee991a1f86b53f27df7cf93a14)  
Influential Citation Count (22), SS-ID (a03bda078490e8ee991a1f86b53f27df7cf93a14)  

**ABSTRACT**  
Video anomaly detection under weak labels is formulated as a typical multiple-instance learning problem in previous works. In this paper, we provide a new perspective, i.e., a supervised learning task under noisy labels. In such a viewpoint, as long as cleaning away label noise, we can directly apply fully supervised action classifiers to weakly supervised anomaly detection, and take maximum advantage of these well-developed classifiers. For this purpose, we devise a graph convolutional network to correct noisy labels. Based upon feature similarity and temporal consistency, our network propagates supervisory signals from high-confidence snippets to low-confidence ones. In this manner, the network is capable of providing cleaned supervision for action classifiers. During the test phase, we only need to obtain snippet-wise predictions from the action classifier without any extra post-processing. Extensive experiments on 3 datasets at different scales with 2 types of action classifiers demonstrate the efficacy of our method. Remarkably, we obtain the frame-level AUC score of 82.12% on UCF-Crime.

{{< /ci-details >}}

{{< ci-details summary="SIDE: Representation Learning in Signed Directed Networks (Junghwan Kim et al., 2018)">}}

Junghwan Kim, Haekyu Park, Ji-Eun Lee, U. Kang. (2018)  
**SIDE: Representation Learning in Signed Directed Networks**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/a0da1be7b7665b8c23d80ad2b03815dd708cd7b9)  
Influential Citation Count (21), SS-ID (a0da1be7b7665b8c23d80ad2b03815dd708cd7b9)  

**ABSTRACT**  
Given a signed directed network, how can we learn node representations which fully encode structural information of the network including sign and direction of edges? Node representation learning or network embedding learns a mapping of each node to a vector. The mapping encodes structural information on network, providing low-dimensional dense node features for general machine learning and data mining frameworks. Since many social networks allow trust (friend) and distrust (enemy) relationships described by signed and directed edges, generalizing network embedding method to learn from sign and direction information in networks is crucial. In addition, social theories are critical tool in signed network analysis. However, none of the existing methods supports all of the desired properties: considering sign, direction, and social theoretical interpretation. In this paper, we propose SIDE, a general network embedding method that represents both sign and direction of edges in the embedding space. SIDE carefully formulates and optimizes likelihood over both direct and indirect signed connections. We provide socio-psychological interpretation for each component of likelihood function. We prove linear scalability of our algorithm and propose additional optimization techniques to reduce the training time and improve accuracy. Through extensive experiments on real-world signed directed networks, we show that SIDE effectively encodes structural information into the learned embedding.

{{< /ci-details >}}

{{< ci-details summary="A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources (Xiao Wang et al., 2020)">}}

Xiao Wang, Deyu Bo, C. Shi, Shaohua Fan, Yanfang Ye, Philip S. Yu. (2020)  
**A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/a11828bb8b2e5f1644360567f0e46d20de342ad6)  
Influential Citation Count (0), SS-ID (a11828bb8b2e5f1644360567f0e46d20de342ad6)  

**ABSTRACT**  
Heterogeneous graphs (HGs) also known as heterogeneous information networks have become ubiquitous in real-world scenarios; therefore, HG embedding, which aims to learn representations in a lower-dimension space while preserving the heterogeneous structures and semantics for downstream tasks (e.g., node/graph classification, node clustering, link prediction), has drawn considerable attentions in recent years. In this survey, we perform a comprehensive review of the recent development on HG embedding methods and techniques. We first introduce the basic concepts of HG and discuss the unique challenges brought by the heterogeneity for HG embedding in comparison with homogeneous graph representation learning; and then we systemically survey and categorize the state-of-the-art HG embedding methods based on the information they used in the learning process to address the challenges posed by the HG heterogeneity. In particular, for each representative HG embedding method, we provide detailed introduction and further analyze its pros and cons; meanwhile, we also explore the transformativeness and applicability of different types of HG embedding methods in the real-world industrial environments for the first time. In addition, we further present several widely deployed systems that have demonstrated the success of HG embedding techniques in resolving real-world application problems with broader impacts. To facilitate future research and applications in this area, we also summarize the open-source code, existing graph learning platforms and benchmark datasets. Finally, we explore the additional issues and challenges of HG embedding and forecast the future research directions in this field.

{{< /ci-details >}}

{{< ci-details summary="STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems (Jiani Zhang et al., 2019)">}}

Jiani Zhang, Xingjian Shi, Shenglin Zhao, Irwin King. (2019)  
**STAR-GCN: Stacked and Reconstructed Graph Convolutional Networks for Recommender Systems**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/a3c6926a1d90385b746a16cbb9f0ad6fe714dc1c)  
Influential Citation Count (10), SS-ID (a3c6926a1d90385b746a16cbb9f0ad6fe714dc1c)  

**ABSTRACT**  
We propose a new STAcked and Reconstructed Graph Convolutional Networks (STAR-GCN) architecture to learn node representations for boosting the performance in recommender systems, especially in the cold start scenario. STAR-GCN employs a stack of GCN encoder-decoders combined with intermediate supervision to improve the final prediction performance. Unlike the graph convolutional matrix completion model with one-hot encoding node inputs, our STAR-GCN learns low-dimensional user and item latent factors as the input to restrain the model space complexity. Moreover, our STAR-GCN can produce node embeddings for new nodes by reconstructing masked input node embeddings, which essentially tackles the cold start problem. Furthermore, we discover a label leakage issue when training GCN-based models for link prediction tasks and propose a training strategy to avoid the issue. Empirical results on multiple rating prediction benchmarks demonstrate our model achieves state-of-the-art performance in four out of five real-world datasets and significant improvements in predicting ratings in the cold start scenario. The code implementation is available in https://github.com/jennyzhang0215/STAR-GCN.

{{< /ci-details >}}

{{< ci-details summary="Learning Graph-based POI Embedding for Location-based Recommendation (M. Xie et al., 2016)">}}

M. Xie, Hongzhi Yin, Hao Wang, Fanjiang Xu, Weitong Chen, Sen Wang. (2016)  
**Learning Graph-based POI Embedding for Location-based Recommendation**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/a468d2f55cddbf87c3caf1f9ce45b838e24a8ac7)  
Influential Citation Count (18), SS-ID (a468d2f55cddbf87c3caf1f9ce45b838e24a8ac7)  

**ABSTRACT**  
With the rapid prevalence of smart mobile devices and the dramatic proliferation of location-based social networks (LBSNs), location-based recommendation has become an important means to help people discover attractive and interesting points of interest (POIs). However, the extreme sparsity of user-POI matrix and cold-start issue create severe challenges, causing CF-based methods to degrade significantly in their recommendation performance. Moreover, location-based recommendation requires spatiotemporal context awareness and dynamic tracking of the user's latest preferences in a real-time manner. To address these challenges, we stand on recent advances in embedding learning techniques and propose a generic graph-based embedding model, called GE, in this paper. GE jointly captures the sequential effect, geographical influence, temporal cyclic effect and semantic effect in a unified way by embedding the four corresponding relational graphs (POI-POI, POI-Region, POI-Time and POI-Word)into a shared low dimensional space. Then, to support the real-time recommendation, we develop a novel time-decay method to dynamically compute the user's latest preferences based on the embedding of his/her checked-in POIs learnt in the latent space. We conduct extensive experiments to evaluate the performance of our model on two real large-scale datasets, and the experimental results show its superiority over other competitors, especially in recommending cold-start POIs. Besides, we study the contribution of each factor to improve location-based recommendation and find that both sequential effect and temporal cyclic effect play more important roles than geographical influence and semantic effect.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Completion via Complex Tensor Factorization (Théo Trouillon et al., 2017)">}}

Théo Trouillon, C. Dance, Éric Gaussier, Johannes Welbl, S. Riedel, Guillaume Bouchard. (2017)  
**Knowledge Graph Completion via Complex Tensor Factorization**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/a4dfb121275a6408d290b803baf8c9caeb23dc5b)  
Influential Citation Count (29), SS-ID (a4dfb121275a6408d290b803baf8c9caeb23dc5b)  

**ABSTRACT**  
In statistical relational learning, knowledge graph completion deals with automatically understanding the structure of large knowledge graphs---labeled directed graphs---and predicting missing relationships---labeled edges. State-of-the-art embedding models propose different trade-offs between modeling expressiveness, and time and space complexity. We reconcile both expressiveness and complexity through the use of complex-valued embeddings and explore the link between such complex-valued embeddings and unitary diagonalization. We corroborate our approach theoretically and show that all real square matrices---thus all possible relation/adjacency matrices---are the real part of some unitarily diagonalizable matrix. This results opens the door to a lot of other applications of square matrices factorization. Our approach based on complex embeddings is arguably simple, as it only involves a Hermitian dot product, the complex counterpart of the standard dot product between real vectors, whereas other methods resort to more and more complicated composition functions to increase their expressiveness. The proposed complex embeddings are scalable to large data sets as it remains linear in both space and time, while consistently outperforming alternative approaches on standard link prediction benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Relational learning via latent social dimensions (Lei Tang et al., 2009)">}}

Lei Tang, Huan Liu. (2009)  
**Relational learning via latent social dimensions**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/a505e4c2bf30cd88afe483f7541409e2ba5ab3d4)  
Influential Citation Count (72), SS-ID (a505e4c2bf30cd88afe483f7541409e2ba5ab3d4)  

**ABSTRACT**  
Social media such as blogs, Facebook, Flickr, etc., presents data in a network format rather than classical IID distribution. To address the interdependency among data instances, relational learning has been proposed, and collective inference based on network connectivity is adopted for prediction. However, connections in social media are often multi-dimensional. An actor can connect to another actor for different reasons, e.g., alumni, colleagues, living in the same city, sharing similar interests, etc. Collective inference normally does not differentiate these connections. In this work, we propose to extract latent social dimensions based on network information, and then utilize them as features for discriminative learning. These social dimensions describe diverse affiliations of actors hidden in the network, and the discriminative learning can automatically determine which affiliations are better aligned with the class labels. Such a scheme is preferred when multiple diverse relations are associated with the same network. We conduct extensive experiments on social media data (one from a real-world blog site and the other from a popular content sharing site). Our model outperforms representative relational learning methods based on collective inference, especially when few labeled data are available. The sensitivity of this model and its connection to existing methods are also examined.

{{< /ci-details >}}

{{< ci-details summary="Stochastic Training of Graph Convolutional Networks with Variance Reduction (Jianfei Chen et al., 2017)">}}

Jianfei Chen, Jun Zhu, Le Song. (2017)  
**Stochastic Training of Graph Convolutional Networks with Variance Reduction**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/a60c69c2fae27ebbb73c87f7f2a4765556bd7f9f)  
Influential Citation Count (44), SS-ID (a60c69c2fae27ebbb73c87f7f2a4765556bd7f9f)  

**ABSTRACT**  
Graph convolutional networks (GCNs) are powerful deep neural networks for graph-structured data. However, GCN computes the representation of a node recursively from its neighbors, making the receptive field size grow exponentially with the number of layers. Previous attempts on reducing the receptive field size by subsampling neighbors do not have a convergence guarantee, and their receptive field size per node is still in the order of hundreds. In this paper, we develop control variate based algorithms which allow sampling an arbitrarily small neighbor size. Furthermore, we prove new theoretical guarantee for our algorithms to converge to a local optimum of GCN. Empirical results show that our algorithms enjoy a similar convergence with the exact algorithm using only two neighbors per node. The runtime of our algorithms on a large Reddit dataset is only one seventh of previous neighbor sampling algorithms.

{{< /ci-details >}}

{{< ci-details summary="Joint Node-Edge Network Embedding for Link Prediction (Ilya Makarov et al., 2018)">}}

Ilya Makarov, Olga Gerasimova, Pavel Sulimov, Ksenia Korovina, L. Zhukov. (2018)  
**Joint Node-Edge Network Embedding for Link Prediction**  
AIST  
[Paper Link](https://www.semanticscholar.org/paper/a6ef59c8c64adfb82908a31651711d7710b4d93d)  
Influential Citation Count (0), SS-ID (a6ef59c8c64adfb82908a31651711d7710b4d93d)  

**ABSTRACT**  
In this paper, we consider new formulation of graph embedding algorithm, while learning node and edge representation under common constraints. We evaluate our approach on link prediction problem for co-authorship network of HSE researchers’ publications. We compare it with existing structural network embeddings and feature-engineering models.

{{< /ci-details >}}

{{< ci-details summary="Discriminative Deep Random Walk for Network Classification (Juzheng Li et al., 2016)">}}

Juzheng Li, Jun Zhu, Bo Zhang. (2016)  
**Discriminative Deep Random Walk for Network Classification**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/a6fd225417efdbf0bb9aef2ef2046335d2d0885e)  
Influential Citation Count (2), SS-ID (a6fd225417efdbf0bb9aef2ef2046335d2d0885e)  

**ABSTRACT**  
Deep Random Walk (DeepWalk) can learn a latent space representation for describing the topological structure of a network. However, for relational network classification, DeepWalk can be suboptimal as it lacks a mechanism to optimize the objective of the target task. In this paper, we present Discriminative Deep Random Walk (DDRW), a novel method for relational network classification. By solving a joint optimization problem, DDRW can learn the latent space representations that well capture the topological structure and meanwhile are discriminative for the network classification task. Our experimental results on several real social networks demonstrate that DDRW significantly outperforms DeepWalk on multilabel network classification tasks, while retaining the topological structure in the latent space. DDRW is stable and consistently outperforms the baseline methods by various percentages of labeled data. DDRW is also an online method that is scalable and can be naturally parallelized.

{{< /ci-details >}}

{{< ci-details summary="SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels (Matthias Fey et al., 2017)">}}

Matthias Fey, J. E. Lenssen, F. Weichert, H. Müller. (2017)  
**SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels**  
2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition  
[Paper Link](https://www.semanticscholar.org/paper/a73531abe4cafbccd5b3e949e84410a50016bd33)  
Influential Citation Count (39), SS-ID (a73531abe4cafbccd5b3e949e84410a50016bd33)  

**ABSTRACT**  
We present Spline-based Convolutional Neural Networks (SplineCNNs), a variant of deep neural networks for irregular structured and geometric input, e.g., graphs or meshes. Our main contribution is a novel convolution operator based on B-splines, that makes the computation time independent from the kernel size due to the local support property of the B-spline basis functions. As a result, we obtain a generalization of the traditional CNN convolution operator by using continuous kernel functions parametrized by a fixed number of trainable weights. In contrast to related approaches that filter in the spectral domain, the proposed method aggregates features purely in the spatial domain. In addition, SplineCNN allows entire end-to-end training of deep architectures, using only the geometric structure as input, instead of handcrafted feature descriptors. For validation, we apply our method on tasks from the fields of image graph classification, shape correspondence and graph node classification, and show that it outperforms or pars state-of-the-art approaches while being significantly faster and having favorable properties like domain-independence. Our source code is available on GitHub1.

{{< /ci-details >}}

{{< ci-details summary="GL2vec: Graph Embedding Enriched by Line Graphs with Edge Features (Hong Chen et al., 2019)">}}

Hong Chen, H. Koga. (2019)  
**GL2vec: Graph Embedding Enriched by Line Graphs with Edge Features**  
ICONIP  
[Paper Link](https://www.semanticscholar.org/paper/a7df35d17d05d69e7085d1cfa288a235a8a86be1)  
Influential Citation Count (5), SS-ID (a7df35d17d05d69e7085d1cfa288a235a8a86be1)  

**ABSTRACT**  
Recently, several techniques to learn the embedding for a given graph dataset have been proposed. Among them, Graph2vec is significant in that it unsupervisedly learns the embedding of entire graphs which is useful for graph classification. This paper develops an algorithm which improves Graph2vec. First, we point out two limitations of Graph2vec: (1) Edge labels cannot be handled and (2) Graph2vec does not always preserve structural information enough to evaluate the structural similarity, because it bundles the node label information and the structural information in extracting subgraphs. Our algorithm overcomes these limitations by exploiting the line graphs (edge-to-vertex dual graphs) of given graphs. Specifically, it complements either the edge label information or the structural information which Graph2vec misses with the embeddings of the line graphs. Our method is named as GL2vec (Graph and Line graph to vector) because it concatenates the embedding of an original graph to that of the corresponding line graph. Experimentally, GL2vec achieves significant improvements in graph classification task over Graph2vec for many benchmark datasets.

{{< /ci-details >}}

{{< ci-details summary="ProSNet: integrating homology with molecular networks for protein function prediction (Sheng Wang et al., 2017)">}}

Sheng Wang, Meng Qu, Jian Peng. (2017)  
**ProSNet: integrating homology with molecular networks for protein function prediction**  
PSB  
[Paper Link](https://www.semanticscholar.org/paper/a7f0d7d7ce9af0e20ab2854563d269bd2beb0cbf)  
Influential Citation Count (0), SS-ID (a7f0d7d7ce9af0e20ab2854563d269bd2beb0cbf)  

**ABSTRACT**  
Automated annotation of protein function has become a critical task in the post-genomic era. Network-based approaches and homology-based approaches have been widely used and recently tested in large-scale community-wide assessment experiments. It is natural to integrate network data with homology information to further improve the predictive performance. However, integrating these two heterogeneous, high-dimensional and noisy datasets is non-trivial. In this work, we introduce a novel protein function prediction algorithm ProSNet. An integrated heterogeneous network is first built to include molecular networks of multiple species and link together homologous proteins across multiple species. Based on this integrated network, a dimensionality reduction algorithm is introduced to obtain compact low-dimensional vectors to encode proteins in the network. Finally, we develop machine learning classification algorithms that take the vectors as input and make predictions by transferring annotations both within each species and across different species. Extensive experiments on five major species demonstrate that our integration of homology with molecular networks substantially improves the predictive performance over existing approaches.

{{< /ci-details >}}

{{< ci-details summary="Compression of weighted graphs (Hannu (TT) Toivonen et al., 2011)">}}

Hannu (TT) Toivonen, Fang Zhou, Aleksi Hartikainen, Atte Hinkka. (2011)  
**Compression of weighted graphs**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/a8b6b6baaa0d81ed01bb5f387b7ab14f7f234393)  
Influential Citation Count (14), SS-ID (a8b6b6baaa0d81ed01bb5f387b7ab14f7f234393)  

**ABSTRACT**  
We propose to compress weighted graphs (networks), motivated by the observation that large networks of social, biological, or other relations can be complex to handle and visualize. In the process also known as graph simplification, nodes and (unweighted) edges are grouped to supernodes and superedges, respectively, to obtain a smaller graph. We propose models and algorithms for weighted graphs. The interpretation (i.e. decompression) of a compressed, weighted graph is that a pair of original nodes is connected by an edge if their supernodes are connected by one, and that the weight of an edge is approximated to be the weight of the superedge. The compression problem now consists of choosing supernodes, superedges, and superedge weights so that the approximation error is minimized while the amount of compression is maximized.  In this paper, we formulate this task as the 'simple weighted graph compression problem'. We then propose a much wider class of tasks under the name of 'generalized weighted graph compression problem'. The generalized task extends the optimization to preserve longer-range connectivities between nodes, not just individual edge weights. We study the properties of these problems and propose a range of algorithms to solve them, with different balances between complexity and quality of the result. We evaluate the problems and algorithms experimentally on real networks. The results indicate that weighted graphs can be compressed efficiently with relatively little compression error.

{{< /ci-details >}}

{{< ci-details summary="Graph Convolutional Network with Sequential Attention for Goal-Oriented Dialogue Systems (Suman Banerjee et al., 2019)">}}

Suman Banerjee, Mitesh M. Khapra. (2019)  
**Graph Convolutional Network with Sequential Attention for Goal-Oriented Dialogue Systems**  
Transactions of the Association for Computational Linguistics  
[Paper Link](https://www.semanticscholar.org/paper/a9c895dc9d6443588ffd9d6c748215d8c48209a0)  
Influential Citation Count (0), SS-ID (a9c895dc9d6443588ffd9d6c748215d8c48209a0)  

**ABSTRACT**  
Abstract Domain-specific goal-oriented dialogue systems typically require modeling three types of inputs, namely, (i) the knowledge-base associated with the domain, (ii) the history of the conversation, which is a sequence of utterances, and (iii) the current utterance for which the response needs to be generated. While modeling these inputs, current state-of-the-art models such as Mem2Seq typically ignore the rich structure inherent in the knowledge graph and the sentences in the conversation context. Inspired by the recent success of structure-aware Graph Convolutional Networks (GCNs) for various NLP tasks such as machine translation, semantic role labeling, and document dating, we propose a memory-augmented GCN for goal-oriented dialogues. Our model exploits (i) the entity relation graph in a knowledge-base and (ii) the dependency graph associated with an utterance to compute richer representations for words and entities. Further, we take cognizance of the fact that in certain situations, such as when the conversation is in a code-mixed language, dependency parsers may not be available. We show that in such situations we could use the global word co-occurrence graph to enrich the representations of utterances. We experiment with four datasets: (i) the modified DSTC2 dataset, (ii) recently released code-mixed versions of DSTC2 dataset in four languages, (iii) Wizard-of-Oz style CAM676 dataset, and (iv) Wizard-of-Oz style MultiWOZ dataset. On all four datasets our method outperforms existing methods, on a wide range of evaluation metrics.

{{< /ci-details >}}

{{< ci-details summary="Signed networks in social media (J. Leskovec et al., 2010)">}}

J. Leskovec, D. Huttenlocher, J. Kleinberg. (2010)  
**Signed networks in social media**  
CHI  
[Paper Link](https://www.semanticscholar.org/paper/aa3c0e570211d30b26121d5172cef27425ff71ad)  
Influential Citation Count (101), SS-ID (aa3c0e570211d30b26121d5172cef27425ff71ad)  

**ABSTRACT**  
Relations between users on social media sites often reflect a mixture of positive (friendly) and negative (antagonistic) interactions. In contrast to the bulk of research on social networks that has focused almost exclusively on positive interpretations of links between people, we study how the interplay between positive and negative relationships affects the structure of on-line social networks. We connect our analyses to theories of signed networks from social psychology. We find that the classical theory of structural balance tends to capture certain common patterns of interaction, but that it is also at odds with some of the fundamental phenomena we observe --- particularly related to the evolving, directed nature of these on-line networks. We then develop an alternate theory of status that better explains the observed edge signs and provides insights into the underlying social mechanisms. Our work provides one of the first large-scale evaluations of theories of signed networks using on-line datasets, as well as providing a perspective for reasoning about social media sites.

{{< /ci-details >}}

{{< ci-details summary="Spherical and Hyperbolic Embeddings of Data (Richard C. Wilson et al., 2014)">}}

Richard C. Wilson, E. Hancock, E. Pekalska, R. Duin. (2014)  
**Spherical and Hyperbolic Embeddings of Data**  
IEEE Transactions on Pattern Analysis and Machine Intelligence  
[Paper Link](https://www.semanticscholar.org/paper/ab0e17513d180a68aad7680b3deea3844176cedc)  
Influential Citation Count (5), SS-ID (ab0e17513d180a68aad7680b3deea3844176cedc)  

**ABSTRACT**  
Many computer vision and pattern recognition problems may be posed as the analysis of a set of dissimilarities between objects. For many types of data, these dissimilarities are not euclidean (i.e., they do not represent the distances between points in a euclidean space), and therefore cannot be isometrically embedded in a euclidean space. Examples include shape-dissimilarities, graph distances and mesh geodesic distances. In this paper, we provide a means of embedding such non-euclidean data onto surfaces of constant curvature. We aim to embed the data on a space whose radius of curvature is determined by the dissimilarity data. The space can be either of positive curvature (spherical) or of negative curvature (hyperbolic). We give an efficient method for solving the spherical and hyperbolic embedding problems on symmetric dissimilarity data. Our approach gives the radius of curvature and a method for approximating the objects as points on a hyperspherical manifold without optimisation. For objects which do not reside exactly on the manifold, we develop a optimisation-based procedure for approximate embedding on a hyperspherical manifold. We use the exponential map between the manifold and its local tangent space to solve the optimisation problem locally in the euclidean tangent space. This process is efficient enough to allow us to embed data sets of several thousand objects. We apply our method to a variety of data including time warping functions, shape similarities, graph similarity and gesture similarity data. In each case the embedding maintains the local structure of the data while placing the points in a metric space.

{{< /ci-details >}}

{{< ci-details summary="Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks (Shikhar Vashishth et al., 2018)">}}

Shikhar Vashishth, Manik Bhandari, Prateek Yadav, Piyush Rai, C. Bhattacharyya, P. Talukdar. (2018)  
**Incorporating Syntactic and Semantic Information in Word Embeddings using Graph Convolutional Networks**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/ab571a354f0847677862da027a69db9531eb08e8)  
Influential Citation Count (9), SS-ID (ab571a354f0847677862da027a69db9531eb08e8)  

**ABSTRACT**  
Word embeddings have been widely adopted across several NLP applications. Most existing word embedding methods utilize sequential context of a word to learn its embedding. While there have been some attempts at utilizing syntactic context of a word, such methods result in an explosion of the vocabulary size. In this paper, we overcome this problem by proposing SynGCN, a flexible Graph Convolution based method for learning word embeddings. SynGCN utilizes the dependency context of a word without increasing the vocabulary size. Word embeddings learned by SynGCN outperform existing methods on various intrinsic and extrinsic tasks and provide an advantage when used with ELMo. We also propose SemGCN, an effective framework for incorporating diverse semantic knowledge for further enhancing learned word representations. We make the source code of both models available to encourage reproducible research.

{{< /ci-details >}}

{{< ci-details summary="Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning (Sijing Zhou et al., 2020)">}}

Sijing Zhou, Xinyi Dai, Haokun Chen, Weinan Zhang, Kan Ren, Ruiming Tang, Xiuqiang He, Yong Yu. (2020)  
**Interactive Recommender System via Knowledge Graph-enhanced Reinforcement Learning**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/ac738ab5b57c2e337da303a5e1ec0e2c81c4d963)  
Influential Citation Count (5), SS-ID (ac738ab5b57c2e337da303a5e1ec0e2c81c4d963)  

**ABSTRACT**  
Interactive recommender system (IRS) has drawn huge attention because of its flexible recommendation strategy and the consideration of optimal long-term user experiences. To deal with the dynamic user preference and optimize accumulative utilities, researchers have introduced reinforcement learning (RL) into IRS. However, RL methods share a common issue of sample efficiency, i.e., huge amount of interaction data is required to train an effective recommendation policy, which is caused by the sparse user responses and the large action space consisting of a large number of candidate items. Moreover, it is infeasible to collect much data with explorative policies in online environments, which will probably harm user experience. In this work, we investigate the potential of leveraging knowledge graph (KG) in dealing with these issues of RL methods for IRS, which provides rich side information for recommendation decision making. Instead of learning RL policies from scratch, we make use of the prior knowledge of the item correlation learned from KG to (i) guide the candidate selection for better candidate item retrieval, (ii) enrich the representation of items and user states, and (iii) propagate user preferences among the correlated items over KG to deal with the sparsity of user feedback. Comprehensive experiments have been conducted on two real-world datasets, which demonstrate the superiority of our approach with significant improvements against state-of-the-arts.

{{< /ci-details >}}

{{< ci-details summary="TemporalNode2vec: Temporal Node Embedding in Temporal Networks (Mounir Haddad et al., 2019)">}}

Mounir Haddad, Cécile Bothorel, P. Lenca, Dominique Bedart. (2019)  
**TemporalNode2vec: Temporal Node Embedding in Temporal Networks**  
COMPLEX NETWORKS  
[Paper Link](https://www.semanticscholar.org/paper/ad43d8ba1b9619211052615f24da3ecb3c8519db)  
Influential Citation Count (0), SS-ID (ad43d8ba1b9619211052615f24da3ecb3c8519db)  

**ABSTRACT**  
The goal of graph embedding is to learn a representation of graphs vertices in a latent low-dimensional space in order to encode the structural information that lies in graphs. While real-world networks evolve over time, the majority of research focuses on static networks, ignoring local and global evolution patterns. A simplistic approach consists of learning nodes embeddings independently for each time step. This can cause unstable and inefficient representations over time.

{{< /ci-details >}}

{{< ci-details summary="Author2Vec: Learning Author Representations by Combining Content and Link Information (Ganesh Jawahar et al., 2016)">}}

Ganesh Jawahar, S. Ganguly, Manish Gupta, Vasudeva Varma, Vikram Pudi. (2016)  
**Author2Vec: Learning Author Representations by Combining Content and Link Information**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/aee808e9c5fb20b2dd1a51bb020112cdb908d80b)  
Influential Citation Count (3), SS-ID (aee808e9c5fb20b2dd1a51bb020112cdb908d80b)  

**ABSTRACT**  
In this paper, we consider the problem of learning representations for authors from bibliographic co-authorship networks. Existing methods for deep learning on graphs, such as DeepWalk, suffer from link sparsity problem as they focus on modeling the link information only. We hypothesize that capturing both the content and link information in a unified way will help mitigate the sparsity problem. To this end, we present a novel model 'Author2Vec', which learns low-dimensional author representations such that authors who write similar content and share similar network structure are closer in vector space. Such embeddings are useful in a variety of applications such as link prediction, node classification, recommendation and visualization. The author embeddings we learn are empirically shown to outperform DeepWalk by 2.35% and 0.83% for link prediction and clustering task respectively.

{{< /ci-details >}}

{{< ci-details summary="GraPASA: Parametric graph embedding via siamese architecture (Yujun Chen et al., 2020)">}}

Yujun Chen, Ke Sun, Juhua Pu, Zhang Xiong, Xiangliang Zhang. (2020)  
**GraPASA: Parametric graph embedding via siamese architecture**  
Inf. Sci.  
[Paper Link](https://www.semanticscholar.org/paper/af46a816ad045238c1c8cca734ec9d1f48279d8f)  
Influential Citation Count (0), SS-ID (af46a816ad045238c1c8cca734ec9d1f48279d8f)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Explainable, Stable, and Scalable Graph Convolutional Networks for Learning Graph Representation (Ping-En Lu et al., 2020)">}}

Ping-En Lu, Cheng-Shang Chang. (2020)  
**Explainable, Stable, and Scalable Graph Convolutional Networks for Learning Graph Representation**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/af4b880f6b30b5510cc78092dcf71d3ea52329e0)  
Influential Citation Count (0), SS-ID (af4b880f6b30b5510cc78092dcf71d3ea52329e0)  

**ABSTRACT**  
The network embedding problem that maps nodes in a graph to vectors in Euclidean space can be very useful for addressing several important tasks on a graph. Recently, graph neural networks (GNNs) have been proposed for solving such a problem. However, most embedding algorithms and GNNs are difficult to interpret and do not scale well to handle millions of nodes. In this paper, we tackle the problem from a new perspective based on the equivalence of three constrained optimization problems: the network embedding problem, the trace maximization problem of the modularity matrix in a sampled graph, and the matrix factorization problem of the modularity matrix in a sampled graph. The optimal solutions to these three problems are the dominant eigenvectors of the modularity matrix. We proposed two algorithms that belong to a special class of graph convolutional networks (GCNs) for solving these problems: (i) Clustering As Feature Embedding GCN (CAFE-GCN) and (ii) sphere-GCN. Both algorithms are stable trace maximization algorithms, and they yield good approximations of dominant eigenvectors. Moreover, there are linear-time implementations for sparse graphs. In addition to solving the network embedding problem, both proposed GCNs are capable of performing dimensionality reduction. Various experiments are conducted to evaluate our proposed GCNs and show that our proposed GCNs outperform almost all the baseline methods. Moreover, CAFE-GCN could be benefited from the labeled data and have tremendous improvements in various performance metrics.

{{< /ci-details >}}

{{< ci-details summary="Nonlinear dimensionality reduction by locally linear embedding. (S. Roweis et al., 2000)">}}

S. Roweis, L. Saul. (2000)  
**Nonlinear dimensionality reduction by locally linear embedding.**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/afcd6da7637ddeef6715109aca248da7a24b1c65)  
Influential Citation Count (1523), SS-ID (afcd6da7637ddeef6715109aca248da7a24b1c65)  

**ABSTRACT**  
Many areas of science depend on exploratory data analysis and visualization. The need to analyze large amounts of multivariate data raises the fundamental problem of dimensionality reduction: how to discover compact representations of high-dimensional data. Here, we introduce locally linear embedding (LLE), an unsupervised learning algorithm that computes low-dimensional, neighborhood-preserving embeddings of high-dimensional inputs. Unlike clustering methods for local dimensionality reduction, LLE maps its inputs into a single global coordinate system of lower dimensionality, and its optimizations do not involve local minima. By exploiting the local symmetries of linear reconstructions, LLE is able to learn the global structure of nonlinear manifolds, such as those generated by images of faces or documents of text.

{{< /ci-details >}}

{{< ci-details summary="Exponential Family Graph Embeddings (Abdulkadir Çelikkanat et al., 2019)">}}

Abdulkadir Çelikkanat, Fragkiskos D. Malliaros. (2019)  
**Exponential Family Graph Embeddings**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/b176355c33564beb8c7864572877ee4c7dcb40c4)  
Influential Citation Count (0), SS-ID (b176355c33564beb8c7864572877ee4c7dcb40c4)  

**ABSTRACT**  
Representing networks in a low dimensional latent space is a crucial task with many interesting applications in graph learning problems, such as link prediction and node classification. A widely applied network representation learning paradigm is based on the combination of random walks for sampling context nodes and the traditional \textit{Skip-Gram} model to capture center-context node relationships. In this paper, we emphasize on exponential family distributions to capture rich interaction patterns between nodes in random walk sequences. We introduce the generic \textit{exponential family graph embedding} model, that generalizes random walk-based network representation learning techniques to exponential family conditional distributions. We study three particular instances of this model, analyzing their properties and showing their relationship to existing unsupervised learning models. Our experimental evaluation on real-world datasets demonstrates that the proposed techniques outperform well-known baseline methods in two downstream machine learning tasks.

{{< /ci-details >}}

{{< ci-details summary="Coherent Comments Generation for Chinese Articles with a Graph-to-Sequence Model (Wei Li et al., 2019)">}}

Wei Li, Jingjing Xu, Yancheng He, Shengli Yan, Yunfang Wu, Xu Sun. (2019)  
**Coherent Comments Generation for Chinese Articles with a Graph-to-Sequence Model**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/b2125d912941244c243a33e31b01e34467cea457)  
Influential Citation Count (6), SS-ID (b2125d912941244c243a33e31b01e34467cea457)  

**ABSTRACT**  
Automatic article commenting is helpful in encouraging user engagement on online news platforms. However, the news documents are usually too long for models under traditional encoder-decoder frameworks, which often results in general and irrelevant comments. In this paper, we propose to generate comments with a graph-to-sequence model that models the input news as a topic interaction graph. By organizing the article into graph structure, our model can better understand the internal structure of the article and the connection between topics, which makes it better able to generate coherent and informative comments. We collect and release a large scale news-comment corpus from a popular Chinese online news platform Tencent Kuaibao. Extensive experiment results show that our model can generate much more coherent and informative comments compared with several strong baseline models.

{{< /ci-details >}}

{{< ci-details summary="Finding and evaluating community structure in networks. (M. Newman et al., 2003)">}}

M. Newman, M. Girvan. (2003)  
**Finding and evaluating community structure in networks.**  
Physical review. E, Statistical, nonlinear, and soft matter physics  
[Paper Link](https://www.semanticscholar.org/paper/b222526a2990d9073d734e2a1830210ca14cd8bd)  
Influential Citation Count (1165), SS-ID (b222526a2990d9073d734e2a1830210ca14cd8bd)  

**ABSTRACT**  
We propose and study a set of algorithms for discovering community structure in networks-natural divisions of network nodes into densely connected subgroups. Our algorithms all share two definitive features: first, they involve iterative removal of edges from the network to split it into communities, the edges removed being identified using any one of a number of possible "betweenness" measures, and second, these measures are, crucially, recalculated after each removal. We also propose a measure for the strength of the community structure found by our algorithms, which gives us an objective metric for choosing the number of communities into which a network should be divided. We demonstrate that our algorithms are highly effective at discovering community structure in both computer-generated and real-world network data, and show how they can be used to shed light on the sometimes dauntingly complex structure of networked systems.

{{< /ci-details >}}

{{< ci-details summary="Textbook Question Answering with Knowledge Graph Understanding and Unsupervised Open-set Text Comprehension (Daesik Kim et al., 2018)">}}

Daesik Kim, Seonhoon Kim, Nojun Kwak. (2018)  
**Textbook Question Answering with Knowledge Graph Understanding and Unsupervised Open-set Text Comprehension**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/b22ce969f203fb548c42034ae7fc78cc043fdc16)  
Influential Citation Count (0), SS-ID (b22ce969f203fb548c42034ae7fc78cc043fdc16)  

**ABSTRACT**  
In this work, we introduce a novel algorithm for solving the textbook question answering (TQA) task which describes more realistic QA problems compared to other recent tasks. We mainly focus on two related issues with analysis of TQA dataset. First, it requires to comprehend long lessons to extract knowledge. To tackle this issue of extracting knowledge features from long lessons, we establish knowledge graph from texts and incorporate graph convolutional network (GCN). Second, scientific terms are not spread over the chapters and data splits in TQA dataset. To overcome this so called `out-of-domain' issue, we add novel unsupervised text learning process without any annotations before learning QA problems. The experimental results show that our model significantly outperforms prior state-of-the-art methods. Moreover, ablation studies validate that both methods of incorporating GCN for extracting knowledge from long lessons and our newly proposed unsupervised learning process are meaningful to solve this problem.

{{< /ci-details >}}

{{< ci-details summary="Explicit Semantic Ranking for Academic Search via Knowledge Graph Embedding (Chenyan Xiong et al., 2017)">}}

Chenyan Xiong, Russell Power, Jamie Callan. (2017)  
**Explicit Semantic Ranking for Academic Search via Knowledge Graph Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/b30481dd5467a187b7e1a5a2dd326d97cafd95ac)  
Influential Citation Count (9), SS-ID (b30481dd5467a187b7e1a5a2dd326d97cafd95ac)  

**ABSTRACT**  
This paper introduces Explicit Semantic Ranking (ESR), a new ranking technique that leverages knowledge graph embedding. Analysis of the query log from our academic search engine, SemanticScholar.org, reveals that a major error source is its inability to understand the meaning of research concepts in queries. To addresses this challenge, ESR represents queries and documents in the entity space and ranks them based on their semantic connections from their knowledge graph embedding. Experiments demonstrate ESR's ability in improving Semantic Scholar's online production system, especially on hard queries where word-based ranking fails.

{{< /ci-details >}}

{{< ci-details summary="Robust Attribute and Structure Preserving Graph Embedding (B. Hettige et al., 2020)">}}

B. Hettige, Weiqing Wang, Yuan-Fang Li, Wray L. Buntine. (2020)  
**Robust Attribute and Structure Preserving Graph Embedding**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/b40f3e3b7bb11949b26fc97febdc2c0d973b8021)  
Influential Citation Count (0), SS-ID (b40f3e3b7bb11949b26fc97febdc2c0d973b8021)  

**ABSTRACT**  
Graph embedding methods are useful for a wide range of graph analysis tasks including link prediction and node classification. Most graph embedding methods learn only the topological structure of graphs. Nevertheless, it has been shown that the incorporation of node attributes is beneficial in improving the expressive power of node embeddings. However, real-world graphs are often noisy in terms of structure and/or attributes (missing and/or erroneous edges/attributes). Most existing graph embedding methods are susceptible to this noise, as they do not consider uncertainty during the modelling process. In this paper, we introduce RASE, a Robust Attribute and Structure preserving graph Embedding model. RASE is a novel graph representation learning model which effectively preserves both graph structure and node attributes through a unified loss function. To be robust, RASE uses a denoising attribute auto-encoder to deal with node attribute noise, and models uncertainty in the embedding space as Gaussians to cope with graph structure noise. We evaluate the performance of RASE through an extensive experimental study on various real-world datasets. Results demonstrate that RASE outperforms state-of-the-art embedding methods on multiple graph analysis tasks and is robust to both structure and attribute noise.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Graph Representation Learning With Variable Heat Kernel (Yongjun Jing et al., 2020)">}}

Yongjun Jing, Hao Wang, Kun Shao, X. Huo, Yangyang Zhang. (2020)  
**Unsupervised Graph Representation Learning With Variable Heat Kernel**  
IEEE Access  
[Paper Link](https://www.semanticscholar.org/paper/b48fa95aa2a63466c87d176ee2caf4e5b975c085)  
Influential Citation Count (1), SS-ID (b48fa95aa2a63466c87d176ee2caf4e5b975c085)  

**ABSTRACT**  
Graph representation learning aims to learn a low-dimension latent representation of nodes, and the learned representation is used for downstream graph analysis tasks. However, most of the existing graph embedding models focus on how to aggregate all the neighborhood node features to encode the semantic information into the representation and neglect the global structural features of the node such as community structure and centrality. In the paper, we propose a novel unsupervised graph representation learning method (VHKRep), where a variable heat kernel is designed to better capture implicit global features via heat diffusion with the different time scale and generate the robust node representation. We conduct extensive experiment on three real-world datasets for node classification and link prediction tasks. Compared with the state-of-the-art seven models, the experimental results demonstrate the effectiveness of our proposed method on both node classification and link prediction tasks.

{{< /ci-details >}}

{{< ci-details summary="CayleyNets: Graph Convolutional Neural Networks With Complex Rational Spectral Filters (R. Levie et al., 2017)">}}

R. Levie, Federico Monti, X. Bresson, M. Bronstein. (2017)  
**CayleyNets: Graph Convolutional Neural Networks With Complex Rational Spectral Filters**  
IEEE Transactions on Signal Processing  
[Paper Link](https://www.semanticscholar.org/paper/b5007972c6f5a2294f83357c73e12664dd7c85b3)  
Influential Citation Count (25), SS-ID (b5007972c6f5a2294f83357c73e12664dd7c85b3)  

**ABSTRACT**  
The rise of graph-structured data such as social networks, regulatory networks, citation graphs, and functional brain networks, in combination with resounding success of deep learning in various applications, has brought the interest in generalizing deep learning models to non-Euclidean domains. In this paper, we introduce a new spectral domain convolutional architecture for deep learning on graphs. The core ingredient of our model is a new class of parametric rational complex functions (Cayley polynomials) allowing to efficiently compute spectral filters on graphs that specialize on frequency bands of interest. Our model generates rich spectral filters that are localized in space, scales linearly with the size of the input data for sparsely connected graphs, and can handle different constructions of Laplacian operators. Extensive experimental results show the superior performance of our approach, in comparison to other spectral domain convolutional architectures, on spectral image classification, community detection, vertex classification, and matrix completion tasks.

{{< /ci-details >}}

{{< ci-details summary="Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints (K. Hayashi et al., 2020)">}}

K. Hayashi, M. Ohsaki. (2020)  
**Reinforcement Learning and Graph Embedding for Binary Truss Topology Optimization Under Stress and Displacement Constraints**  
Frontiers in Built Environment  
[Paper Link](https://www.semanticscholar.org/paper/b518e6b261a4272793accb44205eccc86d041e80)  
Influential Citation Count (0), SS-ID (b518e6b261a4272793accb44205eccc86d041e80)  

**ABSTRACT**  
This paper addresses a combined method of reinforcement learning and graph embedding for binary topology optimization of trusses to minimize total structural volume under stress and displacement constraints. Although conventional deep learning methods owe their success to a convolutional neural network that is capable of capturing higher level latent information from pixels, the convolution is difficult to apply to discrete structures due to their irregular connectivity. Instead, a method based on graph embedding is proposed here to extract the features of bar members. This way, all the members have a feature vector with the same size representing their neighbor information such as connectivity and force flows from the loaded nodes to the supports. The features are used to implement reinforcement learning where an action taker called agent is trained to sequentially eliminate unnecessary members from Level-1 ground structure, where all neighboring nodes are connected by members. The trained agent is capable of finding sub-optimal solutions at a low computational cost, and it is reusable to other trusses with different geometry, topology, and boundary conditions.

{{< /ci-details >}}

{{< ci-details summary="ET-GRU: using multi-layer gated recurrent units to identify electron transport proteins (N. Le et al., 2019)">}}

N. Le, E. Yapp, Hui-Yuan Yeh. (2019)  
**ET-GRU: using multi-layer gated recurrent units to identify electron transport proteins**  
BMC Bioinformatics  
[Paper Link](https://www.semanticscholar.org/paper/b5cc2c4a7409fee41a7ab0d920a3cb2e3e165a27)  
Influential Citation Count (0), SS-ID (b5cc2c4a7409fee41a7ab0d920a3cb2e3e165a27)  

**ABSTRACT**  
BackgroundElectron transport chain is a series of protein complexes embedded in the process of cellular respiration, which is an important process to transfer electrons and other macromolecules throughout the cell. It is also the major process to extract energy via redox reactions in the case of oxidation of sugars. Many studies have determined that the electron transport protein has been implicated in a variety of human diseases, i.e. diabetes, Parkinson, Alzheimer’s disease and so on. Few bioinformatics studies have been conducted to identify the electron transport proteins with high accuracy, however, their performance results require a lot of improvements. Here, we present a novel deep neural network architecture to address this problem.ResultsMost of the previous studies could not use the original position specific scoring matrix (PSSM) profiles to feed into neural networks, leading to a lack of information and the neural networks consequently could not achieve the best results. In this paper, we present a novel approach by using deep gated recurrent units (GRU) on full PSSMs to resolve this problem. Our approach can precisely predict the electron transporters with the cross-validation and independent test accuracy of 93.5 and 92.3%, respectively. Our approach demonstrates superior performance to all of the state-of-the-art predictors on electron transport proteins.ConclusionsThrough the proposed study, we provide ET-GRU, a web server for discriminating electron transport proteins in particular and other protein functions in general. Also, our achievement could promote the use of GRU in computational biology, especially in protein function prediction.

{{< /ci-details >}}

{{< ci-details summary="ATP: Directed Graph Embedding with Asymmetric Transitivity Preservation (Jiankai Sun et al., 2018)">}}

Jiankai Sun, Bortik Bandyopadhyay, Armin Bashizade, Jiongqian Liang, P. Sadayappan, S. Parthasarathy. (2018)  
**ATP: Directed Graph Embedding with Asymmetric Transitivity Preservation**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/b764d0070d07957d4b9621988ea3d020e9ecbe36)  
Influential Citation Count (4), SS-ID (b764d0070d07957d4b9621988ea3d020e9ecbe36)  

**ABSTRACT**  
Directed graphs have been widely used in Community Question Answering services (CQAs) to model asymmetric relationships among different types of nodes in CQA graphs, e.g., question, answer, user. Asymmetric transitivity is an essential property of directed graphs, since it can play an important role in downstream graph inference and analysis. Question difficulty and user expertise follow the characteristic of asymmetric transitivity. Maintaining such properties, while reducing the graph to a lower dimensional vector embedding space, has been the focus of much recent research. In this paper, we tackle the challenge of directed graph embedding with asymmetric transitivity preservation and then leverage the proposed embedding method to solve a fundamental task in CQAs: how to appropriately route and assign newly posted questions to users with the suitable expertise and interest in CQAs. The technique incorporates graph hierarchy and reachability information naturally by relying on a nonlinear transformation that operates on the core reachability and implicit hierarchy within such graphs. Subsequently, the methodology levers a factorization-based approach to generate two embedding vectors for each node within the graph, to capture the asymmetric transitivity. Extensive experiments show that our framework consistently and significantly outperforms the state-of-the-art baselines on three diverse realworld tasks: link prediction, and question difficulty estimation and expert finding in online forums like Stack Exchange. Particularly, our framework can support inductive embedding learning for newly posted questions (unseen nodes during training), and therefore can properly route and assign these kinds of questions to experts in CQAs.

{{< /ci-details >}}

{{< ci-details summary="Predicting multicellular function through multi-layer tissue networks (M. Zitnik et al., 2017)">}}

M. Zitnik, J. Leskovec. (2017)  
**Predicting multicellular function through multi-layer tissue networks**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/b7c4570d7d97f327e7f82fe28100172ec5e94cac)  
Influential Citation Count (19), SS-ID (b7c4570d7d97f327e7f82fe28100172ec5e94cac)  

**ABSTRACT**  
Motivation: Understanding functions of proteins in specific human tissues is essential for insights into disease diagnostics and therapeutics, yet prediction of tissue‐specific cellular function remains a critical challenge for biomedicine. Results: Here, we present OhmNet, a hierarchy‐aware unsupervised node feature learning approach for multi‐layer networks. We build a multi‐layer network, where each layer represents molecular interactions in a different human tissue. OhmNet then automatically learns a mapping of proteins, represented as nodes, to a neural embedding‐based low‐dimensional space of features. OhmNet encourages sharing of similar features among proteins with similar network neighborhoods and among proteins activated in similar tissues. The algorithm generalizes prior work, which generally ignores relationships between tissues, by modeling tissue organization with a rich multiscale tissue hierarchy. We use OhmNet to study multicellular function in a multi‐layer protein interaction network of 107 human tissues. In 48 tissues with known tissue‐specific cellular functions, OhmNet provides more accurate predictions of cellular function than alternative approaches, and also generates more accurate hypotheses about tissue‐specific protein actions. We show that taking into account the tissue hierarchy leads to improved predictive power. Remarkably, we also demonstrate that it is possible to leverage the tissue hierarchy in order to effectively transfer cellular functions to a functionally uncharacterized tissue. Overall, OhmNet moves from flat networks to multiscale models able to predict a range of phenotypes spanning cellular subsystems. Availability and implementation: Source code and datasets are available at http://snap.stanford.edu/ohmnet. Contact: jure@cs.stanford.edu

{{< /ci-details >}}

{{< ci-details summary="Semi-supervised Learning on Graphs with Generative Adversarial Nets (Ming Ding et al., 2018)">}}

Ming Ding, Jie Tang, Jie Zhang. (2018)  
**Semi-supervised Learning on Graphs with Generative Adversarial Nets**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/b8da4337c92acda632e8138be1b525a3aef54b85)  
Influential Citation Count (5), SS-ID (b8da4337c92acda632e8138be1b525a3aef54b85)  

**ABSTRACT**  
We investigate how generative adversarial nets (GANs) can help semi-supervised learning on graphs. We first provide insights on working principles of adversarial learning over graphs and then present GraphSGAN, a novel approach to semi-supervised learning on graphs. In GraphSGAN, generator and classifier networks play a novel competitive game. At equilibrium, generator generates fake samples in low-density areas between subgraphs. In order to discriminate fake samples from the real, classifier implicitly takes the density property of subgraph into consideration. An efficient adversarial learning algorithm has been developed to improve traditional normalized graph Laplacian regularization with a theoretical guarantee. Experimental results on several different genres of datasets show that the proposed GraphSGAN significantly outperforms several state-of-the-art methods. GraphSGAN can be also trained using mini-batch, thus enjoys the scalability advantage.

{{< /ci-details >}}

{{< ci-details summary="Atrributed Graph Embedding Based on Multiobjective Evolutionary Algorithm for Overlapping Community Detection (Xiangyi Teng et al., 2020)">}}

Xiangyi Teng, Jing Liu. (2020)  
**Atrributed Graph Embedding Based on Multiobjective Evolutionary Algorithm for Overlapping Community Detection**  
2020 IEEE Congress on Evolutionary Computation (CEC)  
[Paper Link](https://www.semanticscholar.org/paper/b92a31918b95ae220d1b23f0ecc4f0f8cf00599f)  
Influential Citation Count (0), SS-ID (b92a31918b95ae220d1b23f0ecc4f0f8cf00599f)  

**ABSTRACT**  
Graph embedding methods aim to represent nodes in the network into a low-dimensional and continuous vector space while preserving the topological structure and varieties of relational information maximally. Nowadays the structural connections of networks and the attribute information about each node are more easily available than before. As a result, many community detection algorithms for attributed networks have been proposed. However, the majority of these methods cannot deal with the overlapping community detection problem, which is one of the most significant issues in the real-world complex network study. In addition, it is quite challenging to make full use of both structural and attribute information instead of only focusing on one part. To this end, in this paper we innovatively combine the graph embedding with multiobjective evolutionary algorithms (MOEAs) for overlapping community detection problems in attributed networks. As far as I am concerned, MOEA is first used to integrate with graph embedding methods for overlapping community detection. We term our method as MOEA-GEOV, which can automatically determine the number of communities without any prior knowledge and consider topological structure and vertex properties synchronously. In MOEA-GEOV, two objective functions concerning community structure and attribute similarity are carefully designed. Moreover, a heuristic initialization method is proposed to get a relatively good initial population. Then a novel encoding and decoding strategy is designed to efficiently represent the overlapping communities and corresponding embedded representation. In the experiments, the performance of MOEA-GEOV is validated on both single and multiple attribute real-world networks. The experimental results of community detection tasks demonstrate our method can effectively obtain overlapping community structures with practical significance.

{{< /ci-details >}}

{{< ci-details summary="Normalized cuts and image segmentation (Jianbo Shi et al., 1997)">}}

Jianbo Shi, J. Malik. (1997)  
**Normalized cuts and image segmentation**  
Proceedings of IEEE Computer Society Conference on Computer Vision and Pattern Recognition  
[Paper Link](https://www.semanticscholar.org/paper/b94c7ff9532ab26c3aedbee3988ec4c7a237c173)  
Influential Citation Count (1462), SS-ID (b94c7ff9532ab26c3aedbee3988ec4c7a237c173)  

**ABSTRACT**  
We propose a novel approach for solving the perceptual grouping problem in vision. Rather than focusing on local features and their consistencies in the image data, our approach aims at extracting the global impression of an image. We treat image segmentation as a graph partitioning problem and propose a novel global criterion, the normalized cut, for segmenting the graph. The normalized cut criterion measures both the total dissimilarity between the different groups as well as the total similarity within the groups. We show that an efficient computational technique based on a generalized eigenvalue problem can be used to optimize this criterion. We have applied this approach to segmenting static images and found results very encouraging.

{{< /ci-details >}}

{{< ci-details summary="Expert Finding for Community-Based Question Answering via Ranking Metric Network Learning (Zhou Zhao et al., 2016)">}}

Zhou Zhao, Qifan Yang, Deng Cai, Xiaofei He, Yueting Zhuang. (2016)  
**Expert Finding for Community-Based Question Answering via Ranking Metric Network Learning**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/b98ae01668cae9c01592dab5993cf04a23f0c91a)  
Influential Citation Count (6), SS-ID (b98ae01668cae9c01592dab5993cf04a23f0c91a)  

**ABSTRACT**  
Expert finding for question answering is a challenging problem in Community-based Question Answering (CQA) site, arising in many applications such as question routing and the identification of best answers. In order to provide high-quality experts, many existing approaches learn the user model mainly from their past question-answering activities in CQA sites, which suffer from the sparsity problem of CQA data. In this paper, we consider the problem of expert finding from the viewpoint of learning ranking metric embedding. We propose a novel ranking metric network learning framework for expert finding by exploiting both users' relative quality rank to given questions and their social relations. We then develop a random-walk based learning method with recurrent neural networks for ranking metric network embedding. The extensive experiments on a large-scale dataset from a real world CQA site show that our method achieves better performance than other state-of-the-art solutions to the problem.

{{< /ci-details >}}

{{< ci-details summary="Link Prediction Based on Graph Embedding Method in Unweighted Networks (Chencheng Wu et al., 2020)">}}

Chencheng Wu, Yinzuo Zhou, Lulu Tan, Cong Teng. (2020)  
**Link Prediction Based on Graph Embedding Method in Unweighted Networks**  
2020 39th Chinese Control Conference (CCC)  
[Paper Link](https://www.semanticscholar.org/paper/ba3159d8791903c7fcf0b6761cacf1be5b5ac927)  
Influential Citation Count (0), SS-ID (ba3159d8791903c7fcf0b6761cacf1be5b5ac927)  

**ABSTRACT**  
The index of link prediction based on random walk usually has the same transition probability in the process of particle transfer to its neighbor nodes, which has strong randomness and ignores the influence of the particularity of network topology on particle transition probability. In order to resolve this problem, this paper proposes a random walk with restart index based on graph embedding (GERWR). The algorithm uses graph embedding method to randomly sample network nodes and generate node representation vectors containing potential network structure information. By calculating the similarity of node vectors, it redefines a biased transition probability. We apply it to the process of random walk and explore the influence of the particularity of network topology on the transition during the particles walk. Finally, based on biased transition, the index proposed in this paper is compared with five classical similarity indexes in unweighted networks. The results show that the prediction algorithm based on graph embedding method with biased transfer has higher accuracy than other indexes.

{{< /ci-details >}}

{{< ci-details summary="Bibliographic Analysis with the Citation Network Topic Model (K. W. Lim et al., 2016)">}}

K. W. Lim, Wray L. Buntine. (2016)  
**Bibliographic Analysis with the Citation Network Topic Model**  
ACML  
[Paper Link](https://www.semanticscholar.org/paper/bca5472054bdcfd838a416448c070f769a373df6)  
Influential Citation Count (2), SS-ID (bca5472054bdcfd838a416448c070f769a373df6)  

**ABSTRACT**  
Bibliographic analysis considers author’s research areas, the citation network and paper content among other things. In this paper, we combine these three in a topic model that produces a bibliographic model of authors, topics and documents using a non-parametric extension of a combination of the Poisson mixed-topic link model and the author-topic model. We propose a novel and ecient inference algorithm for the model to explore subsets of research publications from CiteSeer X . Our model demonstrates improved performance in both model tting and a clustering task compared to several baselines.

{{< /ci-details >}}

{{< ci-details summary="Neuro-symbolic representation learning on biological knowledge graphs (Mona Alshahrani et al., 2016)">}}

Mona Alshahrani, Mohammed Asif Khan, Omar Maddouri, A. Kinjo, N. Queralt-Rosinach, R. Hoehndorf. (2016)  
**Neuro-symbolic representation learning on biological knowledge graphs**  
Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/bdd297aff687ac360efe28338f5dec884d301221)  
Influential Citation Count (8), SS-ID (bdd297aff687ac360efe28338f5dec884d301221)  

**ABSTRACT**  
Motivation : Biological data and knowledge bases increasingly rely on Semantic Web technologies and the use of knowledge graphs for data integration, retrieval and federated queries. In the past years, feature learning methods that are applicable to graph‐structured data are becoming available, but have not yet widely been applied and evaluated on structured biological knowledge. Results: We develop a novel method for feature learning on biological knowledge graphs. Our method combines symbolic methods, in particular knowledge representation using symbolic logic and automated reasoning, with neural networks to generate embeddings of nodes that encode for related information within knowledge graphs. Through the use of symbolic logic, these embeddings contain both explicit and implicit information. We apply these embeddings to the prediction of edges in the knowledge graph representing problems of function prediction, finding candidate genes of diseases, protein‐protein interactions, or drug target relations, and demonstrate performance that matches and sometimes outperforms traditional approaches based on manually crafted features. Our method can be applied to any biological knowledge graph, and will thereby open up the increasing amount of Semantic Web based knowledge bases in biology to use in machine learning and data analytics. Availability and implementation : https://github.com/bio‐ontology‐research‐group/walking‐rdf‐and‐owl Contact : robert.hoehndorf@kaust.edu.sa Supplementary information: Supplementary data are available at Bioinformatics online.

{{< /ci-details >}}

{{< ci-details summary="Multi-source information fusion based heterogeneous network embedding (Bentian Li et al., 2020)">}}

Bentian Li, D. Pi, Yunxia Lin, I. A. Khan, Lin Cui. (2020)  
**Multi-source information fusion based heterogeneous network embedding**  
Inf. Sci.  
[Paper Link](https://www.semanticscholar.org/paper/be7bdd550f75acfbdc435e2ca75252779ba9b871)  
Influential Citation Count (0), SS-ID (be7bdd550f75acfbdc435e2ca75252779ba9b871)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="TemporalGAT: Attention-Based Dynamic Graph Representation Learning (A. Fathy et al., 2020)">}}

A. Fathy, Kan Li. (2020)  
**TemporalGAT: Attention-Based Dynamic Graph Representation Learning**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/c06fc165523554b79ce59db9a8ce113b074359a0)  
Influential Citation Count (0), SS-ID (c06fc165523554b79ce59db9a8ce113b074359a0)  

**ABSTRACT**  
Learning representations for dynamic graphs is fundamental as it supports numerous graph analytic tasks such as dynamic link prediction, node classification, and visualization. Real-world dynamic graphs are continuously evolved where new nodes and edges are introduced or removed during graph evolution. Most existing dynamic graph representation learning methods focus on modeling dynamic graphs with fixed nodes due to the complexity of modeling dynamic graphs, and therefore, cannot efficiently learn the evolutionary patterns of real-world evolving graphs. Moreover, existing methods generally model the structural information of evolving graphs separately from temporal information. This leads to the loss of important structural and temporal information that could cause the degradation of predictive performance of the model. By employing an innovative neural network architecture based on graph attention networks and temporal convolutions, our framework jointly learns graph representations contemplating evolving graph structure and temporal patterns. We propose a deep attention model to learn low-dimensional feature representations which preserves the graph structure and features among series of graph snapshots over time. Experimental results on multiple real-world dynamic graph datasets show that, our proposed method is competitive against various state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="metapath2vec: Scalable Representation Learning for Heterogeneous Networks (Yuxiao Dong et al., 2017)">}}

Yuxiao Dong, N. Chawla, A. Swami. (2017)  
**metapath2vec: Scalable Representation Learning for Heterogeneous Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/c0af91371f426ff92117d2ccdadb2032bec23d2c)  
Influential Citation Count (164), SS-ID (c0af91371f426ff92117d2ccdadb2032bec23d2c)  

**ABSTRACT**  
We study the problem of representation learning in heterogeneous networks. Its unique challenges come from the existence of multiple types of nodes and links, which limit the feasibility of the conventional network embedding techniques. We develop two scalable representation learning models, namely metapath2vec and metapath2vec++. The metapath2vec model formalizes meta-path-based random walks to construct the heterogeneous neighborhood of a node and then leverages a heterogeneous skip-gram model to perform node embeddings. The metapath2vec++ model further enables the simultaneous modeling of structural and semantic correlations in heterogeneous networks. Extensive experiments show that metapath2vec and metapath2vec++ are able to not only outperform state-of-the-art embedding models in various heterogeneous network mining tasks, such as node classification, clustering, and similarity search, but also discern the structural and semantic correlations between diverse network objects.

{{< /ci-details >}}

{{< ci-details summary="Large-Scale Embedding Learning in Heterogeneous Event Data (Huan Gui et al., 2016)">}}

Huan Gui, Jialu Liu, Fangbo Tao, Meng Jiang, Brandon Norick, Jiawei Han. (2016)  
**Large-Scale Embedding Learning in Heterogeneous Event Data**  
2016 IEEE 16th International Conference on Data Mining (ICDM)  
[Paper Link](https://www.semanticscholar.org/paper/c18c30b9b1090e752031d23d219c1007b9954229)  
Influential Citation Count (10), SS-ID (c18c30b9b1090e752031d23d219c1007b9954229)  

**ABSTRACT**  
Heterogeneous events, which are defined as events connecting strongly-typed objects, are ubiquitous in the real world. We propose a HyperEdge-Based Embedding (Hebe) framework for heterogeneous event data, where a hyperedge represents the interaction among a set of involving objects in an event. The Hebe framework models the proximity among objects in an event by predicting a target object given the other participating objects in the event (hyperedge). Since each hyperedge encapsulates more information on a given event, Hebe is robust to data sparseness. In addition, Hebe is scalable when the data size spirals. Extensive experiments on large-scale real-world datasets demonstrate the efficacy and robustness of Hebe.

{{< /ci-details >}}

{{< ci-details summary="Understanding Coarsening for Embedding Large-Scale Graphs (Taha Atahan Akyildiz et al., 2020)">}}

Taha Atahan Akyildiz, Amro Alabsi Aljundi, K. Kaya. (2020)  
**Understanding Coarsening for Embedding Large-Scale Graphs**  
2020 IEEE International Conference on Big Data (Big Data)  
[Paper Link](https://www.semanticscholar.org/paper/c23308cf3cfc42002fcb212bcc6f5c9cd3f5d09e)  
Influential Citation Count (0), SS-ID (c23308cf3cfc42002fcb212bcc6f5c9cd3f5d09e)  

**ABSTRACT**  
A significant portion of the data today, e.g, social networks, web connections, etc., can be modeled by graphs. A proper analysis of graphs with Machine Learning (ML) algorithms has the potential to yield far-reaching insights into many areas of research and industry. However, the irregular structure of graph data constitutes an obstacle for running ML tasks on graphs such as link prediction, node classification, and anomaly detection. Graph embedding is a compute-intensive process of representing graphs as a set of vectors in a d-dimensional space, which in turn makes it amenable to ML tasks. Many approaches have been proposed in the literature to improve the performance of graph embedding, e.g., using distributed algorithms, accelerators, and pre-processing techniques. Graph coarsening, which can be considered a pre-processing step, is a structural approximation of a given, large graph with a smaller one. As the literature suggests, the cost of embedding significantly decreases when coarsening is employed. In this work, we thoroughly analyze the impact of the coarsening quality on the embedding performance both in terms of speed and accuracy. Our experiments with a state-of-the-art, fast graph embedding tool show that there is an interplay between the coarsening decisions taken and the embedding quality.

{{< /ci-details >}}

{{< ci-details summary="GraRep: Learning Graph Representations with Global Structural Information (Shaosheng Cao et al., 2015)">}}

Shaosheng Cao, Wei Lu, Qiongkai Xu. (2015)  
**GraRep: Learning Graph Representations with Global Structural Information**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  
Influential Citation Count (131), SS-ID (c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  

**ABSTRACT**  
In this paper, we present {GraRep}, a novel model for learning vertex representations of weighted graphs. This model learns low dimensional vectors to represent vertices appearing in a graph and, unlike existing work, integrates global structural information of the graph into the learning process. We also formally analyze the connections between our work and several previous research efforts, including the DeepWalk model of Perozzi et al. as well as the skip-gram model with negative sampling of Mikolov et al. We conduct experiments on a language network, a social network as well as a citation network and show that our learned global representations can be effectively used as features in tasks such as clustering, classification and visualization. Empirical results demonstrate that our representation significantly outperforms other state-of-the-art methods in such tasks.

{{< /ci-details >}}

{{< ci-details summary="Pre-training of Graph Augmented Transformers for Medication Recommendation (Junyuan Shang et al., 2019)">}}

Junyuan Shang, Tengfei Ma, Cao Xiao, Jimeng Sun. (2019)  
**Pre-training of Graph Augmented Transformers for Medication Recommendation**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/c3229debfda1b015c88404cf98f1074237d80809)  
Influential Citation Count (6), SS-ID (c3229debfda1b015c88404cf98f1074237d80809)  

**ABSTRACT**  
Medication recommendation is an important healthcare application. It is commonly formulated as a temporal prediction task. Hence, most existing works only utilize longitudinal electronic health records (EHRs) from a small number of patients with multiple visits ignoring a large number of patients with a single visit (selection bias). Moreover, important hierarchical knowledge such as diagnosis hierarchy is not leveraged in the representation learning process. Despite the success of deep learning techniques in computational phenotyping, most previous approaches have two limitations: task-oriented representation and ignoring hierarchies of medical codes.  To address these challenges, we propose G-BERT, a new model to combine the power of Graph Neural Networks (GNNs) and BERT (Bidirectional Encoder Representations from Transformers) for medical code representation and medication recommendation. We use GNNs to represent the internal hierarchical structures of medical codes. Then we integrate the GNN representation into a transformer-based visit encoder and pre-train it on EHR data from patients only with a single visit. The pre-trained visit encoder and representation are then fine-tuned for downstream predictive tasks on longitudinal EHRs from patients with multiple visits. G-BERT is the first to bring the language model pre-training schema into the healthcare domain and it achieved state-of-the-art performance on the medication recommendation task.

{{< /ci-details >}}

{{< ci-details summary="Signed Network Embedding in Social Media (Suhang Wang et al., 2017)">}}

Suhang Wang, Jiliang Tang, C. Aggarwal, Yi Chang, Huan Liu. (2017)  
**Signed Network Embedding in Social Media**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/c34336d3bfb7a3c22caa7958779f40bb2ab70a3d)  
Influential Citation Count (22), SS-ID (c34336d3bfb7a3c22caa7958779f40bb2ab70a3d)  

**ABSTRACT**  
Network embedding is to learn low-dimensional vector representations for nodes of a given social network, facilitating many tasks in social network analysis such as link prediction. The vast majority of existing embedding algorithms are designed for unsigned social networks or social networks with only positive links. However, networks in social media could have both positive and negative links, and little work exists for signed social networks. From recent findings of signed network analysis, it is evident that negative links have distinct properties and added value besides positive links, which brings about both challenges and opportunities for signed network embedding. In this paper, we propose a deep learning framework SiNE for signed network embedding. The framework optimizes an objective function guided by social theories that provide a fundamental understanding of signed social networks. Experimental results on two realworld datasets of social media demonstrate the effectiveness of the proposed framework SiNE.

{{< /ci-details >}}

{{< ci-details summary="Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling (Diego Marcheggiani et al., 2017)">}}

Diego Marcheggiani, Ivan Titov. (2017)  
**Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/c3a3c163f25b9181f1fb7e71a32482a7393d2088)  
Influential Citation Count (79), SS-ID (c3a3c163f25b9181f1fb7e71a32482a7393d2088)  

**ABSTRACT**  
Semantic role labeling (SRL) is the task of identifying the predicate-argument structure of a sentence. It is typically regarded as an important step in the standard NLP pipeline. As the semantic representations are closely related to syntactic ones, we exploit syntactic information in our model. We propose a version of graph convolutional networks (GCNs), a recent class of neural networks operating on graphs, suited to model syntactic dependency graphs. GCNs over syntactic dependency trees are used as sentence encoders, producing latent feature representations of words in a sentence. We observe that GCN layers are complementary to LSTM ones: when we stack both GCN and LSTM layers, we obtain a substantial improvement over an already state-of-the-art LSTM SRL model, resulting in the best reported scores on the standard benchmark (CoNLL-2009) both for Chinese and English.

{{< /ci-details >}}

{{< ci-details summary="Cross View Link Prediction by Learning Noise-resilient Representation Consensus (Xiaokai Wei et al., 2017)">}}

Xiaokai Wei, Linchuan Xu, Bokai Cao, Philip S. Yu. (2017)  
**Cross View Link Prediction by Learning Noise-resilient Representation Consensus**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/c3d62bcb84fc3a2aa9b8f4691677d7c02738f1bc)  
Influential Citation Count (3), SS-ID (c3d62bcb84fc3a2aa9b8f4691677d7c02738f1bc)  

**ABSTRACT**  
Link Prediction has been an important task for social and information networks. Existing approaches usually assume the completeness of network structure. However, in many real-world networks, the links and node attributes can usually be partially observable. In this paper, we study the problem of Cross View Link Prediction (CVLP) on partially observable networks, where the focus is to recommend nodes with only links to nodes with only attributes (or vice versa). We aim to bridge the information gap by learning a robust consensus for link-based and attribute-based representations so that nodes become comparable in the latent space. Also, the link-based and attribute-based representations can lend strength to each other via this consensus learning. Moreover, attribute selection is performed jointly with the representation learning to alleviate the effect of noisy high-dimensional attributes. We present two instantiations of this framework with different loss functions and develop an alternating optimization framework to solve the problem. Experimental results on four real-world datasets show the proposed algorithm outperforms the baseline methods significantly for cross-view link prediction.

{{< /ci-details >}}

{{< ci-details summary="Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (M. Defferrard et al., 2016)">}}

M. Defferrard, X. Bresson, P. Vandergheynst. (2016)  
**Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/c41eb895616e453dcba1a70c9b942c5063cc656c)  
Influential Citation Count (544), SS-ID (c41eb895616e453dcba1a70c9b942c5063cc656c)  

**ABSTRACT**  
In this work, we are interested in generalizing convolutional neural networks (CNNs) from low-dimensional regular grids, where image, video and speech are represented, to high-dimensional irregular domains, such as social networks, brain connectomes or words' embedding, represented by graphs. We present a formulation of CNNs in the context of spectral graph theory, which provides the necessary mathematical background and efficient numerical schemes to design fast localized convolutional filters on graphs. Importantly, the proposed technique offers the same linear computational complexity and constant learning complexity as classical CNNs, while being universal to any graph structure. Experiments on MNIST and 20NEWS demonstrate the ability of this novel deep learning system to learn local, stationary, and compositional features on graphs.

{{< /ci-details >}}

{{< ci-details summary="Adversarial Link Prediction in Social Networks (Kai Zhou et al., 2018)">}}

Kai Zhou, Tomasz P. Michalak, Talal Rahwan, Marcin Waniek, Y. Vorobeychik. (2018)  
**Adversarial Link Prediction in Social Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/c48bc4db8eacb97834c72b5fce3119040da011dc)  
Influential Citation Count (0), SS-ID (c48bc4db8eacb97834c72b5fce3119040da011dc)  

**ABSTRACT**  
Link prediction is one of the fundamental tools in social network analysis, used to identify relationships that are not otherwise observed. Commonly, link prediction is performed by means of a similarity metric, with the idea that a pair of similar nodes are likely to be connected. However, traditional link prediction based on similarity metrics assumes that available network data is accurate. We study the problem of adversarial link prediction, where an adversary aims to hide a target link by removing a limited subset of edges from the observed subgraph. We show that optimal attacks on local similarity metrics---that is, metrics which use only the information about the node pair and their network neighbors---can be found in linear time. In contrast, attacking Katz and ACT metrics which use global information about network topology is NP-Hard. We present an approximation algorithm for optimal attacks on Katz similarity, and a principled heuristic for ACT attacks. Extensive experiments demonstrate the efficacy of our methods.

{{< /ci-details >}}

{{< ci-details summary="Graph Convolutional Matrix Completion (Rianne van den Berg et al., 2017)">}}

Rianne van den Berg, Thomas Kipf, M. Welling. (2017)  
**Graph Convolutional Matrix Completion**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/c509de93b3d34ecd178f598814bd5177a0a29726)  
Influential Citation Count (121), SS-ID (c509de93b3d34ecd178f598814bd5177a0a29726)  

**ABSTRACT**  
We consider matrix completion for recommender systems from the point of view of link prediction on graphs. Interaction data such as movie ratings can be represented by a bipartite user-item graph with labeled edges denoting observed ratings. Building on recent progress in deep learning on graph-structured data, we propose a graph auto-encoder framework based on differentiable message passing on the bipartite interaction graph. Our model shows competitive performance on standard collaborative filtering benchmarks. In settings where complimentary feature information or structured data such as a social network is available, our framework outperforms recent state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Collective Classification in Network Data (P. Sen et al., 2008)">}}

P. Sen, Galileo Namata, M. Bilgic, L. Getoor, B. Gallagher, Tina Eliassi-Rad. (2008)  
**Collective Classification in Network Data**  
AI Mag.  
[Paper Link](https://www.semanticscholar.org/paper/c5f2f13778af201f486b0b3c4c8f6fcf36d4ca36)  
Influential Citation Count (432), SS-ID (c5f2f13778af201f486b0b3c4c8f6fcf36d4ca36)  

**ABSTRACT**  
Many real-world applications produce networked data such as the world-wide web (hypertext documents connected via hyperlinks), social networks (for example, people connected by friendship links), communication networks (computers connected via communication links) and biological networks (for example, protein interaction networks). A recent focus in machine learning research has been to extend traditional machine learning classification techniques to classify nodes in such networks. In this article, we provide a brief introduction to this area of research and how it has progressed during the past decade. We introduce four of the most widely used inference algorithms for classifying networked data and empirically compare them on both synthetic and real-world data.

{{< /ci-details >}}

{{< ci-details summary="MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding (Xinyu Fu et al., 2020)">}}

Xinyu Fu, Jiani Zhang, Ziqiao Meng, Irwin King. (2020)  
**MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/c7fd29fdd2e0b50a571db4f607eab138e9ecb644)  
Influential Citation Count (42), SS-ID (c7fd29fdd2e0b50a571db4f607eab138e9ecb644)  

**ABSTRACT**  
A large number of real-world graphs or networks are inherently heterogeneous, involving a diversity of node types and relation types. Heterogeneous graph embedding is to embed rich structural and semantic information of a heterogeneous graph into low-dimensional node representations. Existing models usually define multiple metapaths in a heterogeneous graph to capture the composite relations and guide neighbor selection. However, these models either omit node content features, discard intermediate nodes along the metapath, or only consider one metapath. To address these three limitations, we propose a new model named Metapath Aggregated Graph Neural Network (MAGNN) to boost the final performance. Specifically, MAGNN employs three major components, i.e., the node content transformation to encapsulate input node attributes, the intra-metapath aggregation to incorporate intermediate semantic nodes, and the inter-metapath aggregation to combine messages from multiple metapaths. Extensive experiments on three real-world heterogeneous graph datasets for node classification, node clustering, and link prediction show that MAGNN achieves more accurate prediction results than state-of-the-art baselines.

{{< /ci-details >}}

{{< ci-details summary="Learning Community Embedding with Community Detection and Node Embedding on Graphs (Sandro Cavallari et al., 2017)">}}

Sandro Cavallari, V. Zheng, Hongyun Cai, K. Chang, E. Cambria. (2017)  
**Learning Community Embedding with Community Detection and Node Embedding on Graphs**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/c8cee328b1774c2d38bea10f9fe9d081d8074307)  
Influential Citation Count (24), SS-ID (c8cee328b1774c2d38bea10f9fe9d081d8074307)  

**ABSTRACT**  
In this paper, we study an important yet largely under-explored setting of graph embedding, i.e., embedding communities instead of each individual nodes. We find that community embedding is not only useful for community-level applications such as graph visualization, but also beneficial to both community detection and node classification. To learn such embedding, our insight hinges upon a closed loop among community embedding, community detection and node embedding. On the one hand, node embedding can help improve community detection, which outputs good communities for fitting better community embedding. On the other hand, community embedding can be used to optimize the node embedding by introducing a community-aware high-order proximity. Guided by this insight, we propose a novel community embedding framework that jointly solves the three tasks together. We evaluate such a framework on multiple real-world datasets, and show that it improves graph visualization and outperforms state-of-the-art baselines in various application tasks, e.g., community detection and node classification.

{{< /ci-details >}}

{{< ci-details summary="A Deep Learning Approach to Link Prediction in Dynamic Networks (Xiaoyi Li et al., 2014)">}}

Xiaoyi Li, Nan Du, Hui Li, Kang Li, Jing Gao, A. Zhang. (2014)  
**A Deep Learning Approach to Link Prediction in Dynamic Networks**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/c90dd7731b01b3b486b5a28e1ce9bec547c9cfab)  
Influential Citation Count (11), SS-ID (c90dd7731b01b3b486b5a28e1ce9bec547c9cfab)  

**ABSTRACT**  
Time varying problems usually have complex underlying structures represented as dynamic networks where entities and relationships appear and disappear over time. The problem of efficiently performing dynamic link inference is extremely challenging due to the dynamic nature in massive evolving networks especially when there exist sparse connectivities and nonlinear transitional patterns. In this paper, we propose a novel deep learning framework, i.e., Conditional Temporal Restricted Boltzmann Machine (ctRBM), which predicts links based on individual transition variance as well as influence introduced by local neighbors. The proposed model is robust to noise and have the exponential capability to capture nonlinear variance. We tackle the computational challenges by developing an efficient algorithm for learning and inference of the proposed model. To improve the efficiency of the approach, we give a faster approximated implementation based on a proposed Neighbor Influence Clustering algorithm. Extensive experiments on simulated as well as real-world dynamic networks show that the proposed method outperforms existing algorithms in link inference on dynamic networks.

{{< /ci-details >}}

{{< ci-details summary="Graph summarization with bounded error (S. Navlakha et al., 2008)">}}

S. Navlakha, R. Rastogi, Nisheeth Shrivastava. (2008)  
**Graph summarization with bounded error**  
SIGMOD Conference  
[Paper Link](https://www.semanticscholar.org/paper/c948d5342f4ffe8163fc91893a100c9617a2a305)  
Influential Citation Count (49), SS-ID (c948d5342f4ffe8163fc91893a100c9617a2a305)  

**ABSTRACT**  
We propose a highly compact two-part representation of a given graph G consisting of a graph summary and a set of corrections. The graph summary is an aggregate graph in which each node corresponds to a set of nodes in G, and each edge represents the edges between all pair of nodes in the two sets. On the other hand, the corrections portion specifies the list of edge-corrections that should be applied to the summary to recreate G. Our representations allow for both lossless and lossy graph compression with bounds on the introduced error. Further, in combination with the MDL principle, they yield highly intuitive coarse-level summaries of the input graph G. We develop algorithms to construct highly compressed graph representations with small sizes and guaranteed accuracy, and validate our approach through an extensive set of experiments with multiple real-life graph data sets.  To the best of our knowledge, this is the first work to compute graph summaries using the MDL principle, and use the summaries (along with corrections) to compress graphs with bounded error.

{{< /ci-details >}}

{{< ci-details summary="In situ click chemistry generation of cyclooxygenase-2 inhibitors (A. Bhardwaj et al., 2017)">}}

A. Bhardwaj, J. Kaur, M. Wuest, F. Wuest. (2017)  
**In situ click chemistry generation of cyclooxygenase-2 inhibitors**  
Nature Communications  
[Paper Link](https://www.semanticscholar.org/paper/c9938045034626c41dd66ad1b5490bff331c1264)  
Influential Citation Count (1), SS-ID (c9938045034626c41dd66ad1b5490bff331c1264)  

**ABSTRACT**  
Cyclooxygenase-2 isozyme is a promising anti-inflammatory drug target, and overexpression of this enzyme is also associated with several cancers and neurodegenerative diseases. The amino-acid sequence and structural similarity between inducible cyclooxygenase-2 and housekeeping cyclooxygenase-1 isoforms present a significant challenge to design selective cyclooxygenase-2 inhibitors. Herein, we describe the use of the cyclooxygenase-2 active site as a reaction vessel for the in situ generation of its own highly specific inhibitors. Multi-component competitive-binding studies confirmed that the cyclooxygenase-2 isozyme can judiciously select most appropriate chemical building blocks from a pool of chemicals to build its own highly potent inhibitor. Herein, with the use of kinetic target-guided synthesis, also termed as in situ click chemistry, we describe the discovery of two highly potent and selective cyclooxygenase-2 isozyme inhibitors. The in vivo anti-inflammatory activity of these two novel small molecules is significantly higher than that of widely used selective cyclooxygenase-2 inhibitors.Traditional inflammation and pain relief drugs target both cyclooxygenase 1 and 2 (COX-1 and COX-2), causing severe side effects. Here, the authors use in situ click chemistry to develop COX-2 specific inhibitors with high in vivo anti-inflammatory activity.

{{< /ci-details >}}

{{< ci-details summary="Hierarchical graph embedding in vector space by graph pyramid (S. F. Mousavi et al., 2017)">}}

S. F. Mousavi, M. Safayani, A. Mirzaei, Hoda Bahonar. (2017)  
**Hierarchical graph embedding in vector space by graph pyramid**  
Pattern Recognit.  
[Paper Link](https://www.semanticscholar.org/paper/c9f22dde51fb01322212708ef00a61ef580e58bd)  
Influential Citation Count (5), SS-ID (c9f22dde51fb01322212708ef00a61ef580e58bd)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Prediction of drug–target interaction networks from the integration of chemical and genomic spaces (Yoshihiro Yamanishi et al., 2008)">}}

Yoshihiro Yamanishi, M. Araki, Alex Gutteridge, Wataru Honda, M. Kanehisa. (2008)  
**Prediction of drug–target interaction networks from the integration of chemical and genomic spaces**  
ISMB  
[Paper Link](https://www.semanticscholar.org/paper/cb239560296d0dc0bea4ccbcc6a4eefa79b8f100)  
Influential Citation Count (52), SS-ID (cb239560296d0dc0bea4ccbcc6a4eefa79b8f100)  

**ABSTRACT**  
Motivation: The identification of interactions between drugs and target proteins is a key area in genomic drug discovery. Therefore, there is a strong incentive to develop new methods capable of detecting these potential drug–target interactions efficiently. Results: In this article, we characterize four classes of drug–target interaction networks in humans involving enzymes, ion channels, G-protein-coupled receptors (GPCRs) and nuclear receptors, and reveal significant correlations between drug structure similarity, target sequence similarity and the drug–target interaction network topology. We then develop new statistical methods to predict unknown drug–target interaction networks from chemical structure and genomic sequence information simultaneously on a large scale. The originality of the proposed method lies in the formalization of the drug–target interaction inference as a supervised learning problem for a bipartite graph, the lack of need for 3D structure information of the target proteins, and in the integration of chemical and genomic spaces into a unified space that we call ‘pharmacological space’. In the results, we demonstrate the usefulness of our proposed method for the prediction of the four classes of drug–target interaction networks. Our comprehensively predicted drug–target interaction networks enable us to suggest many potential drug–target interactions and to increase research productivity toward genomic drug discovery. Availability: Softwares are available upon request. Contact: Yoshihiro.Yamanishi@ensmp.fr Supplementary information: Datasets and all prediction results are available at http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/.

{{< /ci-details >}}

{{< ci-details summary="Link prediction for interdisciplinary collaboration via co-authorship network (Haeran Cho et al., 2018)">}}

Haeran Cho, Yi Yu. (2018)  
**Link prediction for interdisciplinary collaboration via co-authorship network**  
Social Network Analysis and Mining  
[Paper Link](https://www.semanticscholar.org/paper/cb805fdd5a00e7a22355c4028f8fdec245831b17)  
Influential Citation Count (0), SS-ID (cb805fdd5a00e7a22355c4028f8fdec245831b17)  

**ABSTRACT**  
We analyse the Publication and Research data set of University of Bristol collected between 2008 and 2013. Using the existing co-authorship network and academic information thereof, we propose a new link prediction methodology, with the specific aim of identifying potential interdisciplinary collaboration in a university-wide collaboration network.

{{< /ci-details >}}

{{< ci-details summary="Attention Models in Graphs (J. B. Lee et al., 2018)">}}

J. B. Lee, Ryan A. Rossi, Sungchul Kim, Nesreen Ahmed, Eunyee Koh. (2018)  
**Attention Models in Graphs**  
ACM Trans. Knowl. Discov. Data  
[Paper Link](https://www.semanticscholar.org/paper/cc23c580b7d8063415fb6eb512053d1079b849de)  
Influential Citation Count (1), SS-ID (cc23c580b7d8063415fb6eb512053d1079b849de)  

**ABSTRACT**  
Graph-structured data arise naturally in many different application domains. By representing data as graphs, we can capture entities (i.e., nodes) as well as their relationships (i.e., edges) with each other. Many useful insights can be derived from graph-structured data as demonstrated by an ever-growing body of work focused on graph mining. However, in the real-world, graphs can be both large—with many complex patterns—and noisy, which can pose a problem for effective graph mining. An effective way to deal with this issue is to incorporate “attention” into graph mining solutions. An attention mechanism allows a method to focus on task-relevant parts of the graph, helping it to make better decisions. In this work, we conduct a comprehensive and focused survey of the literature on the emerging field of graph attention models. We introduce three intuitive taxonomies to group existing work. These are based on problem setting (type of input and output), the type of attention mechanism used, and the task (e.g., graph classification, link prediction). We motivate our taxonomies through detailed examples and use each to survey competing approaches from a unique standpoint. Finally, we highlight several challenges in the area and discuss promising directions for future work.

{{< /ci-details >}}

{{< ci-details summary="Fast and Deep Graph Neural Networks (C. Gallicchio et al., 2019)">}}

C. Gallicchio, A. Micheli. (2019)  
**Fast and Deep Graph Neural Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/cc56d1210f4ee7a5a351cf64eb3bbed18b48b22f)  
Influential Citation Count (4), SS-ID (cc56d1210f4ee7a5a351cf64eb3bbed18b48b22f)  

**ABSTRACT**  
We address the efficiency issue for the construction of a deep graph neural network (GNN). The approach exploits the idea of representing each input graph as a fixed point of a dynamical system (implemented through a recurrent neural network), and leverages a deep architectural organization of the recurrent units. Efficiency is gained by many aspects, including the use of small and very sparse networks, where the weights of the recurrent units are left untrained under the stability condition introduced in this work. This can be viewed as a way to study the intrinsic power of the architecture of a deep GNN, and also to provide insights for the set-up of more complex fully-trained models. Through experimental results, we show that even without training of the recurrent connections, the architecture of small deep GNN is surprisingly able to achieve or improve the state-of-the-art performance on a significant set of tasks in the field of graphs classification.

{{< /ci-details >}}

{{< ci-details summary="A Survey on Network Embedding (Peng Cui et al., 2017)">}}

Peng Cui, Xiao Wang, J. Pei, Wenwu Zhu. (2017)  
**A Survey on Network Embedding**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/ce840188f3395815201b7da49f9bb40d24fc046a)  
Influential Citation Count (28), SS-ID (ce840188f3395815201b7da49f9bb40d24fc046a)  

**ABSTRACT**  
Network embedding assigns nodes in a network to low-dimensional representations and effectively preserves the network structure. Recently, a significant amount of progresses have been made toward this emerging network analysis paradigm. In this survey, we focus on categorizing and then reviewing the current development on network embedding methods, and point out its future research directions. We first summarize the motivation of network embedding. We discuss the classical graph embedding algorithms and their relationship with network embedding. Afterwards and primarily, we provide a comprehensive overview of a large number of network embedding methods in a systematic manner, covering the structure- and property-preserving network embedding methods, the network embedding methods with side information, and the advanced information preserving network embedding methods. Moreover, several evaluation approaches for network embedding and some useful online resources, including the network data sets and softwares, are reviewed, too. Finally, we discuss the framework of exploiting these network embedding methods to build an effective system and point out some potential future directions.

{{< /ci-details >}}

{{< ci-details summary="Link prediction in social networks: the state-of-the-art (Peng Wang et al., 2014)">}}

Peng Wang, Baowen Xu, Yu Wu, Xiaoyu Zhou. (2014)  
**Link prediction in social networks: the state-of-the-art**  
Science China Information Sciences  
[Paper Link](https://www.semanticscholar.org/paper/cfe701c64e4ff1d7e5b58a194b02d595b65ce872)  
Influential Citation Count (23), SS-ID (cfe701c64e4ff1d7e5b58a194b02d595b65ce872)  

**ABSTRACT**  
In social networks, link prediction predicts missing links in current networks and new or dissolution links in future networks, is important for mining and analyzing the evolution of social networks. In the past decade, many works have been done about the link prediction in social networks. The goal of this paper is to comprehensively review, analyze and discuss the state-of-the-art of the link prediction in social networks. A systematical category for link prediction techniques and problems is presented. Then link prediction techniques and problems are analyzed and discussed. Typical applications of link prediction are also addressed. Achievements and roadmaps of some active research groups are introduced. Finally, some future challenges of the link prediction in social networks are discussed.创新点对社交网络中的链接预测研究现状进行系统回顾、分析和讨论, 并指出未来研究挑战. 在动态社交网络中, 链接预测是挖掘和分析网络演化的一项重要任务, 其目的是预测当前未知的链接以及未来链接的变化. 过去十余年中, 在社交网络链接预测问题上已有大量研究工作. 本文旨在对该问题的研究现状和趋势进行全面回顾、分析和讨论. 提出一种分类法组织链接预测技术和问题. 详细分析和讨论了链接预测的技术、问题和应用. 介绍了该问题的活跃研究组. 分析和讨论了社交网络链接预测研究的未来挑战.

{{< /ci-details >}}

{{< ci-details summary="Structural Deep Network Embedding (Daixin Wang et al., 2016)">}}

Daixin Wang, Peng Cui, Wenwu Zhu. (2016)  
**Structural Deep Network Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/d0b7c8828f0fca4dd901674e8fb5bd464a187664)  
Influential Citation Count (223), SS-ID (d0b7c8828f0fca4dd901674e8fb5bd464a187664)  

**ABSTRACT**  
Network embedding is an important method to learn low-dimensional representations of vertexes in networks, aiming to capture and preserve the network structure. Almost all the existing network embedding methods adopt shallow models. However, since the underlying network structure is complex, shallow models cannot capture the highly non-linear network structure, resulting in sub-optimal network representations. Therefore, how to find a method that is able to effectively capture the highly non-linear network structure and preserve the global and local structure is an open yet important problem. To solve this problem, in this paper we propose a Structural Deep Network Embedding method, namely SDNE. More specifically, we first propose a semi-supervised deep model, which has multiple layers of non-linear functions, thereby being able to capture the highly non-linear network structure. Then we propose to exploit the first-order and second-order proximity jointly to preserve the network structure. The second-order proximity is used by the unsupervised component to capture the global network structure. While the first-order proximity is used as the supervised information in the supervised component to preserve the local network structure. By jointly optimizing them in the semi-supervised deep model, our method can preserve both the local and global network structure and is robust to sparse networks. Empirically, we conduct the experiments on five real-world networks, including a language network, a citation network and three social networks. The results show that compared to the baselines, our method can reconstruct the original network significantly better and achieves substantial gains in three applications, i.e. multi-label classification, link prediction and visualization.

{{< /ci-details >}}

{{< ci-details summary="Fast Sequence-Based Embedding with Diffusion Graphs (Benedek Rozemberczki et al., 2018)">}}

Benedek Rozemberczki, R. Sarkar. (2018)  
**Fast Sequence-Based Embedding with Diffusion Graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/d1764864dc0a676ff1732972e179fb2d8d4f564b)  
Influential Citation Count (5), SS-ID (d1764864dc0a676ff1732972e179fb2d8d4f564b)  

**ABSTRACT**  
A graph embedding is a representation of graph vertices in a low- dimensional space, which approximately preserves properties such as distances between nodes. Vertex sequence-based embedding procedures use features extracted from linear sequences of nodes to create embeddings using a neural network. In this paper, we propose diffusion graphs as a method to rapidly generate vertex sequences for network embedding. Its computational efficiency is superior to previous methods due to simpler sequence generation, and it produces more accurate results. In experiments, we found that the performance relative to other methods improves with increasing edge density in the graph. In a community detection task, clustering nodes in the embedding space produces better results compared to other sequence-based embedding methods.

{{< /ci-details >}}

{{< ci-details summary="Hierarchical Graph Representation Learning with Differentiable Pooling (Rex Ying et al., 2018)">}}

Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, J. Leskovec. (2018)  
**Hierarchical Graph Representation Learning with Differentiable Pooling**  
NeurIPS  
[Paper Link](https://www.semanticscholar.org/paper/d18b48f77eb5c517a6d2c1fa434d2952a1b0a825)  
Influential Citation Count (183), SS-ID (d18b48f77eb5c517a6d2c1fa434d2952a1b0a825)  

**ABSTRACT**  
Recently, graph neural networks (GNNs) have revolutionized the field of graph representation learning through effectively learned node embeddings, and achieved state-of-the-art results in tasks such as node classification and link prediction. However, current GNN methods are inherently flat and do not learn hierarchical representations of graphs---a limitation that is especially problematic for the task of graph classification, where the goal is to predict the label associated with an entire graph. Here we propose DiffPool, a differentiable graph pooling module that can generate hierarchical representations of graphs and can be combined with various graph neural network architectures in an end-to-end fashion. DiffPool learns a differentiable soft cluster assignment for nodes at each layer of a deep GNN, mapping nodes to a set of clusters, which then form the coarsened input for the next GNN layer. Our experimental results show that combining existing GNN methods with DiffPool yields an average improvement of 5-10% accuracy on graph classification benchmarks, compared to all existing pooling approaches, achieving a new state-of-the-art on four out of five benchmark datasets.

{{< /ci-details >}}

{{< ci-details summary="Using deep neural networks and biological subwords to detect protein S-sulfenylation sites (D. Do et al., 2020)">}}

D. Do, Trang T. Le, N. Le. (2020)  
**Using deep neural networks and biological subwords to detect protein S-sulfenylation sites**  
Briefings Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/d1e34705c7b26b40ba9bd2b1bdb2239b66da9045)  
Influential Citation Count (0), SS-ID (d1e34705c7b26b40ba9bd2b1bdb2239b66da9045)  

**ABSTRACT**  
Protein S-sulfenylation is one kind of crucial post-translational modifications (PTMs) in which the hydroxyl group covalently binds to the thiol of cysteine. Some recent studies have shown that this modification plays an important role in signaling transduction, transcriptional regulation and apoptosis. To date, the dynamic of sulfenic acids in proteins remains unclear because of its fleeting nature. Identifying S-sulfenylation sites, therefore, could be the key to decipher its mysterious structures and functions, which are important in cell biology and diseases. However, due to the lack of effective methods, scientists in this field tend to be limited in merely a handful of some wet lab techniques that are time-consuming and not cost-effective. Thus, this motivated us to develop an in silico model for detecting S-sulfenylation sites only from protein sequence information. In this study, protein sequences served as natural language sentences comprising biological subwords. The deep neural network was consequentially employed to perform classification. The performance statistics within the independent dataset including sensitivity, specificity, accuracy, Matthews correlation coefficient and area under the curve rates achieved 85.71%, 69.47%, 77.09%, 0.5554 and 0.833, respectively. Our results suggested that the proposed method (fastSulf-DNN) achieved excellent performance in predicting S-sulfenylation sites compared to other well-known tools on a benchmark dataset.

{{< /ci-details >}}

{{< ci-details summary="Dynamic Graph Embedding (Sujit Rokka Chhetri et al., 2020)">}}

Sujit Rokka Chhetri, Mohammad Abdullah Al Faruque. (2020)  
**Dynamic Graph Embedding**  
  
[Paper Link](https://www.semanticscholar.org/paper/d28823f812b83ac957ac5077216766cba29d211d)  
Influential Citation Count (0), SS-ID (d28823f812b83ac957ac5077216766cba29d211d)  

**ABSTRACT**  
In Chap. 9, we presented a structural graph convolutional neural network which is capable of performing supervising learning to estimate a function between non-euclidean data and categorical data. In this chapter, we focus on non-euclidean data which are evolving over time. In the cyber-physical system, most of the non-euclidean data (such as engineering data, energy, and signal flow graph, call graph of the firmware, etc.) are always evolving. Hence, it is necessary to utilize algorithms that are capable of handling such temporally evolving non-euclidean data. In this chapter, we present a novel dynamic graph embedding algorithm to handle this issue. In the rest of the chapter, we consider temporally evolving graphs as the non-euclidean data and present an algorithm capable of capturing the pattern of time-varying links.

{{< /ci-details >}}

{{< ci-details summary="Modeling By Shortest Data Description* (J. Rissanen, 1978)">}}

J. Rissanen. (1978)  
**Modeling By Shortest Data Description***  
Autom.  
[Paper Link](https://www.semanticscholar.org/paper/d382b9c11e5c6a8e173fbeb442545e3be8d3e3a5)  
Influential Citation Count (402), SS-ID (d382b9c11e5c6a8e173fbeb442545e3be8d3e3a5)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Differentiating Concepts and Instances for Knowledge Graph Embedding (Xin Lv et al., 2018)">}}

Xin Lv, Lei Hou, Juan-Zi Li, Zhiyuan Liu. (2018)  
**Differentiating Concepts and Instances for Knowledge Graph Embedding**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/d3c287ff061f295ddf8dc3cb02a6f39e301cae3b)  
Influential Citation Count (8), SS-ID (d3c287ff061f295ddf8dc3cb02a6f39e301cae3b)  

**ABSTRACT**  
Concepts, which represent a group of different instances sharing common properties, are essential information in knowledge representation. Most conventional knowledge embedding methods encode both entities (concepts and instances) and relations as vectors in a low dimensional semantic space equally, ignoring the difference between concepts and instances. In this paper, we propose a novel knowledge graph embedding model named TransC by differentiating concepts and instances. Specifically, TransC encodes each concept in knowledge graph as a sphere and each instance as a vector in the same semantic space. We use the relative positions to model the relations between concepts and instances (i.e.,instanceOf), and the relations between concepts and sub-concepts (i.e., subClassOf). We evaluate our model on both link prediction and triple classification tasks on the dataset based on YAGO. Experimental results show that TransC outperforms state-of-the-art methods, and captures the semantic transitivity for instanceOf and subClassOf relation. Our codes and datasets can be obtained from https://github.com/davidlvxin/TransC.

{{< /ci-details >}}

{{< ci-details summary="Community Preserving Network Embedding (Xiao Wang et al., 2017)">}}

Xiao Wang, Peng Cui, Jing Wang, J. Pei, Wenwu Zhu, Shiqiang Yang. (2017)  
**Community Preserving Network Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/d3e0d596efd9d19b93d357565a68dfa925dce2bb)  
Influential Citation Count (52), SS-ID (d3e0d596efd9d19b93d357565a68dfa925dce2bb)  

**ABSTRACT**  
Network embedding, aiming to learn the low-dimensional representations of nodes in networks, is of paramount importance in many real applications. One basic requirement of network embedding is to preserve the structure and inherent properties of the networks. While previous network embedding methods primarily preserve the microscopic structure, such as the first- and second-order proximities of nodes, the mesoscopic community structure, which is one of the most prominent feature of networks, is largely ignored. In this paper, we propose a novel Modularized Nonnegative Matrix Factorization (M-NMF) model to incorporate the community structure into network embedding. We exploit the consensus relationship between the representations of nodes and community structure, and then jointly optimize NMF based representation learning model and modularity based community detection model in a unified framework, which enables the learned representations of nodes to preserve both of the microscopic and community structures. We also provide efficient updating rules to infer the parameters of our model, together with the correctness and convergence guarantees. Extensive experimental results on a variety of real-world networks show the superior performance of the proposed method over the state-of-the-arts.

{{< /ci-details >}}

{{< ci-details summary="Link Prediction Methods and Their Accuracy for Different Social Networks and Network Metrics (F. Gao et al., 2015)">}}

F. Gao, Katarzyna Musial, C. Cooper, S. Tsoka. (2015)  
**Link Prediction Methods and Their Accuracy for Different Social Networks and Network Metrics**  
Sci. Program.  
[Paper Link](https://www.semanticscholar.org/paper/d47e5c2dfb5dcd58e8d0f513807e5671e4607a35)  
Influential Citation Count (2), SS-ID (d47e5c2dfb5dcd58e8d0f513807e5671e4607a35)  

**ABSTRACT**  
Currently, we are experiencing a rapid growth of the number of social-based online systems. The availability of the vast amounts of data gathered in those systems brings new challenges that we face when trying to analyse it. One of the intensively researched topics is the prediction of social connections between users. Although a lot of effort has been made to develop new prediction approaches, the existing methods are not comprehensively analysed. In this paper we investigate the correlation between network metrics and accuracy of different prediction methods. We selected six time-stamped real-world social networks and ten most widely used link prediction methods. The results of the experiments show that the performance of some methods has a strong correlation with certain network metrics. We managed to distinguish "prediction friendly" networks, for which most of the prediction methods give good performance, as well as "prediction unfriendly" networks, for which most of the methods result in high prediction error. Correlation analysis between network metrics and prediction accuracy of prediction methods may form the basis of a metalearning system where based on network characteristics it will be able to recommend the right prediction method for a given network.

{{< /ci-details >}}

{{< ci-details summary="PCA versus LDA (Aleix M. Martinez et al., 2001)">}}

Aleix M. Martinez, A. Kak. (2001)  
**PCA versus LDA**  
IEEE Trans. Pattern Anal. Mach. Intell.  
[Paper Link](https://www.semanticscholar.org/paper/d544475dc01daa0c4f9847ef72adb8878df8ce99)  
Influential Citation Count (241), SS-ID (d544475dc01daa0c4f9847ef72adb8878df8ce99)  

**ABSTRACT**  
In the context of the appearance-based paradigm for object recognition, it is generally believed that algorithms based on LDA (linear discriminant analysis) are superior to those based on PCA (principal components analysis). In this communication, we show that this is not always the case. We present our case first by using intuitively plausible arguments and, then, by showing actual results on a face database. Our overall conclusion is that when the training data set is small, PCA can outperform LDA and, also, that PCA is less sensitive to different training data sets.

{{< /ci-details >}}

{{< ci-details summary="GraphSAINT: Graph Sampling Based Inductive Learning Method (Hanqing Zeng et al., 2019)">}}

Hanqing Zeng, Hongkuan Zhou, Ajitesh Srivastava, R. Kannan, V. Prasanna. (2019)  
**GraphSAINT: Graph Sampling Based Inductive Learning Method**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/d589e4018278e219733b156d44d0ba881a32195e)  
Influential Citation Count (85), SS-ID (d589e4018278e219733b156d44d0ba881a32195e)  

**ABSTRACT**  
Graph Convolutional Networks (GCNs) are powerful models for learning representations of attributed this http URL scale GCNs to large graphs, state-of-the-art methods use various layer sampling techniques to alleviate the "neighbor explosion" problem during minibatch training. Here we proposeGraphSAINT, a graph sampling based inductive learning method that improves training efficiency in a fundamentally different way. By a change of perspective, GraphSAINT constructs minibatches by sampling the training graph, rather than the nodes or edges across GCN layers. Each iteration, a complete GCN is built from the properly sampled subgraph. Thus, we ensure fixed number of well-connected nodes in all layers. We further propose normalization technique to eliminate bias, and sampling algorithms for variance reduction. Importantly, we can decouple the sampling process from the forward and backward propagation of training, and extend GraphSAINT with other graph samplers and GCN variants. Comparing with strong baselines using layer sampling, GraphSAINT demonstrates superior performance in both accuracy and training time on four large graphs.

{{< /ci-details >}}

{{< ci-details summary="Collective dynamics of ‘small-world’ networks (D. Watts et al., 1998)">}}

D. Watts, S. Strogatz. (1998)  
**Collective dynamics of ‘small-world’ networks**  
Nature  
[Paper Link](https://www.semanticscholar.org/paper/d61031326150ba23f90e6587c13d99188209250e)  
Influential Citation Count (2067), SS-ID (d61031326150ba23f90e6587c13d99188209250e)  

**ABSTRACT**  
Networks of coupled dynamical systems have been used to model biological oscillators, Josephson junction arrays,, excitable media, neural networks, spatial games, genetic control networks and many other self-organizing systems. Ordinarily, the connection topology is assumed to be either completely regular or completely random. But many biological, technological and social networks lie somewhere between these two extremes. Here we explore simple models of networks that can be tuned through this middle ground: regular networks ‘rewired’ to introduce increasing amounts of disorder. We find that these systems can be highly clustered, like regular lattices, yet have small characteristic path lengths, like random graphs. We call them ‘small-world’ networks, by analogy with the small-world phenomenon, (popularly known as six degrees of separation). The neural network of the worm Caenorhabditis elegans, the power grid of the western United States, and the collaboration graph of film actors are shown to be small-world networks. Models of dynamical systems with small-world coupling display enhanced signal-propagation speed, computational power, and synchronizability. In particular, infectious diseases spread more easily in small-world networks than in regular lattices.

{{< /ci-details >}}

{{< ci-details summary="Knowledge Graph Completion with Adaptive Sparse Transfer Matrix (Guoliang Ji et al., 2016)">}}

Guoliang Ji, Kang Liu, Shizhu He, Jun Zhao. (2016)  
**Knowledge Graph Completion with Adaptive Sparse Transfer Matrix**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/d6e5c0cabb07081e750d6426b649978584918216)  
Influential Citation Count (23), SS-ID (d6e5c0cabb07081e750d6426b649978584918216)  

**ABSTRACT**  
We model knowledge graphs for their completion by encoding each entity and relation into a numerical space. All previous work including Trans(E, H, R, and D) ignore the heterogeneity (some relations link many entity pairs and others do not) and the imbalance (the number of head entities and that of tail entities in a relation could be different) of knowledge graphs. In this paper, we propose a novel approach TranSparse to deal with the two issues. In TranSparse, transfer matrices are replaced by adaptive sparse matrices, whose sparse degrees are determined by the number of entities (or entity pairs) linked by relations. In experiments, we design structured and unstructured sparse patterns for transfer matrices and analyze their advantages and disadvantages. We evaluate our approach on triplet classification and link prediction tasks. Experimental results show that TranSparse outperforms Trans(E, H, R, and D) significantly, and achieves state-of-the-art performance.

{{< /ci-details >}}

{{< ci-details summary="Uniform Pooling for Graph Networks (Jian Qin et al., 2020)">}}

Jian Qin, Li Liu, Hui Shen, D. Hu. (2020)  
**Uniform Pooling for Graph Networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/d83c64f96d9a904280534bc3e0c5bd702aab5d94)  
Influential Citation Count (0), SS-ID (d83c64f96d9a904280534bc3e0c5bd702aab5d94)  

**ABSTRACT**  
The graph convolution network has received a lot of attention because it extends the convolution to non-Euclidean domains. However, the graph pooling method is still less concerned, which can learn coarse graph embedding to facilitate graph classification. Previous pooling methods were based on assigning a score to each node and then pooling only the highest-scoring nodes, which might throw away whole neighbourhoods of nodes and therefore information. Here, we proposed a novel pooling method UGPool with a new point-of-view on selecting nodes. UGPool learns node scores based on node features and uniformly pools neighboring nodes instead of top nodes in the score-space, resulting in a uniformly coarsened graph. In multiple graph classification tasks, including the protein graphs, the biological graphs and the brain connectivity graphs, we demonstrated that UGPool outperforms other graph pooling methods while maintaining high efficiency. Moreover, we also show that UGPool can be integrated with multiple graph convolution networks to effectively improve performance compared to no pooling.

{{< /ci-details >}}

{{< ci-details summary="Graph Embedding For Link Prediction Using Residual Variational Graph Autoencoders (Reyhan Kevser Keser et al., 2020)">}}

Reyhan Kevser Keser, Indrit Nallbani, Nurullah Çalik, Aydin Ayanzadeh, B. Töreyin. (2020)  
**Graph Embedding For Link Prediction Using Residual Variational Graph Autoencoders**  
2020 28th Signal Processing and Communications Applications Conference (SIU)  
[Paper Link](https://www.semanticscholar.org/paper/d94a322106c9161813360d8cfd108ec95e9fad67)  
Influential Citation Count (0), SS-ID (d94a322106c9161813360d8cfd108ec95e9fad67)  

**ABSTRACT**  
Graphs are usually represented by high dimensional data. Hence, graph embedding is an essential task, which aims to represent a graph in a lower dimension while protecting the original graph's properties. In this paper, we propose a novel graph embedding method called Residual Variational Graph Autoencoder (RVGAE), which boosts variational graph autoencoder's performance utilizing residual connections. Our method's performance is evaluated on the link prediction task. The results demonstrate that our model can achieve better results than graph convolutional neural network (GCN) and variational graph autoencoder (VGAE).

{{< /ci-details >}}

{{< ci-details summary="Aligning Users across Social Networks Using Network Embedding (Li Liu et al., 2016)">}}

Li Liu, W. K. Cheung, Xin Li, L. Liao. (2016)  
**Aligning Users across Social Networks Using Network Embedding**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/da7ee47ee1ccee8080f5827c3c8ee60af90e5fa0)  
Influential Citation Count (36), SS-ID (da7ee47ee1ccee8080f5827c3c8ee60af90e5fa0)  

**ABSTRACT**  
In this paper, we adopt the representation learning approach to align users across multiple social networks where the social structures of the users are exploited. In particular, we propose to learn a network embedding with the followership/ followee-ship of each user explicitly modeled as input/output context vector representations so as to preserve the proximity of users with "similar" followers/followees in the embedded space. For the alignment, we add both known and potential anchor users across the networks to facilitate the transfer of context information across networks. We solve both the network embedding problem and the user alignment problem simultaneously under a unified optimization framework. The stochastic gradient descent and negative sampling algorithms are used to address scalability issues. Extensive experiments on real social network datasets demonstrate the effectiveness and efficiency of the proposed approach compared with several state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="A Unified Feature Selection Framework for Graph Embedding on High Dimensional Data (Marcus Chen et al., 2015)">}}

Marcus Chen, I. Tsang, Mingkui Tan, T. Cham. (2015)  
**A Unified Feature Selection Framework for Graph Embedding on High Dimensional Data**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/dab795b562c7cc270c9099b925d685bea0abe82a)  
Influential Citation Count (3), SS-ID (dab795b562c7cc270c9099b925d685bea0abe82a)  

**ABSTRACT**  
Although graph embedding has been a powerful tool for modeling data intrinsic structures, simply employing all features for data structure discovery may result in noise amplification. This is particularly severe for high dimensional data with small samples. To meet this challenge, this paper proposes a novel efficient framework to perform feature selection for graph embedding, in which a category of graph embedding methods is cast as a least squares regression problem. In this framework, a binary feature selector is introduced to naturally handle the feature cardinality in the least squares formulation. The resultant integral programming problem is then relaxed into a convex Quadratically Constrained Quadratic Program (QCQP) learning problem, which can be efficiently solved via a sequence of accelerated proximal gradient (APG) methods. Since each APG optimization is w.r.t. only a subset of features, the proposed method is fast and memory efficient. The proposed framework is applied to several graph embedding learning problems, including supervised, unsupervised, and semi-supervised graph embedding. Experimental results on several high dimensional data demonstrated that the proposed method outperformed the considered state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="The Applications of Stochastic Models in Network Embedding: A Survey (Minglong Lei et al., 2018)">}}

Minglong Lei, Yong Shi, Lingfeng Niu. (2018)  
**The Applications of Stochastic Models in Network Embedding: A Survey**  
2018 IEEE/WIC/ACM International Conference on Web Intelligence (WI)  
[Paper Link](https://www.semanticscholar.org/paper/dbc0c4e613d2ae942694058861dd2a4aae6ff117)  
Influential Citation Count (0), SS-ID (dbc0c4e613d2ae942694058861dd2a4aae6ff117)  

**ABSTRACT**  
Network embedding is a promising topic that maps the vertices to the latent space while keeps the structural proximity in the original space. The network embedding task is difficult since the network vertices have no specific time or space orders. Models that used to extract information from images and texts with regular space or time structures can not be directly applied in network heading. The key feature of network embedding methods should be further exploited. Previous network embedding reviews mainly focus on the models and algorithms used in different methods. In this survey, we review the network embedding works in the stochastic perspective either in data side or model side. Roughly, the network embedding methods fall into three main categories: matrix based methods, random walk based methods and aggregated based methods. We focus on the applications of stochastic models in solving the challenges of network embedding in data processing and modeling following the line of the three categories.

{{< /ci-details >}}

{{< ci-details summary="Global Vectors for Node Representations (Robin Brochier et al., 2019)">}}

Robin Brochier, Adrien Guille, Julien Velcin. (2019)  
**Global Vectors for Node Representations**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/dbdc3e34b5a476d9e760882b7779690b4a987d27)  
Influential Citation Count (1), SS-ID (dbdc3e34b5a476d9e760882b7779690b4a987d27)  

**ABSTRACT**  
Most network embedding algorithms consist in measuring co-occur-rences of nodes via random walks then learning the embeddings using Skip-Gram with Negative Sampling. While it has proven to be a relevant choice, there are alternatives, such as GloVe, which has not been investigated yet for network embedding. Even though SGNS better handles non co-occurrence than GloVe, it has a worse time-complexity. In this paper, we propose a matrix factorization approach for network embedding, inspired by GloVe, that better handles non co-occurrence with a competitive time-complexity. We also show how to extend this model to deal with networks where nodes are documents, by simultaneously learning word, node and document representations. Quantitative evaluations show that our model achieves state-of-the-art performance, while not being so sensitive to the choice of hyper-parameters. Qualitatively speaking, we show how our model helps exploring a network of documents by generating complementary network-oriented and content-oriented keywords.

{{< /ci-details >}}

{{< ci-details summary="Learning Deep Network Representations with Adversarially Regularized Autoencoders (Wenchao Yu et al., 2018)">}}

Wenchao Yu, Cheng Zheng, Wei Cheng, C. Aggarwal, Dongjin Song, Bo Zong, Haifeng Chen, Wei Wang. (2018)  
**Learning Deep Network Representations with Adversarially Regularized Autoencoders**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/dbf125bfe07856a63d4aab612c0063fc8c7b6484)  
Influential Citation Count (6), SS-ID (dbf125bfe07856a63d4aab612c0063fc8c7b6484)  

**ABSTRACT**  
The problem of network representation learning, also known as network embedding, arises in many machine learning tasks assuming that there exist a small number of variabilities in the vertex representations which can capture the "semantics" of the original network structure. Most existing network embedding models, with shallow or deep architectures, learn vertex representations from the sampled vertex sequences such that the low-dimensional embeddings preserve the locality property and/or global reconstruction capability. The resultant representations, however, are difficult for model generalization due to the intrinsic sparsity of sampled sequences from the input network. As such, an ideal approach to address the problem is to generate vertex representations by learning a probability density function over the sampled sequences. However, in many cases, such a distribution in a low-dimensional manifold may not always have an analytic form. In this study, we propose to learn the network representations with adversarially regularized autoencoders (NetRA). NetRA learns smoothly regularized vertex representations that well capture the network structure through jointly considering both locality-preserving and global reconstruction constraints. The joint inference is encapsulated in a generative adversarial training process to circumvent the requirement of an explicit prior distribution, and thus obtains better generalization performance. We demonstrate empirically how well key properties of the network structure are captured and the effectiveness of NetRA on a variety of tasks, including network reconstruction, link prediction, and multi-label classification.

{{< /ci-details >}}

{{< ci-details summary="Deep Learning Approaches for Link Prediction in Social Network Services (Feng Liu et al., 2013)">}}

Feng Liu, Bingquan Liu, Chengjie Sun, Ming Liu, Xiaolong Wang. (2013)  
**Deep Learning Approaches for Link Prediction in Social Network Services**  
ICONIP  
[Paper Link](https://www.semanticscholar.org/paper/dc995128c156b587d9b627e89d413563cd1e05df)  
Influential Citation Count (3), SS-ID (dc995128c156b587d9b627e89d413563cd1e05df)  

**ABSTRACT**  
With the fast development of online Social Network ServicesSNS, social members get large amounts of interactions which can be presented as links with values. The link prediction problem is to estimate the values of unknown links by the known links' information. In this paper, based on deep learning approaches, methods for link prediction are proposed. Firstly, an unsupervised method that can works well with little samples is introduced. Secondly, we propose a feature representation method, and the represented features perform better than original ones for link prediction. Thirdly, based on Restricted Boltzmann Machine RBM that present the joint distribution of link samples and their values, we propose a method for link prediction. By the experiments' results, our method can predict links' values with high accuracy for data from SNS websites.

{{< /ci-details >}}

{{< ci-details summary="Hyperbolic embedding of internet graph for distance estimation and overlay construction (Y. Shavitt et al., 2008)">}}

Y. Shavitt, Tomer Tankel. (2008)  
**Hyperbolic embedding of internet graph for distance estimation and overlay construction**  
TNET  
[Paper Link](https://www.semanticscholar.org/paper/dd1d4e8acfabf225686d294660e0deb0059bdfd7)  
Influential Citation Count (8), SS-ID (dd1d4e8acfabf225686d294660e0deb0059bdfd7)  

**ABSTRACT**  
Estimating distances in the Internet has been studied in the recent years due to its ability to improve the performance of many applications, e.g., in the peer-to-peer realm. One scalable approach to estimate distances between nodes is to embed the nodes in some d dimensional geometric space and to use the pair distances in this space as the estimate for the real distances. Several algorithms were suggested in the past to do this in low dimensional Euclidean spaces.  It was noted in recent years that the Internet structure has a highly connected core and long stretched tendrils, and that most of the routing paths between nodes in the tendrils pass through the core. Therefore, we suggest in this work, to embed the Internet distance metric in a hyperbolic space where routes are bent toward the center. We found that if the curvature, that defines the extend of the bending, is selected in the adequate range, the accuracy of Internet distance embedding can be improved.  We demonstrate the strength of our hyperbolic embedding with two applications: selecting the closest server and building an application level multicast tree. For the latter, we present a distributed algorithm for building geometric multicast trees that achieve good trade-offs between delay (stretch) and load (stress). We also present a new efficient centralized embedding algorithm that enables the accurate embedding of short distances, something that have never been done before.

{{< /ci-details >}}

{{< ci-details summary="Language Modeling with Graph Temporal Convolutional Networks (Hongyin Luo et al., 2018)">}}

Hongyin Luo, Yichen Li, Jie Fu, James R. Glass. (2018)  
**Language Modeling with Graph Temporal Convolutional Networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/de23b3889d121102e463853269ec0bfa7cf4332f)  
Influential Citation Count (0), SS-ID (de23b3889d121102e463853269ec0bfa7cf4332f)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Spectral Graph Wavelets for Structural Role Similarity in Networks (C. Donnat et al., 2017)">}}

C. Donnat, M. Zitnik, David Hallac, J. Leskovec. (2017)  
**Spectral Graph Wavelets for Structural Role Similarity in Networks**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/ded841318dbc807c47608a697629fdc4fa3f01da)  
Influential Citation Count (9), SS-ID (ded841318dbc807c47608a697629fdc4fa3f01da)  

**ABSTRACT**  
Nodes residing in different parts of a graph can have similar structural roles within their local network topology. The identification of such roles provides key insight into the organization of networks and can also be used to inform machine learning on graphs. However, learning structural representations of nodes is a challenging unsupervised-learning task, which typically involves manually specifying and tailoring topological features for each node. Here we develop GRAPHWAVE, a method that represents each node’s local network neighborhood via a low-dimensional embedding by leveraging spectral graph wavelet diffusion patterns. We prove that nodes with similar local network neighborhoods will have similar GRAPHWAVE embeddings even though these nodes may reside in very different parts of the network. Our method scales linearly with the number of edges and does not require any hand-tailoring of topological features. We evaluate performance on both synthetic and real-world datasets, obtaining improvements of up to 71% over state-of-the-art baselines.

{{< /ci-details >}}

{{< ci-details summary="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Jacob Devlin et al., 2019)">}}

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019)  
**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
NAACL  
[Paper Link](https://www.semanticscholar.org/paper/df2b0e26d0599ce3e70df8a9da02e51594e0e992)  
Influential Citation Count (9863), SS-ID (df2b0e26d0599ce3e70df8a9da02e51594e0e992)  

**ABSTRACT**  
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5 (7.7 point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

{{< /ci-details >}}

{{< ci-details summary="Structure preserving embedding (B. Shaw et al., 2009)">}}

B. Shaw, T. Jebara. (2009)  
**Structure preserving embedding**  
ICML '09  
[Paper Link](https://www.semanticscholar.org/paper/df30fe0aeac5a530c9499598251a3854fe45ee94)  
Influential Citation Count (4), SS-ID (df30fe0aeac5a530c9499598251a3854fe45ee94)  

**ABSTRACT**  
Structure Preserving Embedding (SPE) is an algorithm for embedding graphs in Euclidean space such that the embedding is low-dimensional and preserves the global topological properties of the input graph. Topology is preserved if a connectivity algorithm, such as k-nearest neighbors, can easily recover the edges of the input graph from only the coordinates of the nodes after embedding. SPE is formulated as a semidefinite program that learns a low-rank kernel matrix constrained by a set of linear inequalities which captures the connectivity structure of the input graph. Traditional graph embedding algorithms do not preserve structure according to our definition, and thus the resulting visualizations can be misleading or less informative. SPE provides significant improvements in terms of visualization and lossless compression of graphs, outperforming popular methods such as spectral embedding and Laplacian eigen-maps. We find that many classical graphs and networks can be properly embedded using only a few dimensions. Furthermore, introducing structure preserving constraints into dimensionality reduction algorithms produces more accurate representations of high-dimensional data.

{{< /ci-details >}}

{{< ci-details summary="Learning Deep Representations for Graph Clustering (Fei Tian et al., 2014)">}}

Fei Tian, Bin Gao, Qing Cui, Enhong Chen, Tie-Yan Liu. (2014)  
**Learning Deep Representations for Graph Clustering**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/df787a974fff59f557ed1ec620fc345568aec491)  
Influential Citation Count (39), SS-ID (df787a974fff59f557ed1ec620fc345568aec491)  

**ABSTRACT**  
Recently deep learning has been successfully adopted in many applications such as speech recognition and image classification. In this work, we explore the possibility of employing deep learning in graph clustering. We propose a simple method, which first learns a nonlinear embedding of the original graph by stacked autoencoder, and then runs k-means algorithm on the embedding to obtain clustering result. We show that this simple method has solid theoretical foundation, due to the similarity between autoencoder and spectral clustering in terms of what they actually optimize. Then, we demonstrate that the proposed method is more efficient and flexible than spectral clustering. First, the computational complexity of autoencoder is much lower than spectral clustering: the former can be linear to the number of nodes in a sparse graph while the latter is super quadratic due to eigenvalue decomposition. Second, when additional sparsity constraint is imposed, we can simply employ the sparse autoencoder developed in the literature of deep learning; however, it is nonstraightforward to implement a sparse spectral method. The experimental results on various graph datasets show that the proposed method significantly outperforms conventional spectral clustering, which clearly indicates the effectiveness of deep learning in graph clustering.

{{< /ci-details >}}

{{< ci-details summary="subgraph2vec: Learning Distributed Representations of Rooted Sub-graphs from Large Graphs (A. Narayanan et al., 2016)">}}

A. Narayanan, Mahinthan Chandramohan, Lihui Chen, Yang Liu, S. Saminathan. (2016)  
**subgraph2vec: Learning Distributed Representations of Rooted Sub-graphs from Large Graphs**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/e02f59cf876cb40233573ff78a1609f969d301cc)  
Influential Citation Count (11), SS-ID (e02f59cf876cb40233573ff78a1609f969d301cc)  

**ABSTRACT**  
In this paper, we present subgraph2vec, a novel approach for learning latent representations of rooted subgraphs from large graphs inspired by recent advancements in Deep Learning and Graph Kernels. These latent representations encode semantic substructure dependencies in a continuous vector space, which is easily exploited by statistical models for tasks such as graph classification, clustering, link prediction and community detection. subgraph2vec leverages on local information obtained from neighbourhoods of nodes to learn their latent representations in an unsupervised fashion. We demonstrate that subgraph vectors learnt by our approach could be used in conjunction with classifiers such as CNNs, SVMs and relational data clustering algorithms to achieve significantly superior accuracies. Also, we show that the subgraph vectors could be used for building a deep learning variant of Weisfeiler-Lehman graph kernel. Our experiments on several benchmark and large-scale real-world datasets reveal that subgraph2vec achieves significant improvements in accuracies over existing graph kernels on both supervised and unsupervised learning tasks. Specifically, on two realworld program analysis tasks, namely, code clone and malware detection, subgraph2vec outperforms state-of-the-art kernels by more than 17% and 4%, respectively.

{{< /ci-details >}}

{{< ci-details summary="What about statistical relational learning? (CACM staff, 2015)">}}

CACM staff. (2015)  
**What about statistical relational learning?**  
Commun. ACM  
[Paper Link](https://www.semanticscholar.org/paper/e106966bbf727b7c705288c9a7ef8155bb8a67ab)  
Influential Citation Count (7), SS-ID (e106966bbf727b7c705288c9a7ef8155bb8a67ab)  

**ABSTRACT**  
letters to the editor the Halting Problem, as well as general program synthesis. I am also accused of ignoring " the entire field of statistical relational learning. " The excellent book Introduction to Statistical Relational Learning, compiled by my former student Lise Getoor and the late Ben Taskar (MIT Press, 2007), has 13 chapters on SRL languages and systems. My article referred to 10 of them. My comment that IBAL was the first probabilistic programming language (PPL) was in no way intended as a slight to Sato's PRISM and Poole's ICL, contributions for which I have the highest respect. My article placed these approaches, along with BLOG, within the tradition of languages for defining probability distributions over logical worlds, as did Sato and Poole. For example, the PRISM website (http://rjida.meijo-u.ac.jp/prism/) says, " The program defines a probability distribution over the set of possible Herbrand interpretations. " My article clearly distinguished this approach from the PPL tradition based on distributions over execution traces of an arbitrary programming language, due to Koller, McAllester, and Pfeffer. Perhaps this is just a matter of terminology, although it seems worthwhile to point out that execution traces need have no relational structure at all. At the time of writing, the extensive bibliography at http://probabilistic-programming.org/research/ does not include the early papers by Sato and Poole, but a broader notion of PPL might well include them. Give Me 'Naked' Braces I was appalled by A. Frank Ackerman's letter to the editor " Ban 'Naked' Braces! " (Oct. 2015), which recommended programmers adopt a policy of always following the closing brace of each code block (presumably, in Algol-like languages like C and Java) with a comment intended to make it clear exactly which code block the closing brace belongs to. However, review article " Unifying Logic and Probability " (July 2015) provided an excellent summary of a number of attempts to unify these two representations, it also gave an incomplete picture of the state of the art. The entire field of statistical relational learning (SRL), which was never mentioned in the article, is devoted to learning logical probabilistic models. Although the article said little is known about com-putationally feasible algorithms for learning the structure of these models , SRL researchers have developed a wide variety of them. Likewise, contrary to the article's statement that generic inference for logical probabi-listic models remains too slow, many efficient algorithms for this purpose …

{{< /ci-details >}}

{{< ci-details summary="Neural Message Passing for Quantum Chemistry (J. Gilmer et al., 2017)">}}

J. Gilmer, S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl. (2017)  
**Neural Message Passing for Quantum Chemistry**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/e24cdf73b3e7e590c2fe5ecac9ae8aa983801367)  
Influential Citation Count (440), SS-ID (e24cdf73b3e7e590c2fe5ecac9ae8aa983801367)  

**ABSTRACT**  
Supervised learning on molecules has incredible potential to be useful in chemistry, drug discovery, and materials science. Luckily, several promising and closely related neural network models invariant to molecular symmetries have already been described in the literature. These models learn a message passing algorithm and aggregation procedure to compute a function of their entire input graph. At this point, the next step is to find a particularly effective variant of this general approach and apply it to chemical prediction benchmarks until we either solve them or reach the limits of the approach. In this paper, we reformulate existing models into a single common framework we call Message Passing Neural Networks (MPNNs) and explore additional novel variations within this framework. Using MPNNs we demonstrate state of the art results on an important molecular property prediction benchmark; these results are strong enough that we believe future work should focus on datasets with larger molecules or more accurate ground truth labels.

{{< /ci-details >}}

{{< ci-details summary="Structure2vec: Deep Learning for Security Analytics over Graphs (Le Song, 2018)">}}

Le Song. (2018)  
**Structure2vec: Deep Learning for Security Analytics over Graphs**  
  
[Paper Link](https://www.semanticscholar.org/paper/e3dfc4fc64b2e056359a6482930c04cea6c9570a)  
Influential Citation Count (0), SS-ID (e3dfc4fc64b2e056359a6482930c04cea6c9570a)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Attention Guided Graph Convolutional Networks for Relation Extraction (Zhijiang Guo et al., 2019)">}}

Zhijiang Guo, Yan Zhang, Wei Lu. (2019)  
**Attention Guided Graph Convolutional Networks for Relation Extraction**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/e4363d077a890c8d5c5e66b82fe69a1bbbdd5c80)  
Influential Citation Count (45), SS-ID (e4363d077a890c8d5c5e66b82fe69a1bbbdd5c80)  

**ABSTRACT**  
Dependency trees convey rich structural information that is proven useful for extracting relations among entities in text. However, how to effectively make use of relevant information while ignoring irrelevant information from the dependency trees remains a challenging research question. Existing approaches employing rule based hard-pruning strategies for selecting relevant partial dependency structures may not always yield optimal results. In this work, we propose Attention Guided Graph Convolutional Networks (AGGCNs), a novel model which directly takes full dependency trees as inputs. Our model can be understood as a soft-pruning approach that automatically learns how to selectively attend to the relevant sub-structures useful for the relation extraction task. Extensive results on various tasks including cross-sentence n-ary relation extraction and large-scale sentence-level relation extraction show that our model is able to better leverage the structural information of the full dependency trees, giving significantly better results than previous approaches.

{{< /ci-details >}}

{{< ci-details summary="Principled Multilayer Network Embedding (Weiyi Liu et al., 2017)">}}

Weiyi Liu, Pin-Yu Chen, S. Yeung, T. Suzumura, Lingli Chen. (2017)  
**Principled Multilayer Network Embedding**  
2017 IEEE International Conference on Data Mining Workshops (ICDMW)  
[Paper Link](https://www.semanticscholar.org/paper/e445ce942f2cd572aed76160febe35973e0fc42f)  
Influential Citation Count (14), SS-ID (e445ce942f2cd572aed76160febe35973e0fc42f)  

**ABSTRACT**  
Multilayer network analysis has become a vital tool for understanding different relationships and their interactions in a complex system, where each layer in a multilayer network depicts the topological structure of a group of nodes corresponding to a particular relationship. The interactions among different layers imply how the interplay of different relations on the topology of each layer. For a single-layer network, network embedding methods have been proposed to project the nodes in a network into a continuous vector space with a relatively small number of dimensions, where the space embeds the social representations among nodes. These algorithms have been proved to have a better performance on a variety of regular graph analysis tasks, such as link prediction, or multi-label classification. In this paper, by extending a standard graph mining into multilayer network, we have proposed three methods ("network aggregation," "results aggregation" and "layer co-analysis") to project a multilayer network into a continuous vector space. On one hand, without leveraging interactions among layers, "network aggregation" and "results aggregation" apply the standard network embedding method on the merged graph or each layer to find a vector space for multilayer network. On the other hand, in order to consider the influence of interactions among layers, "layer co-analysis" expands any single-layer network embedding method to a multilayer network. By introducing the link transition probability based on information distance, this method not only uses the first and second order random walk to traverse on a layer, but also has the ability to traverse between layers by leveraging interactions. From the evaluation, we have proved that comparing with regular link prediction methods, "layer co-analysis" achieved the best performance on most of the datasets, while "network aggregation" and "results aggregation" also have better performance than regular link prediction methods.

{{< /ci-details >}}

{{< ci-details summary="Embedding Networks with Edge Attributes (Palash Goyal et al., 2018)">}}

Palash Goyal, Homa Hosseinmardi, Emilio Ferrara, A. Galstyan. (2018)  
**Embedding Networks with Edge Attributes**  
HT  
[Paper Link](https://www.semanticscholar.org/paper/e4853de6d86315073a9e9e5d8957500cd24402c1)  
Influential Citation Count (0), SS-ID (e4853de6d86315073a9e9e5d8957500cd24402c1)  

**ABSTRACT**  
Predicting links in information networks requires deep understanding and careful modeling of network structure. Network embedding, which aims to learn low-dimensional representations of nodes, has been used successfully for the task of link prediction in the past few decades. Existing methods utilize the observed edges in the network to model the interactions between nodes and learn representations which explain the behavior. In addition to the presence of edges, networks often have information which can be used to improve the embedding. For example, in author collaboration networks, the bag of words representing the abstract of co-authored paper can be used as edge attributes. In this paper, we propose a novel approach, which uses the edges and their associated labels to learn node embeddings. Our model jointly optimizes higher order node neighborhood, social roles and edge attributes reconstruction error using deep architecture which can model highly non-linear interactions. We demonstrate the efficacy of our model over existing state-of-the-art methods on two real world data sets. We observe that such attributes can improve the quality of embedding and yield better performance in link prediction.

{{< /ci-details >}}

{{< ci-details summary="Deep Convolutional Networks on Graph-Structured Data (Mikael Henaff et al., 2015)">}}

Mikael Henaff, Joan Bruna, Yann LeCun. (2015)  
**Deep Convolutional Networks on Graph-Structured Data**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/e49ff72d420c8d72e62a9353e3abc053445e59bd)  
Influential Citation Count (55), SS-ID (e49ff72d420c8d72e62a9353e3abc053445e59bd)  

**ABSTRACT**  
Deep Learning's recent successes have mostly relied on Convolutional Networks, which exploit fundamental statistical properties of images, sounds and video data: the local stationarity and multi-scale compositional structure, that allows expressing long range interactions in terms of shorter, localized interactions. However, there exist other important examples, such as text documents or bioinformatic data, that may lack some or all of these strong statistical regularities.  In this paper we consider the general question of how to construct deep architectures with small learning complexity on general non-Euclidean domains, which are typically unknown and need to be estimated from the data. In particular, we develop an extension of Spectral Networks which incorporates a Graph Estimation procedure, that we test on large-scale classification problems, matching or improving over Dropout Networks with far less parameters to estimate.

{{< /ci-details >}}

{{< ci-details summary="Network embedding in biomedical data science (Chang Su et al., 2018)">}}

Chang Su, Jie Tong, Yongjun Zhu, Peng Cui, Fei Wang. (2018)  
**Network embedding in biomedical data science**  
Briefings Bioinform.  
[Paper Link](https://www.semanticscholar.org/paper/e4c66275e46a66586365c851f0974a3c88baf3d7)  
Influential Citation Count (2), SS-ID (e4c66275e46a66586365c851f0974a3c88baf3d7)  

**ABSTRACT**  
Owning to the rapid development of computer technologies, an increasing number of relational data have been emerging in modern biomedical research. Many network-based learning methods have been proposed to perform analysis on such data, which provide people a deep understanding of topology and knowledge behind the biomedical networks and benefit a lot of applications for human healthcare. However, most network-based methods suffer from high computational and space cost. There remain challenges on handling high dimensionality and sparsity of the biomedical networks. The latest advances in network embedding technologies provide new effective paradigms to solve the network analysis problem. It converts network into a low-dimensional space while maximally preserves structural properties. In this way, downstream tasks such as link prediction and node classification can be done by traditional machine learning methods. In this survey, we conduct a comprehensive review of the literature on applying network embedding to advance the biomedical domain. We first briefly introduce the widely used network embedding models. After that, we carefully discuss how the network embedding approaches were performed on biomedical networks as well as how they accelerated the downstream tasks in biomedical science. Finally, we discuss challenges the existing network embedding applications in biomedical domains are faced with and suggest several promising future directions for a better improvement in human healthcare.

{{< /ci-details >}}

{{< ci-details summary="Emergence of scaling in random networks (Barabási et al., 1999)">}}

Barabási, Albert. (1999)  
**Emergence of scaling in random networks**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/e50573b554cfa9ee77dcc2e298d7073a152b7199)  
Influential Citation Count (1960), SS-ID (e50573b554cfa9ee77dcc2e298d7073a152b7199)  

**ABSTRACT**  
Systems as diverse as genetic networks or the World Wide Web are best described as networks with complex topology. A common property of many large networks is that the vertex connectivities follow a scale-free power-law distribution. This feature was found to be a consequence of two generic mechanisms: (i) networks expand continuously by the addition of new vertices, and (ii) new vertices attach preferentially to sites that are already well connected. A model based on these two ingredients reproduces the observed stationary scale-free distributions, which indicates that the development of large networks is governed by robust self-organizing phenomena that go beyond the particulars of the individual systems.

{{< /ci-details >}}

{{< ci-details summary="Clique partitions, graph compression and speeding-up algorithms (T. Feder et al., 1991)">}}

T. Feder, R. Motwani. (1991)  
**Clique partitions, graph compression and speeding-up algorithms**  
STOC '91  
[Paper Link](https://www.semanticscholar.org/paper/e527be9afd581b7e2a4e5b6e6a802be7d7590373)  
Influential Citation Count (18), SS-ID (e527be9afd581b7e2a4e5b6e6a802be7d7590373)  

**ABSTRACT**  
We first consider the problem of partitioning the edges of a graph ~ into bipartite cliques such that the total order of the cliques is minimized, where the order of a clique is the number of vertices in it. It is shown that the problem is NP-complete. We then prove the existence of a partition of small total order in a sufficiently dense graph and devise an efilcient algorithm to compute such a partition. It turns out that our algorithm exhibits a trade-off between the total order of the partition and the running time. Next, we define the notion of a compression of a graph ~ and use the result on graph partitioning to efficiently compute an optimal compression for graphs of a given size. An interesting application of the graph compression result arises from the fact that several graph algorithms can be adapted to work with the compressed rep~esentation of the input graph, thereby improving the bound on their running times particularly on dense graphs. This makes use of the trade-off result we obtain from our partitioning algorithm. The algorithms analyzed include those for matchings, vertex connectivity, edge connectivity and shortest paths. In each case, we improve upon the running times of the best-known algorithms for these problems.

{{< /ci-details >}}

{{< ci-details summary="Indexing by Latent Semantic Analysis (S. Deerwester et al., 1990)">}}

S. Deerwester, S. Dumais, G. Furnas, T. Landauer, R. Harshman. (1990)  
**Indexing by Latent Semantic Analysis**  
  
[Paper Link](https://www.semanticscholar.org/paper/e5305866d701a2c102c5f81fbbf48bf6ac29f252)  
Influential Citation Count (950), SS-ID (e5305866d701a2c102c5f81fbbf48bf6ac29f252)  

**ABSTRACT**  
A new method for automatic indexing and retrieval is described. The approach is to take advantage of implicit higher-order structure in the association of terms with documents (“semantic structure”) in order to improve the detection of relevant documents on the basis of terms found in queries. The particular technique used is singular-value decomposition, in which a large term by document matrix is decomposed into a set of ca. 100 orthogonal factors from which the original matrix can be approximated by linear combination. Documents are represented by ca. 100 item vectors of factor weights. Queries are represented as pseudo-document vectors formed from weighted combinations of terms, and documents with supra-threshold cosine values are returned. initial tests find this completely automatic method for retrieval to be promising.

{{< /ci-details >}}

{{< ci-details summary="Don't Walk, Skip!: Online Learning of Multi-scale Network Embeddings (Bryan Perozzi et al., 2016)">}}

Bryan Perozzi, Vivek Kulkarni, Haochen Chen, S. Skiena. (2016)  
**Don't Walk, Skip!: Online Learning of Multi-scale Network Embeddings**  
ASONAM  
[Paper Link](https://www.semanticscholar.org/paper/e75491aba169909922c6e836a39037a5e6be426e)  
Influential Citation Count (11), SS-ID (e75491aba169909922c6e836a39037a5e6be426e)  

**ABSTRACT**  
We present WALKLETS, a novel approach for learning multiscale representations of vertices in a network. In contrast to previous works, these representations explicitly encode multi-scale vertex relationships in a way that is analytically derivable. WALKLETS generates these multiscale relationships by sub-sampling short random walks on the vertices of a graph. By 'skipping' over steps in each random walk, our method generates a corpus of vertex pairs which are reachable via paths of a fixed length. This corpus can then be used to learn a series of latent representations, each of which captures successively higher order relationships from the adjacency matrix. We demonstrate the efficacy of WALKLETS's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, DBLP, Flickr, and YouTube. Our results show that WALKLETS outperforms new methods based on neural matrix factorization. Specifically, we outperform DeepWalk by up to 10% and LINE by 58% Micro-F1 on challenging multi-label classification tasks. Finally, WALKLETS is an online algorithm, and can easily scale to graphs with millions of vertices and edges.

{{< /ci-details >}}

{{< ci-details summary="A Survey of Link Prediction in Social Networks (M. Hasan et al., 2011)">}}

M. Hasan, Mohammed J. Zaki. (2011)  
**A Survey of Link Prediction in Social Networks**  
Social Network Data Analytics  
[Paper Link](https://www.semanticscholar.org/paper/e7d30fefe1b99c21813873f976e46d03dc82b4fc)  
Influential Citation Count (27), SS-ID (e7d30fefe1b99c21813873f976e46d03dc82b4fc)  

**ABSTRACT**  
Link prediction is an important task for analying social networks which also has applications in other domains like, information retrieval, bioinformatics and e-commerce. There exist a variety of techniques for link prediction, ranging from feature-based classification and kernel-based method to matrix factorization and probabilistic graphical models. These methods differ from each other with respect to model complexity, prediction performance, scalability, and generalization ability. In this article, we survey some representative link prediction methods by categorizing them by the type of the models. We largely consider three types of models: first, the traditional (non-Bayesian) models which extract a set of features to train a binary classification model. Second, the probabilistic approaches which model the joint-probability among the entities in a network by Bayesian graphical models. And, finally the linear algebraic approach which computes the similarity between the nodes in a network by rank-reduced similarity matrices. We discuss various existing link prediction models that fall in these broad categories and analyze their strength and weakness. We conclude the survey with a discussion on recent developments and future research direction.

{{< /ci-details >}}

{{< ci-details summary="Learning with Similarity Functions on Graphs using Matchings of Geometric Embeddings (Fredrik D. Johansson et al., 2015)">}}

Fredrik D. Johansson, Devdatt P. Dubhashi. (2015)  
**Learning with Similarity Functions on Graphs using Matchings of Geometric Embeddings**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/e8f16ec1024a6cffbb4e0d57529e9432207d4a5c)  
Influential Citation Count (1), SS-ID (e8f16ec1024a6cffbb4e0d57529e9432207d4a5c)  

**ABSTRACT**  
We develop and apply the Balcan-Blum-Srebro (BBS) theory of classification via similarity functions (which are not necessarily kernels) to the problem of graph classification. First we place the BBS theory into the unifying framework of optimal transport theory. This also opens the way to exploit coupling methods for establishing properties required of a good similarity function as per their definition. Next, we use the approach to the problem of graph classification via geometric embeddings such as the Laplacian, pseudo-inverse Laplacian and the Lovász orthogonal labellings. We consider the similarity function given by optimal and near--optimal matchings with respect to Euclidean distance of the corresponding embeddings of the graphs in high dimensions. We use optimal couplings to rigorously establish that this yields a "good" similarity measure in the BBS sense for two well known families of graphs. Further, we show that the similarity yields better classification accuracy in practice, on these families, than matchings of other well-known graph embeddings. Finally we perform an extensive empirical evaluation on benchmark data sets where we show that classifying graphs using matchings of geometric embeddings outperforms the previous state-of-the-art methods.

{{< /ci-details >}}

{{< ci-details summary="Drug-Target Interaction Prediction with Graph Regularized Matrix Factorization (Ali Ezzat et al., 2017)">}}

Ali Ezzat, P. Zhao, Min Wu, Xiaoli Li, C. Kwoh. (2017)  
**Drug-Target Interaction Prediction with Graph Regularized Matrix Factorization**  
IEEE/ACM Transactions on Computational Biology and Bioinformatics  
[Paper Link](https://www.semanticscholar.org/paper/e964e8f29e79eb01063836a73dcb6c3c45565914)  
Influential Citation Count (17), SS-ID (e964e8f29e79eb01063836a73dcb6c3c45565914)  

**ABSTRACT**  
Experimental determination of drug-target interactions is expensive and time-consuming. Therefore, there is a continuous demand for more accurate predictions of interactions using computational techniques. Algorithms have been devised to infer novel interactions on a global scale where the input to these algorithms is a drug-target network (i.e., a bipartite graph where edges connect pairs of drugs and targets that are known to interact). However, these algorithms had difficulty predicting interactions involving new drugs or targets for which there are no known interactions (i.e., “orphan” nodes in the network). Since data usually lie on or near to low-dimensional non-linear manifolds, we propose two matrix factorization methods that use graph regularization in order to learn such manifolds. In addition, considering that many of the non-occurring edges in the network are actually unknown or missing cases, we developed a preprocessing step to enhance predictions in the “new drug” and “new target” cases by adding edges with intermediate interaction likelihood scores. In our cross validation experiments, our methods achieved better results than three other state-of-the-art methods in most cases. Finally, we simulated some “new drug” and “new target” cases and found that GRMF predicted the left-out interactions reasonably well.

{{< /ci-details >}}

{{< ci-details summary="Dynamics-Preserving Graph Embedding for Community Mining and Network Immunization (Jianan Zhong et al., 2020)">}}

Jianan Zhong, Hongjun Qiu, B. Shi. (2020)  
**Dynamics-Preserving Graph Embedding for Community Mining and Network Immunization**  
Inf.  
[Paper Link](https://www.semanticscholar.org/paper/ebbe61c75100df486632a9518b4c04ff5795aea9)  
Influential Citation Count (0), SS-ID (ebbe61c75100df486632a9518b4c04ff5795aea9)  

**ABSTRACT**  
In recent years, the graph embedding approach has drawn a lot of attention in the field of network representation and analytics, the purpose of which is to automatically encode network elements into a low-dimensional vector space by preserving certain structural properties. On this basis, downstream machine learning methods can be implemented to solve static network analytic tasks, for example, node clustering based on community-preserving embeddings. However, by focusing only on structural properties, it would be difficult to characterize and manipulate various dynamics operating on the network. In the field of complex networks, epidemic spreading is one of the most typical dynamics in networks, while network immunization is one of the effective methods to suppress the epidemics. Accordingly, in this paper, we present a dynamics-preserving graph embedding method (EpiEm) to preserve the property of epidemic dynamics on networks, i.e., the infectiousness and vulnerability of network nodes. Specifically, we first generate a set of propagation sequences through simulating the Susceptible-Infectious process on a network. Then, we learn node embeddings from an influence matrix using a singular value decomposition method. Finally, we show that the node embeddings can be used to solve epidemics-related community mining and network immunization problems. The experimental results in real-world networks show that the proposed embedding method outperforms several benchmark methods with respect to both community mining and network immunization. The proposed method offers new insights into the exploration of other collective dynamics in complex networks using the graph embedding approach, such as opinion formation in social networks.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised feature selection for linked social media data (Jiliang Tang et al., 2012)">}}

Jiliang Tang, Huan Liu. (2012)  
**Unsupervised feature selection for linked social media data**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/ec8c47ef5797976594c7b784dcad6776743ef014)  
Influential Citation Count (17), SS-ID (ec8c47ef5797976594c7b784dcad6776743ef014)  

**ABSTRACT**  
The prevalent use of social media produces mountains of unlabeled, high-dimensional data. Feature selection has been shown effective in dealing with high-dimensional data for efficient data mining. Feature selection for unlabeled data remains a challenging task due to the absence of label information by which the feature relevance can be assessed. The unique characteristics of social media data further complicate the already challenging problem of unsupervised feature selection, (e.g., part of social media data is linked, which makes invalid the independent and identically distributed assumption), bringing about new challenges to traditional unsupervised feature selection algorithms. In this paper, we study the differences between social media data and traditional attribute-value data, investigate if the relations revealed in linked data can be used to help select relevant features, and propose a novel unsupervised feature selection framework, LUFS, for linked social media data. We perform experiments with real-world social media datasets to evaluate the effectiveness of the proposed framework and probe the working of its key components.

{{< /ci-details >}}

{{< ci-details summary="Network Embedding With Completely-Imbalanced Labels (Zheng Wang et al., 2020)">}}

Zheng Wang, Xiaojun Ye, Chaokun Wang, Jian Cui, Philip S. Yu. (2020)  
**Network Embedding With Completely-Imbalanced Labels**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/ece57b93c36325d909723564044f06986e5553ff)  
Influential Citation Count (2), SS-ID (ece57b93c36325d909723564044f06986e5553ff)  

**ABSTRACT**  
Network embedding, aiming to project a network into a low-dimensional space, is increasingly becoming a focus of network research. Semi-supervised network embedding takes advantage of labeled data, and has shown promising performance. However, existing semi-supervised methods would get unappealing results in the completely-imbalanced label setting where some classes have no labeled nodes at all. To alleviate this, we propose two novel semi-supervised network embedding methods. The first one is a shallow method named RSDNE. Specifically, to benefit from the completely-imbalanced labels, RSDNE guarantees both intra-class similarity and inter-class dissimilarity in an approximate way. The other method is RECT which is a new class of graph neural networks. Different from RSDNE, to benefit from the completely-imbalanced labels, RECT explores the class-semantic knowledge. This enables RECT to handle networks with node features and multi-label setting. Experimental results on several real-world datasets demonstrate the superiority of the proposed methods.

{{< /ci-details >}}

{{< ci-details summary="Representation Learning on Graphs: Methods and Applications (William L. Hamilton et al., 2017)">}}

William L. Hamilton, Rex Ying, J. Leskovec. (2017)  
**Representation Learning on Graphs: Methods and Applications**  
IEEE Data Eng. Bull.  
[Paper Link](https://www.semanticscholar.org/paper/ecf6c42d84351f34e1625a6a2e4cc6526da45c74)  
Influential Citation Count (140), SS-ID (ecf6c42d84351f34e1625a6a2e4cc6526da45c74)  

**ABSTRACT**  
Machine learning on graphs is an important and ubiquitous task with applications ranging from drug design to friendship recommendation in social networks. The primary challenge in this domain is finding a way to represent, or encode, graph structure so that it can be easily exploited by machine learning models. Traditionally, machine learning approaches relied on user-defined heuristics to extract features encoding structural information about a graph (e.g., degree statistics or kernel functions). However, recent years have seen a surge in approaches that automatically learn to encode graph structure into low-dimensional embeddings, using techniques based on deep learning and nonlinear dimensionality reduction. Here we provide a conceptual review of key advancements in this area of representation learning on graphs, including matrix factorization-based methods, random-walk based algorithms, and graph neural networks. We review methods to embed individual nodes as well as approaches to embed entire (sub)graphs. In doing so, we develop a unified framework to describe these recent approaches, and we highlight a number of important applications and directions for future work.

{{< /ci-details >}}

{{< ci-details summary="HARP: Hierarchical Representation Learning for Networks (Haochen Chen et al., 2017)">}}

Haochen Chen, Bryan Perozzi, Yifan Hu, S. Skiena. (2017)  
**HARP: Hierarchical Representation Learning for Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/ee9cc8e663d650ae96405ad680d6447066e6fb23)  
Influential Citation Count (32), SS-ID (ee9cc8e663d650ae96405ad680d6447066e6fb23)  

**ABSTRACT**  
We present HARP, a novel method for learning low dimensional embeddings of a graph's nodes which preserves higher-order structural features. Our proposed method achieves this by compressing the input graph prior to embedding it, effectively avoiding troublesome embedding configurations (i.e. local minima) which can pose problems to non-convex optimization. HARP works by finding a smaller graph which approximates the global structure of its input. This simplified graph is used to learn a set of initial representations, which serve as good initializations for learning representations in the original, detailed graph. We inductively extend this idea, by decomposing a graph in a series of levels, and then embed the hierarchy of graphs from the coarsest one to the original graph. HARP is a general meta-strategy to improve all of the state-of-the-art neural algorithms for embedding graphs, including DeepWalk, LINE, and Node2vec. Indeed, we demonstrate that applying HARP's hierarchical paradigm yields improved implementations for all three of these methods, as evaluated on both classification tasks on real-world graphs such as DBLP, BlogCatalog, CiteSeer, and Arxiv, where we achieve a performance gain over the original implementations by up to 14% Macro F1.

{{< /ci-details >}}

{{< ci-details summary="Exploiting Latent Social Listening Representations for Music Recommendations (Chih-Ming Chen et al., 2015)">}}

Chih-Ming Chen, Chien Po-Chuan, Lin Yu-Ching, Tsai Ming-Feng, Yang Yi-Hsuan, 陳志銘, 蔡銘峰. (2015)  
**Exploiting Latent Social Listening Representations for Music Recommendations**  
RecSys 2015  
[Paper Link](https://www.semanticscholar.org/paper/effae318686b939a864a9787ffd4fa69b844b8d9)  
Influential Citation Count (1), SS-ID (effae318686b939a864a9787ffd4fa69b844b8d9)  

**ABSTRACT**  
Music listening can be regarded as a social activity, in which people can listen together and make friends with one other. Therefore, social relationships may imply multiple facets of the users, such as their listening behaviors and tastes. In this light, it is considered that social relationships hold abundant valuable information that can be utilized for music recommendation. However, utilizing the information for recommendation could be dicult, because such information is usually sparse. To address this issue, we propose to learn the latent social listening representations by the DeepWalk method, and then integrate the learned representations into Factorization Machines to construct better recommendation models. With the DeepWalk method, user social relationships can be transformed from the sparse and independent and identically distributed (i.i.d.) form into a dense and noni.i.d. form. In addition, the latent representations can also capture the spatial locality among users and items, therefore beneting the constructed recommendation models.

{{< /ci-details >}}

{{< ci-details summary="Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs (Federico Monti et al., 2016)">}}

Federico Monti, D. Boscaini, Jonathan Masci, E. Rodolà, Jan Svoboda, M. Bronstein. (2016)  
**Geometric Deep Learning on Graphs and Manifolds Using Mixture Model CNNs**  
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/f09f7888aa5aeaf88a2a44aea768d9a8747e97d2)  
Influential Citation Count (121), SS-ID (f09f7888aa5aeaf88a2a44aea768d9a8747e97d2)  

**ABSTRACT**  
Deep learning has achieved a remarkable performance breakthrough in several fields, most notably in speech recognition, natural language processing, and computer vision. In particular, convolutional neural network (CNN) architectures currently produce state-of-the-art performance on a variety of image analysis tasks such as object detection and recognition. Most of deep learning research has so far focused on dealing with 1D, 2D, or 3D Euclidean-structured data such as acoustic signals, images, or videos. Recently, there has been an increasing interest in geometric deep learning, attempting to generalize deep learning methods to non-Euclidean structured data such as graphs and manifolds, with a variety of applications from the domains of network analysis, computational social science, or computer graphics. In this paper, we propose a unified framework allowing to generalize CNN architectures to non-Euclidean domains (graphs and manifolds) and learn local, stationary, and compositional task-specific features. We show that various non-Euclidean CNN methods previously proposed in the literature can be considered as particular instances of our framework. We test the proposed method on standard tasks from the realms of image-, graph-and 3D shape analysis and show that it consistently outperforms previous approaches.

{{< /ci-details >}}

{{< ci-details summary="FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling (Jing Chen et al., 2018)">}}

Jing Chen, Tengfei Ma, Cao Xiao. (2018)  
**FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/f19d3e0956d0f2daa3396fc6e9e7554a78a90710)  
Influential Citation Count (118), SS-ID (f19d3e0956d0f2daa3396fc6e9e7554a78a90710)  

**ABSTRACT**  
The graph convolutional networks (GCN) recently proposed by Kipf and Welling are an effective graph model for semi-supervised learning. This model, however, was originally designed to be learned with the presence of both training and test data. Moreover, the recursive neighborhood expansion across layers poses time and memory challenges for training with large, dense graphs. To relax the requirement of simultaneous availability of test data, we interpret graph convolutions as integral transforms of embedding functions under probability measures. Such an interpretation allows for the use of Monte Carlo approaches to consistently estimate the integrals, which in turn leads to a batched training scheme as we propose in this work---FastGCN. Enhanced with importance sampling, FastGCN not only is efficient for training but also generalizes well for inference. We show a comprehensive set of experiments to demonstrate its effectiveness compared with GCN and related models. In particular, training is orders of magnitude more efficient while predictions remain comparably accurate.

{{< /ci-details >}}

{{< ci-details summary="Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing (Antoine Bordes et al., 2012)">}}

Antoine Bordes, Xavier Glorot, J. Weston, Yoshua Bengio. (2012)  
**Joint Learning of Words and Meaning Representations for Open-Text Semantic Parsing**  
AISTATS  
[Paper Link](https://www.semanticscholar.org/paper/f2f72cfb48d15d4d2bd1e91a92e7f3ac8635d433)  
Influential Citation Count (19), SS-ID (f2f72cfb48d15d4d2bd1e91a92e7f3ac8635d433)  

**ABSTRACT**  
Open-text semantic parsers are designed to interpret any statement in natural language by inferring a corresponding meaning representation (MR – a formal representation of its sense). Unfortunately, large scale systems cannot be easily machine-learned due to a lack of directly supervised data. We propose a method that learns to assign MRs to a wide range of text (using a dictionary of more than 70,000 words mapped to more than 40,000 entities) thanks to a training scheme that combines learning from knowledge bases (e.g. WordNet) with learning from raw text. The model jointly learns representations of words, entities and MRs via a multi-task training process operating on these diverse sources of data. Hence, the system ends up providing methods for knowledge acquisition and wordsense disambiguation within the context of semantic parsing in a single elegant framework. Experiments on these various tasks indicate the promise of the approach.

{{< /ci-details >}}

{{< ci-details summary="Learning Combinatorial Optimization Algorithms over Graphs (Elias Boutros Khalil et al., 2017)">}}

Elias Boutros Khalil, H. Dai, Yuyu Zhang, B. Dilkina, Le Song. (2017)  
**Learning Combinatorial Optimization Algorithms over Graphs**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/f306b1a973d9fa8c693036ca75fa8e30ad709635)  
Influential Citation Count (90), SS-ID (f306b1a973d9fa8c693036ca75fa8e30ad709635)  

**ABSTRACT**  
The design of good heuristics or approximation algorithms for NP-hard combinatorial optimization problems often requires significant specialized knowledge and trial-and-error. Can we automate this challenging, tedious process, and learn the algorithms instead? In many real-world applications, it is typically the case that the same optimization problem is solved again and again on a regular basis, maintaining the same problem structure but differing in the data. This provides an opportunity for learning heuristic algorithms that exploit the structure of such recurring problems. In this paper, we propose a unique combination of reinforcement learning and graph embedding to address this challenge. The learned greedy policy behaves like a meta-algorithm that incrementally constructs a solution, and the action is determined by the output of a graph embedding network capturing the current state of the solution. We show that our framework can be applied to a diverse range of optimization problems over graphs, and learns effective algorithms for the Minimum Vertex Cover, Maximum Cut and Traveling Salesman problems.

{{< /ci-details >}}

{{< ci-details summary="GloVe: Global Vectors for Word Representation (Jeffrey Pennington et al., 2014)">}}

Jeffrey Pennington, R. Socher, Christopher D. Manning. (2014)  
**GloVe: Global Vectors for Word Representation**  
EMNLP  
[Paper Link](https://www.semanticscholar.org/paper/f37e1b62a767a307c046404ca96bc140b3e68cb5)  
Influential Citation Count (3451), SS-ID (f37e1b62a767a307c046404ca96bc140b3e68cb5)  

**ABSTRACT**  
Recent methods for learning vector space representations of words have succeeded in capturing fine-grained semantic and syntactic regularities using vector arithmetic, but the origin of these regularities has remained opaque. We analyze and make explicit the model properties needed for such regularities to emerge in word vectors. The result is a new global logbilinear regression model that combines the advantages of the two major model families in the literature: global matrix factorization and local context window methods. Our model efficiently leverages statistical information by training only on the nonzero elements in a word-word cooccurrence matrix, rather than on the entire sparse matrix or on individual context windows in a large corpus. The model produces a vector space with meaningful substructure, as evidenced by its performance of 75% on a recent word analogy task. It also outperforms related models on similarity tasks and named entity recognition.

{{< /ci-details >}}

{{< ci-details summary="SSNE: Status Signed Network Embedding (Chunyu Lu et al., 2019)">}}

Chunyu Lu, Pengfei Jiao, Hongtao Liu, Yaping Wang, Hongyan Xu, Wenjun Wang. (2019)  
**SSNE: Status Signed Network Embedding**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/f429e69863223ca62d2fa1fd667b18ddac0cb3de)  
Influential Citation Count (0), SS-ID (f429e69863223ca62d2fa1fd667b18ddac0cb3de)  

**ABSTRACT**  
This work studies the problem of signed network embedding, which aims to obtain low-dimensional vectors for nodes in signed networks. Existing works mostly focus on learning representations via characterizing the social structural balance theory in signed networks. However, structural balance theory could not well satisfy some of the fundamental phenomena in real-world signed networks such as the direction of links. As a result, in this paper we integrate another theory Status Theory into signed network embedding since status theory can better explain the social mechanisms of signed networks. To be specific, we characterize the status of nodes in the semantic vector space and well design different ranking objectives for positive and negative links respectively. Besides, we utilize graph attention to assemble the information of neighborhoods. We conduct extensive experiments on three real-world datasets and the results show that our model can achieve a significant improvement compared with baselines.

{{< /ci-details >}}

{{< ci-details summary="Semantic Proximity Search on Heterogeneous Graph by Proximity Embedding (Zemin Liu et al., 2017)">}}

Zemin Liu, V. Zheng, Zhou Zhao, Fanwei Zhu, K. Chang, Minghui Wu, Jing Ying. (2017)  
**Semantic Proximity Search on Heterogeneous Graph by Proximity Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/f4ef2c9a97c183b24d08ca67f6f55b98d441e8e6)  
Influential Citation Count (3), SS-ID (f4ef2c9a97c183b24d08ca67f6f55b98d441e8e6)  

**ABSTRACT**  
Many real-world networks have a rich collection of objects. The semantics of these objects allows us to capture different classes of proximities, thus enabling an important task of semantic proximity search. As the core of semantic proximity search, we have to measure the proximity on a heterogeneous graph, whose nodes are various types of objects. Most of the existing methods rely on engineering features about the graph structure between two nodes to measure their proximity. With recent development on graph embedding, we see a good chance to avoid feature engineering for semantic proximity search. There is very little work on using graph embedding for semantic proximity search. We also observe that graph embedding methods typically focus on embedding nodes, which is an “indirect” approach to learn the proximity. Thus, we introduce a new concept of proximity embedding, which directly embeds the network structure between two possibly distant nodes. We also design our proximity embedding, so as to flexibly support both symmetric and asymmetric proximities. Based on the proximity embedding, we can easily estimate the proximity score between two nodes and enable search on the graph. We evaluate our proximity embedding method on three real-world public data sets, and show it outperforms the state-of-the-art baselines. We release the code for proximity embedding.

{{< /ci-details >}}

{{< ci-details summary="Geographic Routing Using Hyperbolic Space (Robert D. Kleinberg, 2007)">}}

Robert D. Kleinberg. (2007)  
**Geographic Routing Using Hyperbolic Space**  
IEEE INFOCOM 2007 - 26th IEEE International Conference on Computer Communications  
[Paper Link](https://www.semanticscholar.org/paper/f506b2ddb142d2ec539400297ba53383d958abef)  
Influential Citation Count (56), SS-ID (f506b2ddb142d2ec539400297ba53383d958abef)  

**ABSTRACT**  
We propose a scalable and reliable point-to-point routing algorithm for ad hoc wireless networks and sensor-nets. Our algorithm assigns to each node of the network a virtual coordinate in the hyperbolic plane, and performs greedy geographic routing with respect to these virtual coordinates. Unlike other proposed greedy routing algorithms based on virtual coordinates, our embedding guarantees that the greedy algorithm is always successful in finding a route to the destination, if such a route exists. We describe a distributed algorithm for computing each node's virtual coordinates in the hyperbolic plane, and for greedily routing packets to a destination point in the hyperbolic plane. (This destination may be the address of another node of the network, or it may be an address associated to a piece of content in a Distributed Hash Table. In the latter case we prove that the greedy routing strategy makes a consistent choice of the node responsible for the address, irrespective of the source address of the request.) We evaluate the resulting algorithm in terms of both path stretch and node congestion.

{{< /ci-details >}}

{{< ci-details summary="Enhanced Unsupervised Graph Embedding via Hierarchical Graph Convolution Network (H. Zhang et al., 2020)">}}

H. Zhang, J. Zhou, R. Li. (2020)  
**Enhanced Unsupervised Graph Embedding via Hierarchical Graph Convolution Network**  
  
[Paper Link](https://www.semanticscholar.org/paper/f597a70c8708b7cda9766a403e3a2a67162d6973)  
Influential Citation Count (0), SS-ID (f597a70c8708b7cda9766a403e3a2a67162d6973)  

**ABSTRACT**  
Graph embedding aims to learn the low-dimensional representation of nodes in the network, which has been paid more and more attention in many graph-based tasks recently. Graph Convolution Network (GCN) is a typical deep semisupervised graph embedding model, which can acquire node representation from the complex network. However, GCN usually needs to use a lot of labeled data and additional expressive features in the graph embedding learning process, so the model cannot be effectively applied to undirected graphs with only network structure information. In this paper, we propose a novel unsupervised graph embedding method via hierarchical graph convolution network (HGCN). Firstly, HGCN builds the initial node embedding and pseudo-labels for the undirected graphs, and then further uses GCNs to learn the node embedding and update labels, finally combines HGCN output representation with the initial embedding to get the graph embedding. Furthermore, we improve the model to match the different undirected networks according to the number of network node label types. Comprehensive experiments demonstrate that our proposed HGCN and HGCN can significantly enhance the performance of the node classification task.

{{< /ci-details >}}

{{< ci-details summary="Greedy forwarding in scale-free networks embedded in hyperbolic metric spaces (Dmitri V. Krioukov et al., 2009)">}}

Dmitri V. Krioukov, F. Papadopoulos, M. Boguñá, Amin Vahdat. (2009)  
**Greedy forwarding in scale-free networks embedded in hyperbolic metric spaces**  
SIGMETRICS Perform. Evaluation Rev.  
[Paper Link](https://www.semanticscholar.org/paper/f61005ce7db38553f2bf87be7c9fbce183b4c375)  
Influential Citation Count (4), SS-ID (f61005ce7db38553f2bf87be7c9fbce183b4c375)  

**ABSTRACT**  
We show that complex (scale-free) network topologies naturally emerge from hyperbolic metric spaces. Hyperbolic geometry facilitates maximally efficient greedy forwarding in these networks. Greedy forwarding is topology-oblivious. Nevertheless, greedy packets find their destinations with 100% probability following almost optimal shortest paths. This remarkable efficiency sustains even in highly dynamic networks. Our findings suggest that forwarding information through complex networks, such as the Internet, is possible without the overhead of existing routing protocols, and may also find practical applications in overlay networks for tasks such as application-level routing, information sharing, and data distribution.

{{< /ci-details >}}

{{< ci-details summary="A Three-Way Model for Collective Learning on Multi-Relational Data (Maximilian Nickel et al., 2011)">}}

Maximilian Nickel, Volker Tresp, H. Kriegel. (2011)  
**A Three-Way Model for Collective Learning on Multi-Relational Data**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/f6764d853a14b0c34df1d2283e76277aead40fde)  
Influential Citation Count (279), SS-ID (f6764d853a14b0c34df1d2283e76277aead40fde)  

**ABSTRACT**  
Relational learning is becoming increasingly important in many areas of application. Here, we present a novel approach to relational learning based on the factorization of a three-way tensor. We show that unlike other tensor approaches, our method is able to perform collective learning via the latent components of the model and provide an efficient algorithm to compute the factorization. We substantiate our theoretical considerations regarding the collective learning capabilities of our model by the means of experiments on both a new dataset and a dataset commonly used in entity resolution. Furthermore, we show on common benchmark datasets that our approach achieves better or on-par results, if compared to current state-of-the-art relational learning solutions, while it is significantly faster to compute.

{{< /ci-details >}}

{{< ci-details summary="Video suggestion and discovery for youtube: taking random walks through the view graph (S. Baluja et al., 2008)">}}

S. Baluja, Rohan Seth, D. Sivakumar, Yushi Jing, J. Yagnik, Shankar Kumar, Deepak Ravichandran, M. Aly. (2008)  
**Video suggestion and discovery for youtube: taking random walks through the view graph**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/f68ba8fe9fc9f7d8b7eb8c3d4a6d1046ee345e4b)  
Influential Citation Count (34), SS-ID (f68ba8fe9fc9f7d8b7eb8c3d4a6d1046ee345e4b)  

**ABSTRACT**  
The rapid growth of the number of videos in YouTube provides enormous potential for users to find content of interest to them. Unfortunately, given the difficulty of searching videos, the size of the video repository also makes the discovery of new content a daunting task. In this paper, we present a novel method based upon the analysis of the entire user-video graph to provide personalized video suggestions for users. The resulting algorithm, termed Adsorption, provides a simple method to efficiently propagate preference information through a variety of graphs. We extensively test the results of the recommendations on a three month snapshot of live data from YouTube.

{{< /ci-details >}}

{{< ci-details summary="Neural IR Meets Graph Embedding: A Ranking Model for Product Search (Yuan Zhang et al., 2019)">}}

Yuan Zhang, Dong Wang, Yan Zhang. (2019)  
**Neural IR Meets Graph Embedding: A Ranking Model for Product Search**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/f6c985149798760da88f871cf71148bbeca69b2a)  
Influential Citation Count (3), SS-ID (f6c985149798760da88f871cf71148bbeca69b2a)  

**ABSTRACT**  
Recently, neural models for information retrieval are becoming increasingly popular. They provide effective approaches for product search due to their competitive advantages in semantic matching. However, it is challenging to use graph-based features, though proved very useful in IR literature, in these neural approaches. In this paper, we leverage the recent advances in graph embedding techniques to enable neural retrieval models to exploit graph-structured data for automatic feature extraction. The proposed approach can not only help to overcome the long-tail problem of click-through data, but also incorporate external heterogeneous information to improve search results. Extensive experiments on a real-world e-commerce dataset demonstrate significant improvement achieved by our proposed approach over multiple strong baselines both as an individual retrieval model and as a feature used in learning-to-rank frameworks.

{{< /ci-details >}}

{{< ci-details summary="dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning (Palash Goyal et al., 2018)">}}

Palash Goyal, Sujit Rokka Chhetri, A. Canedo. (2018)  
**dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning**  
Knowl. Based Syst.  
[Paper Link](https://www.semanticscholar.org/paper/f6e59062382fdec9b95c3abef1c27efc3b2ec1c7)  
Influential Citation Count (23), SS-ID (f6e59062382fdec9b95c3abef1c27efc3b2ec1c7)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Heterogeneous Network Embedding via Deep Architectures (Shiyu Chang et al., 2015)">}}

Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, C. Aggarwal, Thomas S. Huang. (2015)  
**Heterogeneous Network Embedding via Deep Architectures**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
Influential Citation Count (35), SS-ID (f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  

**ABSTRACT**  
Data embedding is used in many machine learning applications to create low-dimensional feature representations, which preserves the structure of data points in their original space. In this paper, we examine the scenario of a heterogeneous network with nodes and content of various types. Such networks are notoriously difficult to mine because of the bewildering combination of heterogeneous contents and structures. The creation of a multidimensional embedding of such data opens the door to the use of a wide variety of off-the-shelf mining techniques for multidimensional data. Despite the importance of this problem, limited efforts have been made on embedding a network of scalable, dynamic and heterogeneous data. In such cases, both the content and linkage structure provide important cues for creating a unified feature representation of the underlying network. In this paper, we design a deep embedding algorithm for networked data. A highly nonlinear multi-layered embedding function is used to capture the complex interactions between the heterogeneous data in a network. Our goal is to create a multi-resolution deep embedding function, that reflects both the local and global network structures, and makes the resulting embedding useful for a variety of data mining tasks. In particular, we demonstrate that the rich content and linkage information in a heterogeneous network can be captured by such an approach, so that similarities among cross-modal data can be measured directly in a common embedding space. Once this goal has been achieved, a wide variety of data mining problems can be solved by applying off-the-shelf algorithms designed for handling vector representations. Our experiments on real-world network datasets show the effectiveness and scalability of the proposed algorithm as compared to the state-of-the-art embedding methods.

{{< /ci-details >}}

{{< ci-details summary="Collective Classification via Discriminative Matrix Factorization on Sparsely Labeled Networks (Daokun Zhang et al., 2016)">}}

Daokun Zhang, Jie Yin, Xingquan Zhu, Chengqi Zhang. (2016)  
**Collective Classification via Discriminative Matrix Factorization on Sparsely Labeled Networks**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/f95b14cba00f995ee1c12444551d92cf43640263)  
Influential Citation Count (1), SS-ID (f95b14cba00f995ee1c12444551d92cf43640263)  

**ABSTRACT**  
We address the problem of classifying sparsely labeled networks, where labeled nodes in the network are extremely scarce. Existing algorithms, such as collective classification, have been shown to be effective for jointly deriving labels of related nodes, by exploiting class label dependencies among neighboring nodes. However, when the underlying network is sparsely labeled, most nodes have too few or even no connections to labeled nodes. This makes it very difficult to leverage supervised knowledge from labeled nodes to accurately estimate label dependencies, thereby largely degrading the classification accuracy. In this paper, we propose a novel discriminative matrix factorization (DMF) based algorithm that effectively learns a latent network representation by exploiting topological paths between labeled and unlabeled nodes, in addition to nodes' content information. The main idea is to use matrix factorization to obtain a compact representation of the network that fully encodes nodes' content information and network structure, and unleash discriminative power inferred from labeled nodes to directly benefit collective classification. To achieve this, we formulate a new matrix factorization objective function that integrates network representation learning with an empirical loss minimization for classifying node labels. An efficient optimization algorithm based on conjugate gradient methods is proposed to solve the new objective function. Experimental results on real-world networks show that DMF yields superior performance gain over the state-of-the-art baselines on sparsely labeled networks.

{{< /ci-details >}}

{{< ci-details summary="Self-Grouping Multi-network Clustering (Jingchao Ni et al., 2016)">}}

Jingchao Ni, Wei Cheng, Wei Fan, X. Zhang. (2016)  
**Self-Grouping Multi-network Clustering**  
2016 IEEE 16th International Conference on Data Mining (ICDM)  
[Paper Link](https://www.semanticscholar.org/paper/fabfba8835470ba7a33dfc25e50ef83a5be7eae0)  
Influential Citation Count (0), SS-ID (fabfba8835470ba7a33dfc25e50ef83a5be7eae0)  

**ABSTRACT**  
Joint clustering of multiple networks has been shown to be more accurate than performing clustering on individual networks separately. Many multi-view and multi-domain network clustering methods have been developed for joint multi-network clustering. These methods typically assume there is a common clustering structure shared by all networks, and different networks can provide complementary information on this underlying clustering structure. However, this assumption is too strict to hold in many emerging real-life applications, where multiple networks have diverse data distributions. More popularly, the networks in consideration belong to different underlying groups. Only networks in the same underlying group share similar clustering structures. Better clustering performance can be achieved by considering such groups differently. As a result, an ideal method should be able to automatically detect network groups so that networks in the same group share a common clustering structure. To address this problem, we propose a novel method, ComClus, to simultaneously group and cluster multiple networks. ComClus treats node clusters as features of networks and uses them to differentiate different network groups. Network grouping and clustering are coupled and mutually enhanced during the learning process. Extensive experimental evaluation on a variety of synthetic and real datasets demonstrates the effectiveness of our method.

{{< /ci-details >}}

{{< ci-details summary="Towards Data Poisoning Attack against Knowledge Graph Embedding (Hengtong Zhang et al., 2019)">}}

Hengtong Zhang, T. Zheng, Jing Gao, Chenglin Miao, Lu Su, Yaliang Li, K. Ren. (2019)  
**Towards Data Poisoning Attack against Knowledge Graph Embedding**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/fc82a1e8b4cf04ce42e14ed6347ce81c9563cf14)  
Influential Citation Count (1), SS-ID (fc82a1e8b4cf04ce42e14ed6347ce81c9563cf14)  

**ABSTRACT**  
Knowledge graph embedding (KGE) is a technique for learning continuous embeddings for entities and relations in the knowledge graph.Due to its benefit to a variety of downstream tasks such as knowledge graph completion, question answering and recommendation, KGE has gained significant attention recently. Despite its effectiveness in a benign environment, KGE' robustness to adversarial attacks is not well-studied. Existing attack methods on graph data cannot be directly applied to attack the embeddings of knowledge graph due to its heterogeneity. To fill this gap, we propose a collection of data poisoning attack strategies, which can effectively manipulate the plausibility of arbitrary targeted facts in a knowledge graph by adding or deleting facts on the graph. The effectiveness and efficiency of the proposed attack strategies are verified by extensive evaluations on two widely-used benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Network Representation Learning with Rich Text Information (Cheng Yang et al., 2015)">}}

Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, Edward Y. Chang. (2015)  
**Network Representation Learning with Rich Text Information**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/fce14c6aa64e888456256ac6796744683165a0ff)  
Influential Citation Count (162), SS-ID (fce14c6aa64e888456256ac6796744683165a0ff)  

**ABSTRACT**  
Representation learning has shown its effectiveness in many tasks such as image classification and text mining. Network representation learning aims at learning distributed vector representation for each vertex in a network, which is also increasingly recognized as an important aspect for network analysis. Most network representation learning methods investigate network structures for learning. In reality, network vertices contain rich information (such as text), which cannot be well applied with algorithmic frameworks of typical representation learning methods. By proving that DeepWalk, a state-of-the-art network representation method, is actually equivalent to matrix factorization (MF), we propose text-associated DeepWalk (TADW). TADW incorporates text features of vertices into network representation learning under the framework of matrix factorization. We evaluate our method and various baseline methods by applying them to the task of multi-class classification of vertices. The experimental results show that, our method outperforms other baselines on all three datasets, especially when networks are noisy and training ratio is small. The source code of this paper can be obtained from https://github.com/albertyang33/TADW.

{{< /ci-details >}}

{{< ci-details summary="SNE: Signed Network Embedding (Shuhan Yuan et al., 2017)">}}

Shuhan Yuan, Xintao Wu, Yang Xiang. (2017)  
**SNE: Signed Network Embedding**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/feee6ea8961398e599577f9f793230d391985b88)  
Influential Citation Count (13), SS-ID (feee6ea8961398e599577f9f793230d391985b88)  

**ABSTRACT**  
Several network embedding models have been developed for unsigned networks. However, these models based on skip-gram cannot be applied to signed networks because they can only deal with one type of link. In this paper, we present our signed network embedding model called SNE. Our SNE adopts the log-bilinear model, uses node representations of all nodes along a given path, and further incorporates two signed-type vectors to capture the positive or negative relationship of each edge along the path. We conduct two experiments, node classification and link prediction, on both directed and undirected signed networks and compare with four baselines including a matrix factorization method and three state-of-the-art unsigned network embedding models. The experimental results demonstrate the effectiveness of our signed network embedding.

{{< /ci-details >}}

{{< ci-details summary="Co-authorship Network Embedding and Recommending Collaborators via Network Embedding (Ilya Makarov et al., 2018)">}}

Ilya Makarov, Olga Gerasimova, Pavel Sulimov, L. Zhukov. (2018)  
**Co-authorship Network Embedding and Recommending Collaborators via Network Embedding**  
AIST  
[Paper Link](https://www.semanticscholar.org/paper/ff48f9b6be3c60101873391782d7eecf3a5a3247)  
Influential Citation Count (0), SS-ID (ff48f9b6be3c60101873391782d7eecf3a5a3247)  

**ABSTRACT**  
Co-authorship networks contain invisible patterns of collaboration among researchers. The process of writing joint paper can depend of different factors, such as friendship, common interests, and policy of university. We show that, having a temporal co-authorship network, it is possible to predict future publications. We solve the problem of recommending collaborators from the point of link prediction using graph embedding, obtained from co-authorship network. We run experiments on data from HSE publications graph and compare it with relevant models.

{{< /ci-details >}}

{{< ci-details summary="Dynamic Heterogeneous Graph Embedding Using Hierarchical Attentions (Luwei Yang et al., 2020)">}}

Luwei Yang, Zhibo Xiao, Wen Jiang, Yi Wei, Y. Hu, Hao Wang. (2020)  
**Dynamic Heterogeneous Graph Embedding Using Hierarchical Attentions**  
ECIR  
[Paper Link](https://www.semanticscholar.org/paper/ffe5b25c6cf8de37823907c3aed7738ea393902e)  
Influential Citation Count (1), SS-ID (ffe5b25c6cf8de37823907c3aed7738ea393902e)  

**ABSTRACT**  
Graph embedding has attracted many research interests. Existing works mainly focus on static homogeneous/heterogeneous networks or dynamic homogeneous networks. However, dynamic heterogeneous networks are more ubiquitous in reality, e.g. social network, e-commerce network, citation network, etc. There is still a lack of research on dynamic heterogeneous graph embedding. In this paper, we propose a novel dynamic heterogeneous graph embedding method using hierarchical attentions (DyHAN) that learns node embeddings leveraging both structural heterogeneity and temporal evolution. We evaluate our method on three real-world datasets. The results show that DyHAN outperforms various state-of-the-art baselines in terms of link prediction task.

{{< /ci-details >}}

{{< ci-details summary="DeepWalk: online learning of social representations (Bryan Perozzi et al., 2014)">}}

Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2014)  
**DeepWalk: online learning of social representations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fff114cbba4f3ba900f33da574283e3de7f26c83)  
Influential Citation Count (1335), SS-ID (fff114cbba4f3ba900f33da574283e3de7f26c83)  

**ABSTRACT**  
We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

{{< /ci-details >}}

