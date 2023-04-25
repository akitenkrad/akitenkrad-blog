---
draft: true
title: "NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks"
date: 2022-09-15
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2018", "Graph Neural Network", "DS:UCI Messages", "DS:arXiv hep-th", "DS:Digg", "DS:DBLP"]
menu:
  sidebar:
    name: "NetWalk: A Flexible Deep Embedding Approach for Anomaly Detection in Dynamic Networks"
    identifier: 20220915
    parent: 202209
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Yu, W., Cheng, W., Aggarwal, C. C., Zhang, K., Chen, H., & Wang, W. (2018).  
NetWalk: A flexible deep embedding approach for anomaly detection in dynamic networks.  
Knowledge Discovery and Data Mining -2018, 2672–2681.  
https://doi.org/10.1145/3219819.3220024
{{< /citation >}}

## Abstract
> Massive and dynamic networks arise in many practical applications such as social media, security and public health. Given an evolutionary network, it is crucial to detect structural anomalies, such as vertices and edges whose “behaviors” deviate from underlying majority of the network, in a real-time fashion. Recently, network embedding has proven a powerful tool in learning the low-dimensional representations of vertices in networks that can capture and preserve the network structure. However, most existing network embedding approaches are designed for static networks, and thus may not be perfectly suited for a dynamic environment in which the network representation has to be constantly updated. In this paper, we propose a novel approach, NetWalk, for anomaly detection in dynamic networks by learning network representations which can be updated dynamically as the network evolves. We first encode the vertices of the dynamic network to vector representations by clique embedding, which jointly minimizes the pairwise distance of vertex representations of each walk derived from the dynamic networks, and the deep autoencoder reconstruction error serving as a global regularization. The vector representations can be computed with constant space requirements using reservoir sampling. On the basis of the learned low-dimensional vertex representations, a clustering-based technique is employed to incrementally and dynamically detect network anomalies. Compared with existing approaches, NetWalk has several advantages: 1) the network embedding can be updated dynamically, 2) streaming network nodes and edges can be encoded efficiently with constant memory space usage, 3). flexible to be applied on different types of networks, and 4) network anomalies can be detected in real-time. Extensive experiments on four real datasets demonstrate the effectiveness of NetWalk.

## Background & Wat's New
- ネットワークの異常検知において，ネットワークの構造に関する情報を保持した分散表現は重要である
- 既存手法では，ノードやエッジが動的に変化するDynamic Networkを扱えない
- ネットワークの構造を動的に捉えるために，dynamic clustering algorithmをベースとした手法であるNetWalkを提案した
- ネットワークの分散表現を学習するにあたって，深層学習とreservoir samplingによる効率的なアルゴリズムを考案した

{{< figure src="NetWalk.png" width="100%" caption="Related Works" >}}

## Dataset

{{< ci-details summary="UCI Messages (Tore Opsahl et al., 2009)">}}
Tore Opsahl, P. Panzarasa. (2009)  
**Clustering in weighted networks**  
Soc. Networks  
[Paper Link](https://www.semanticscholar.org/paper/95f568b1dfc1078b6a6227f522c5406b2d52e949)  
Influential Citation Count (57), SS-ID (95f568b1dfc1078b6a6227f522c5406b2d52e949)  
{{< /ci-details >}}

{{< ci-details summary="arXiv hep-th data (J. Leskovec et al., 2006)">}}
J. Leskovec, J. Kleinberg, C. Faloutsos. (2006)  
**Graph evolution: Densification and shrinking diameters**  
TKDD  
[Paper Link](https://www.semanticscholar.org/paper/5929e6031115e3dfa4b4f12071f4e16b24a003e0)  
Influential Citation Count (200), SS-ID (5929e6031115e3dfa4b4f12071f4e16b24a003e0)  
**ABSTRACT**  
How do real graphs evolve over time? What are normal growth patterns in social, technological, and information networks? Many studies have discovered patterns in static graphs, identifying properties in a single snapshot of a large network or in a very small number of snapshots; these include heavy tails for in- and out-degree distributions, communities, small-world phenomena, and others. However, given the lack of information about network evolution over long periods, it has been hard to convert these findings into statements about trends over time.  Here we study a wide range of real graphs, and we observe some surprising phenomena. First, most of these graphs densify over time with the number of edges growing superlinearly in the number of nodes. Second, the average distance between nodes often shrinks over time in contrast to the conventional wisdom that such distance parameters should increase slowly as a function of the number of nodes (like O(log n) or O(log(log n)).  Existing graph generation models do not exhibit these types of behavior even at a qualitative level. We provide a new graph generator, based on a forest fire spreading process that has a simple, intuitive justification, requires very few parameters (like the flammability of nodes), and produces graphs exhibiting the full range of properties observed both in prior work and in the present study.  We also notice that the forest fire model exhibits a sharp transition between sparse graphs and graphs that are densifying. Graphs with decreasing distance between the nodes are generated around this transition point.  Last, we analyze the connection between the temporal evolution of the degree distribution and densification of a graph. We find that the two are fundamentally related. We also observe that real networks exhibit this type of relation between densification and the degree distribution.
{{< /ci-details >}}


{{< ci-details summary="Digg" >}}
http://konect.uni-koblenz.de/networks
{{< /ci-details>}}

{{< ci-details summary="DBLP" >}}
https://www.aminer.org/citation
{{< /ci-details >}}

{{< figure src="dataset.png" caption="Datasets" >}}

## Model Description

### Problem Formulation

| Notation | Description |
|:--------:|-------------|
| $\mathcal{E}(t)$ | streaming edges received from timestamps 1 to $t$ |
| $\mathcal{V}(t)$ | vertex set across timestamps 1 to $t$ |
| $\mathcal{G}(t) = (\mathcal{E}(t), \mathcal{V}(t))$ | the network at timestamp $t$ |
| $\Omega (t)$ | network walk set of $\mathcal{G}(t)$ |
| $n = \|\mathcal{V}(t)\|$ | number of vertices |
| $m = \|\mathcal{E}(t)\|$ | number of edges |
| $l$ | walk length |
| $\psi$ | number of network walks per vertex |
| $\|\Omega\| = n \times \psi $ | total number of network walks|
| $d$ | latent dimension of vertex representation |
| $\rho$ | sparsity parameter |
| $n\_l$ | total number of layers of the autoencoder network |
| $x\_p^i \in \mathbb{R}^n$ | input vector of vertex $p \in [1,l]$ in walk $i \in [1, \|\Omega\|]$ |
| $W^l \in \mathbb{R}^{n \times d}$ | weight matrix at layer $l$ |
| $b^l \in \mathbb{R}^d$ | bias vector at layer $l$ |
| $D \in \mathbb{R}^{n \times n}$ | diagonal degree matrix |
| $f^l(x)$ | network output of layer $l$ |

### Encoding network strams

#### Network Walk Generation
インプットのネットワークをNetwork Walksに再構築する

{{< box-with-title title="Network Walk" >}}
グラフ $\mathcal{G}(\mathcal{E}, \mathcal{V})$ が与えられたとき，$v\_1 \in \mathcal{V}$ の **Network Walk** を次のように定める．

$$
\Omega\_{v\_1} = \left\lbrace (v\_1, v\_2, \ldots, v\_l) | (v\_i, v\_{i+1}) \in \mathcal{E} \land p(v\_i, v\_{i+1}) = \frac{1}{D\_{v\_i, v\_i}} \right\rbrace
$$

{{< /box-with-title >}}

$\Omega\_{v\_i}$ は長さ $l$-hop のランダムウォークによるノードの集合で，遷移確率は各ノードの次数の逆数で定義される．

単語の出現頻度はZipf's lawにしたがうことが知られているが，ネットワークにおける次数の分布を調べたところ，次数分布もZipf's lawにしたがうことが観察された．

#### Leaning Network Representations
NLPにおけるskip-gramを参考にして，**clique embedding** を提案する．

##### Autoencoder Network

$$
\begin{array}{ll}
  \begin{array}{ll}
    f^{\frac{n\_l}{2}}(x\_p^i) &= \sigma\left( {W^{\frac{n\_l}{2}}}^\mathsf{T} h^{\frac{n\_l}{2}}(x\_p^i) + b^{\frac{n\_l}{2}} \right) &\in \mathbb{R}^n \\\\
    h^{\frac{n\_l}{2}}(x\_p^i) &= W^{\frac{n\_l}{2} - 1}f^{\frac{n\_l}{2} - 1}(x\_p^i) + b^{\frac{n\_l}{2} - 1} &\in \mathbb{R}^d
  \end{array} \tag{1} \\\\
  \text{where} \hspace{10pt} \left\lbrace\begin{array}{ll}
    \sigma \mapsto \text{sigmoid function} \\\\
    n\_l \geqq 2 \\\\
    f^0(x\_p^i) = x\_p^i \\\\
    0 \sim \frac{n\_l}{2} \mapsto \text{Encoder} \\\\
    \frac{n\_l}{2} \sim n\_l \mapsto \text{Decoder}
  \end{array}\right .
\end{array}
$$

##### Loss Function

$$
\begin{array}{ll}
J(W, b) = {\displaystyle \underbrace{\sum\_{i=1}^{\vert \Omega \rvert} \sum\_{1 \leq p, q\leq l} \left\Vert f^{\frac{n\_l}{2}}(x\_p^i) - f^{\frac{n\_l}{2}}(x\_q^i) \right\rVert\_2^2 }\_{\text{Clique Embedding Loss}}} + {\displaystyle \underbrace{\frac{\gamma}{2}\sum\_{i=1}^{\vert\Omega\rvert}\sum\_{p=1}^l \left\Vert f^{n\_l}(x\_p^i) - x\_p^i \right\rVert\_2^2}\_{\text{Reconstruction Error}}} + {\displaystyle \underbrace{\beta \sum\_{l=1}^{n\_l-1}\sum\_j \text{KL}(\rho || \hat{\rho}\_j^l)}\_{\text{Spasity Constraint}}} + {\displaystyle \underbrace{\frac{\lambda}{2}\sum\_{l=1}^{n\_l}\left\Vert W^l \right\rVert\_F^2}\_{\text{Weight Decay}}}
\end{array} \tag{2}
$$

##### Algorithm for Clique Embedding

{{< figure src="algorithm-1.png" caption="Clique Embedding Algorithm" >}}

## Results

### Settings
- 潜在空間の分散表現を用いて，ノードまたはエッジに変化が生じた際にそれらが異常かどうかを判定する
- ノード・エッジの変化においては，2つのシナリオがあり得る
  - 異常な変化である場合
  - 新しいクラスターの最初の要素である場合

{{< box-with-title title="異常検知手順" >}}
1. 新しいノードに対して，最も近いクラスターとの距離を求める
2. 異常度スコアは最も近いクラスターとのユークリッド距離で定義される

$$
  \text{Anomaly Score} = \Vert c - f(\cdot) \rVert\_2
$$

3. 新しいノード $\lbrace \boldsymbol{x}'\_i \rbrace \_{i-1}^{n\_0}$ が新しいタイムステップで出現したとき，下記のようにクラスターの中心を更新する

$$
\begin{array}{ll}
  \begin{array}{ll}
    {\displaystyle c = \frac{\alpha \boldsymbol{c}\_0 n\_0 + (1 - \alpha) \sum\_{i=1}^{n'}{\boldsymbol{x}'}\_i}{\alpha n\_0 + (1 - \alpha)n'}}
  \end{array} \\\\
  \text{where} \hspace{10pt} \left\lbrace\begin{array}{ll}
    \alpha &\mapsto \text{parameter to control decay of estimates} \\\\
    c\_0 &\mapsto \text{the previous cluster center}
  \end{array}\right .
\end{array}
$$

{{< /box-with-title >}}

### Results

{{< figure src="results.png" caption="Anomaly Detection Performance Comparison" >}}

## References


{{< ci-details summary="Dynamic Network Embedding by Modeling Triadic Closure Process (Le-kui Zhou et al., 2018)">}}
Le-kui Zhou, Yang Yang, Xiang Ren, Fei Wu, Yueting Zhuang. (2018)  
**Dynamic Network Embedding by Modeling Triadic Closure Process**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/7bfb46c47c25e46a5f7b168133f4e926ab44725b)  
Influential Citation Count (43), SS-ID (7bfb46c47c25e46a5f7b168133f4e926ab44725b)  
**ABSTRACT**  
Network embedding, which aims to learn the low-dimensional representations of vertices, is an important task and has attracted considerable research efforts recently. In real world, networks, like social network and biological networks, are dynamic and evolving over time. However, almost all the existing network embedding methods focus on static networks while ignore network dynamics. In this paper, we present a novel representation learning approach, DynamicTriad, to preserve both structural information and evolution patterns of a given network. The general idea of our approach is to impose triad, which is a group of three vertices and is one of the basic units of networks. In particular, we model how a closed triad, which consists of three vertices connected with each other, develops from an open triad that has two of three vertices not connected with each other. This triadic closure process is a fundamental mechanism in the formation and evolution of networks, thereby makes our model being able to capture the network dynamics and to learn representation vectors for each vertex at different time steps. Experimental results on three real-world networks demonstrate that, compared with several state-of-the-art techniques, DynamicTriad achieves substantial gains in several application scenarios. For example, our approach can effectively be applied and help to identify telephone frauds in a mobile network, and to predict whether a user will repay her loans or not in a loan network.
{{< /ci-details >}}
{{< ci-details summary="Cooperative Game Theory Approaches for Network Partitioning (K. Avrachenkov et al., 2017)">}}
K. Avrachenkov, Aleksei Y. Kondratev, V. Mazalov. (2017)  
**Cooperative Game Theory Approaches for Network Partitioning**  
COCOON  
[Paper Link](https://www.semanticscholar.org/paper/a7acbafe0853f0db2f40c4819bb6f5161008446f)  
Influential Citation Count (10), SS-ID (a7acbafe0853f0db2f40c4819bb6f5161008446f)  
**ABSTRACT**  
The paper is devoted to game-theoretic methods for community detection in networks. The traditional methods for detecting community structure are based on selecting denser subgraphs inside the network. Here we propose to use the methods of cooperative game theory that highlight not only the link density but also the mechanisms of cluster formation. Specifically, we suggest two approaches from cooperative game theory: the first approach is based on the Myerson value, whereas the second approach is based on hedonic games. Both approaches allow to detect clusters with various resolution. However, the tuning of the resolution parameter in the hedonic games approach is particularly intuitive. Furthermore, the modularity based approach and its generalizations can be viewed as particular cases of the hedonic games.
{{< /ci-details >}}
{{< ci-details summary="Temporally Factorized Network Modeling for Evolutionary Network Analysis (Wenchao Yu et al., 2017)">}}
Wenchao Yu, C. Aggarwal, Wei Wang. (2017)  
**Temporally Factorized Network Modeling for Evolutionary Network Analysis**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/5ea4c598a154f6f0498d01a420bcca54285eb92f)  
Influential Citation Count (9), SS-ID (5ea4c598a154f6f0498d01a420bcca54285eb92f)  
**ABSTRACT**  
The problem of evolutionary network analysis has gained increasing attention in recent years, because of an increasing number of networks, which are encountered in temporal settings. For example, social networks, communication networks, and information networks continuously evolve over time, and it is desirable to learn interesting trends about how the network structure evolves over time, and in terms of other interesting trends. One challenging aspect of networks is that they are inherently resistant to parametric modeling, which allows us to truly express the edges in the network as functions of time. This is because, unlike multidimensional data, the edges in the network reflect interactions among nodes, and it is difficult to independently model the edge as a function of time, without taking into account its correlations and interactions with neighboring edges. Fortunately, we show that it is indeed possible to achieve this goal with the use of a matrix factorization, in which the entries are parameterized by time. This approach allows us to represent the edge structure of the network purely as a function of time, and predict the evolution of the network over time. This opens the possibility of using the approach for a wide variety of temporal network analysis problems, such as predicting future trends in structures, predicting links, and node-centric anomaly/event detection. This flexibility is because of the general way in which the approach allows us to express the structure of the network as a function of time. We present a number of experimental results on a number of temporal data sets showing the effectiveness of the approach.
{{< /ci-details >}}
{{< ci-details summary="Structural Deep Network Embedding (Daixin Wang et al., 2016)">}}
Daixin Wang, Peng Cui, Wenwu Zhu. (2016)  
**Structural Deep Network Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/d0b7c8828f0fca4dd901674e8fb5bd464a187664)  
Influential Citation Count (257), SS-ID (d0b7c8828f0fca4dd901674e8fb5bd464a187664)  
**ABSTRACT**  
Network embedding is an important method to learn low-dimensional representations of vertexes in networks, aiming to capture and preserve the network structure. Almost all the existing network embedding methods adopt shallow models. However, since the underlying network structure is complex, shallow models cannot capture the highly non-linear network structure, resulting in sub-optimal network representations. Therefore, how to find a method that is able to effectively capture the highly non-linear network structure and preserve the global and local structure is an open yet important problem. To solve this problem, in this paper we propose a Structural Deep Network Embedding method, namely SDNE. More specifically, we first propose a semi-supervised deep model, which has multiple layers of non-linear functions, thereby being able to capture the highly non-linear network structure. Then we propose to exploit the first-order and second-order proximity jointly to preserve the network structure. The second-order proximity is used by the unsupervised component to capture the global network structure. While the first-order proximity is used as the supervised information in the supervised component to preserve the local network structure. By jointly optimizing them in the semi-supervised deep model, our method can preserve both the local and global network structure and is robust to sparse networks. Empirically, we conduct the experiments on five real-world networks, including a language network, a citation network and three social networks. The results show that compared to the baselines, our method can reconstruct the original network significantly better and achieves substantial gains in three applications, i.e. multi-label classification, link prediction and visualization.
{{< /ci-details >}}
{{< ci-details summary="Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing Correlations (Wei Cheng et al., 2016)">}}
Wei Cheng, Kai Zhang, Haifeng Chen, G. Jiang, Zhengzhang Chen, Wei Wang. (2016)  
**Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing Correlations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/6ad11e44d7b85b1a01efe28319b4cc9de1154c88)  
Influential Citation Count (1), SS-ID (6ad11e44d7b85b1a01efe28319b4cc9de1154c88)  
**ABSTRACT**  
Modern world has witnessed a dramatic increase in our ability to collect, transmit and distribute real-time monitoring and surveillance data from large-scale information systems and cyber-physical systems. Detecting system anomalies thus attracts significant amount of interest in many fields such as security, fault management, and industrial optimization. Recently, invariant network has shown to be a powerful way in characterizing complex system behaviours. In the invariant network, a node represents a system component and an edge indicates a stable, significant interaction between two components. Structures and evolutions of the invariance network, in particular the vanishing correlations, can shed important light on locating causal anomalies and performing diagnosis. However, existing approaches to detect causal anomalies with the invariant network often use the percentage of vanishing correlations to rank possible casual components, which have several limitations: 1) fault propagation in the network is ignored; 2) the root casual anomalies may not always be the nodes with a high-percentage of vanishing correlations; 3) temporal patterns of vanishing correlations are not exploited for robust detection. To address these limitations, in this paper we propose a network diffusion based framework to identify significant causal anomalies and rank them. Our approach can effectively model fault propagation over the entire invariant network, and can perform joint inference on both the structural, and the time-evolving broken invariance patterns. As a result, it can locate high-confidence anomalies that are truly responsible for the vanishing correlations, and can compensate for unstructured measurement noise in the system. Extensive experiments on synthetic datasets, bank information system datasets, and coal plant cyber-physical system datasets demonstrate the effectiveness of our approach.
{{< /ci-details >}}
{{< ci-details summary="node2vec: Scalable Feature Learning for Networks (Aditya Grover et al., 2016)">}}
Aditya Grover, J. Leskovec. (2016)  
**node2vec: Scalable Feature Learning for Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/36ee2c8bd605afd48035d15fdc6b8c8842363376)  
Influential Citation Count (1192), SS-ID (36ee2c8bd605afd48035d15fdc6b8c8842363376)  
**ABSTRACT**  
Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.
{{< /ci-details >}}
{{< ci-details summary="A Scalable Approach for Outlier Detection in Edge Streams Using Sketch-based Approximations (Stephen Ranshous et al., 2016)">}}
Stephen Ranshous, Steve Harenberg, Kshitij Sharma, N. Samatova. (2016)  
**A Scalable Approach for Outlier Detection in Edge Streams Using Sketch-based Approximations**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/b66dbe242c3fa0a5738c7e76004f1f7b655f5ed6)  
Influential Citation Count (5), SS-ID (b66dbe242c3fa0a5738c7e76004f1f7b655f5ed6)  
**ABSTRACT**  
Dynamic graphs are a powerful way to model an evolving set of objects and their ongoing interactions. A broad spectrum of systems, such as information, communication, and social, are naturally represented by dynamic graphs. Outlier (or anomaly) detection in dynamic graphs can provide unique insights into the relationships of objects and identify novel or emerging relationships. To date, outlier detection in dynamic graphs has been studied in the context of graph streams, focusing on the analysis and comparison of entire graph objects. However, the volume and velocity of data are necessitating a transition from outlier detection in the context of graph streams to outlier detection in the context of edge streams–where the stream consists of individual graph edges instead of entire graph objects. In this paper, we propose the first approach for outlier detection in edge streams. We first describe a highlevel model for outlier detection based on global and local structural properties of a stream. We propose a novel application of the Count-Min sketch for approximating these properties, and prove probabilistic error bounds on our edge outlier scoring functions. Our sketch-based implementation provides a scalable solution, having constant time updates and constant space requirements. Experiments on synthetic and real world datasets demonstrate our method’s scalability, effectiveness for discovering outliers, and the effects of approximation.
{{< /ci-details >}}
{{< ci-details summary="Fast Memory-efficient Anomaly Detection in Streaming Heterogeneous Graphs (Emaad Manzoor et al., 2016)">}}
Emaad Manzoor, Sadegh Momeni, V. Venkatakrishnan, L. Akoglu. (2016)  
**Fast Memory-efficient Anomaly Detection in Streaming Heterogeneous Graphs**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/87f42406de084e60d2365adac8a159ed3e455856)  
Influential Citation Count (12), SS-ID (87f42406de084e60d2365adac8a159ed3e455856)  
**ABSTRACT**  
Given a stream of heterogeneous graphs containing different types of nodes and edges, how can we spot anomalous ones in real-time while consuming bounded memory? This problem is motivated by and generalizes from its application in security to host-level advanced persistent threat (APT) detection. We propose StreamSpot, a clustering based anomaly detection approach that addresses challenges in two key fronts: (1) heterogeneity, and (2) streaming nature. We introduce a new similarity function for heterogeneous graphs that compares two graphs based on their relative frequency of local substructures, represented as short strings. This function lends itself to a vector representation of a graph, which is (a) fast to compute, and (b) amenable to a sketched version with bounded size that preserves similarity. StreamSpot exhibits desirable properties that a streaming application requires: it is (i) fully-streaming; processing the stream one edge at a time as it arrives, (ii) memory-efficient; requiring constant space for the sketches and the clustering, (iii) fast; taking constant time to update the graph sketches and the cluster summaries that can process over 100,000 edges per second, and (iv) online; scoring and flagging anomalies in real time. Experiments on datasets containing simulated system-call flow graphs from normal browser activity and various attack scenarios (ground truth) show that StreamSpot is high-performance; achieving above 95% detection accuracy with small delay, as well as competitive time and memory usage.
{{< /ci-details >}}
{{< ci-details summary="Scalable Anomaly Ranking of Attributed Neighborhoods (Bryan Perozzi et al., 2016)">}}
Bryan Perozzi, L. Akoglu. (2016)  
**Scalable Anomaly Ranking of Attributed Neighborhoods**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/99ac4b8d3c8790a50a468d8268cff00651cb65b6)  
Influential Citation Count (16), SS-ID (99ac4b8d3c8790a50a468d8268cff00651cb65b6)  
**ABSTRACT**  
Given a graph with node attributes, what neighborhoods are anomalous? To answer this question, one needs a quality score that utilizes both structure and attributes. Popular existing measures either quantify the structure only and ignore the attributes (e.g., conductance), or only consider the connectedness of the nodes inside the neighborhood and ignore the cross-edges at the boundary (e.g., density).  In this work we propose normality, a new quality measure for attributed neighborhoods. Normality utilizes structure and attributes together to quantify both internal consistency and external separability. It exhibits two key advantages over other measures: (1) It allows many boundary-edges as long as they can be "exonerated"; i.e., either (i) are expected under a null model, and/or (ii) the boundary nodes do not exhibit the subset of attributes shared by the neighborhood members. Existing measures, in contrast, penalize boundary edges irrespectively. (2) Normality can be efficiently maximized to automatically infer the shared attribute subspace (and respective weights) that characterize a neighborhood. This efficient optimization allows us to process graphs with millions of attributes.  We capitalize on our measure to present a novel approach for Anomaly Mining of Entity Neighborhoods (AMEN). Experiments on real-world attributed graphs illustrate the effectiveness of our measure at anomaly detection, outperforming popular approaches including conductance, density, OddBall, and SODA. In addition to anomaly detection, our qualitative analysis demonstrates the utility of normality as a powerful tool to contrast the correlation between structure and attributes across different graphs.
{{< /ci-details >}}
{{< ci-details summary="Heterogeneous Network Embedding via Deep Architectures (Shiyu Chang et al., 2015)">}}
Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, C. Aggarwal, Thomas S. Huang. (2015)  
**Heterogeneous Network Embedding via Deep Architectures**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
Influential Citation Count (40), SS-ID (f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
**ABSTRACT**  
Data embedding is used in many machine learning applications to create low-dimensional feature representations, which preserves the structure of data points in their original space. In this paper, we examine the scenario of a heterogeneous network with nodes and content of various types. Such networks are notoriously difficult to mine because of the bewildering combination of heterogeneous contents and structures. The creation of a multidimensional embedding of such data opens the door to the use of a wide variety of off-the-shelf mining techniques for multidimensional data. Despite the importance of this problem, limited efforts have been made on embedding a network of scalable, dynamic and heterogeneous data. In such cases, both the content and linkage structure provide important cues for creating a unified feature representation of the underlying network. In this paper, we design a deep embedding algorithm for networked data. A highly nonlinear multi-layered embedding function is used to capture the complex interactions between the heterogeneous data in a network. Our goal is to create a multi-resolution deep embedding function, that reflects both the local and global network structures, and makes the resulting embedding useful for a variety of data mining tasks. In particular, we demonstrate that the rich content and linkage information in a heterogeneous network can be captured by such an approach, so that similarities among cross-modal data can be measured directly in a common embedding space. Once this goal has been achieved, a wide variety of data mining problems can be solved by applying off-the-shelf algorithms designed for handling vector representations. Our experiments on real-world network datasets show the effectiveness and scalability of the proposed algorithm as compared to the state-of-the-art embedding methods.
{{< /ci-details >}}
{{< ci-details summary="Anomaly detection in dynamic networks: a survey (Stephen Ranshous et al., 2015)">}}
Stephen Ranshous, Shitian Shen, Danai Koutra, Steve Harenberg, C. Faloutsos, N. Samatova. (2015)  
**Anomaly detection in dynamic networks: a survey**  
  
[Paper Link](https://www.semanticscholar.org/paper/23f9220e0ca8f2cb522c36d158ca88f043007c68)  
Influential Citation Count (14), SS-ID (23f9220e0ca8f2cb522c36d158ca88f043007c68)  
**ABSTRACT**  
Anomaly detection is an important problem with multiple applications, and thus has been studied for decades in various research domains. In the past decade there has been a growing interest in anomaly detection in data represented as networks, or graphs, largely because of their robust expressiveness and their natural ability to represent complex relationships. Originally, techniques focused on anomaly detection in static graphs, which do not change and are capable of representing only a single snapshot of data. As real‐world networks are constantly changing, there has been a shift in focus to dynamic graphs, which evolve over time.
{{< /ci-details >}}
{{< ci-details summary="Vertex Clustering of Augmented Graph Streams (Ryan McConville et al., 2015)">}}
Ryan McConville, Weiru Liu, P. Miller. (2015)  
**Vertex Clustering of Augmented Graph Streams**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/330c7ba2b4d57272451c08863a9dabb1bca94876)  
Influential Citation Count (0), SS-ID (330c7ba2b4d57272451c08863a9dabb1bca94876)  
**ABSTRACT**  
In this paper we propose a graph stream clustering algorithm with a unified similarity measure on both structural and attribute properties of vertices, with each attribute being treated as a vertex. Unlike others, our approach does not require an input parameter for the number of clusters, instead, it dynamically creates new sketch-based clusters and periodically merges existing similar clusters. Experiments on two publicly available datasets reveal the advantages of our approach in detecting vertex clusters in the graph stream. We provide a detailed investigation into how parameters affect the algorithm performance. We also provide a quantitative evaluation and comparison with a well-known offline community detection algorithm which shows that our streaming algorithm can achieve comparable or better average cluster purity.
{{< /ci-details >}}
{{< ci-details summary="LINE: Large-scale Information Network Embedding (Jian Tang et al., 2015)">}}
Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Q. Mei. (2015)  
**LINE: Large-scale Information Network Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
Influential Citation Count (901), SS-ID (0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
**ABSTRACT**  
This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for real world information networks which usually contain millions of nodes. In this paper, we propose a novel network embedding method called the ``LINE,'' which is suitable for arbitrary types of information networks: undirected, directed, and/or weighted. The method optimizes a carefully designed objective function that preserves both the local and global network structures. An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the effectiveness and the efficiency of the inference. Empirical experiments prove the effectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very efficient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online\footnote{\url{https://github.com/tangjianpku/LINE}}.
{{< /ci-details >}}
{{< ci-details summary="Neural Word Embedding as Implicit Matrix Factorization (Omer Levy et al., 2014)">}}
Omer Levy, Yoav Goldberg. (2014)  
**Neural Word Embedding as Implicit Matrix Factorization**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/f4c018bcc8ea707b83247866bdc8ccb87cd9f5da)  
Influential Citation Count (190), SS-ID (f4c018bcc8ea707b83247866bdc8ccb87cd9f5da)  
**ABSTRACT**  
We analyze skip-gram with negative-sampling (SGNS), a word embedding method introduced by Mikolov et al., and show that it is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs, shifted by a global constant. We find that another embedding method, NCE, is implicitly factorizing a similar matrix, where each cell is the (shifted) log conditional probability of a word given its context. We show that using a sparse Shifted Positive PMI word-context matrix to represent words improves results on two word similarity tasks and one of two analogy tasks. When dense low-dimensional vectors are preferred, exact factorization with SVD can achieve solutions that are at least as good as SGNS's solutions for word similarity tasks. On analogy questions SGNS remains superior to SVD. We conjecture that this stems from the weighted nature of SGNS's factorization.
{{< /ci-details >}}
{{< ci-details summary="Outlier Detection for Temporal Data: A Survey (Manish Gupta et al., 2014)">}}
Manish Gupta, Jing Gao, C. Aggarwal, Jiawei Han. (2014)  
**Outlier Detection for Temporal Data: A Survey**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/43d75d3a22db904d052d4c435e2d1f22be3887e0)  
Influential Citation Count (30), SS-ID (43d75d3a22db904d052d4c435e2d1f22be3887e0)  
**ABSTRACT**  
In the statistics community, outlier detection for time series data has been studied for decades. Recently, with advances in hardware and software technology, there has been a large body of work on temporal outlier detection from a computational perspective within the computer science community. In particular, advances in hardware technology have enabled the availability of various forms of temporal data collection mechanisms, and advances in software technology have enabled a variety of data management mechanisms. This has fueled the growth of different kinds of data sets such as data streams, spatio-temporal data, distributed streams, temporal networks, and time series data, generated by a multitude of applications. There arises a need for an organized and detailed study of the work done in the area of outlier detection with respect to such temporal datasets. In this survey, we provide a comprehensive and structured overview of a large set of interesting outlier definitions for various forms of temporal data, novel techniques, and application scenarios in which specific definitions and techniques have been widely used.
{{< /ci-details >}}
{{< ci-details summary="Focused clustering and outlier detection in large attributed graphs (Bryan Perozzi et al., 2014)">}}
Bryan Perozzi, L. Akoglu, Patricia Iglesias Sánchez, Emmanuel Müller. (2014)  
**Focused clustering and outlier detection in large attributed graphs**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fcc4c7c2dfaf08365c5ecf785d0df3f6f243bd1a)  
Influential Citation Count (21), SS-ID (fcc4c7c2dfaf08365c5ecf785d0df3f6f243bd1a)  
**ABSTRACT**  
Graph clustering and graph outlier detection have been studied extensively on plain graphs, with various applications. Recently, algorithms have been extended to graphs with attributes as often observed in the real-world. However, all of these techniques fail to incorporate the user preference into graph mining, and thus, lack the ability to steer algorithms to more interesting parts of the attributed graph. In this work, we overcome this limitation and introduce a novel user-oriented approach for mining attributed graphs. The key aspect of our approach is to infer user preference by the so-called focus attributes through a set of user-provided exemplar nodes. In this new problem setting, clusters and outliers are then simultaneously mined according to this user preference. Specifically, our FocusCO algorithm identifies the focus, extracts focused clusters and detects outliers. Moreover, FocusCO scales well with graph size, since we perform a local clustering of interest to the user rather than global partitioning of the entire graph. We show the effectiveness and scalability of our method on synthetic and real-world graphs, as compared to both existing graph clustering and outlier detection approaches.
{{< /ci-details >}}
{{< ci-details summary="Learning Deep Representations for Graph Clustering (Fei Tian et al., 2014)">}}
Fei Tian, Bin Gao, Qing Cui, Enhong Chen, Tie-Yan Liu. (2014)  
**Learning Deep Representations for Graph Clustering**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/df787a974fff59f557ed1ec620fc345568aec491)  
Influential Citation Count (38), SS-ID (df787a974fff59f557ed1ec620fc345568aec491)  
**ABSTRACT**  
Recently deep learning has been successfully adopted in many applications such as speech recognition and image classification. In this work, we explore the possibility of employing deep learning in graph clustering. We propose a simple method, which first learns a nonlinear embedding of the original graph by stacked autoencoder, and then runs $k$-means algorithm on the embedding to obtain the clustering result. We show that this simple method has solid theoretical foundation, due to the similarity between autoencoder and spectral clustering in terms of what they actually optimize. Then, we demonstrate that the proposed method is more efficient and flexible than spectral clustering. First, the computational complexity of autoencoder is much lower than spectral clustering: the former can be linear to the number of nodes in a sparse graph while the latter is super quadratic due to eigenvalue decomposition. Second, when additional sparsity constraint is imposed, we can simply employ the sparse autoencoder developed in the literature of deep learning; however, it is non-straightforward to implement a sparse spectral method. The experimental results on various graph datasets show that the proposed method significantly outperforms conventional spectral clustering which clearly indicates the effectiveness of deep learning in graph clustering.
{{< /ci-details >}}
{{< ci-details summary="SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives (Aaron Defazio et al., 2014)">}}
Aaron Defazio, F. Bach, S. Lacoste-Julien. (2014)  
**SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/4daec165c1f4aa1206b0d91c0b26f0287d1ef52d)  
Influential Citation Count (294), SS-ID (4daec165c1f4aa1206b0d91c0b26f0287d1ef52d)  
**ABSTRACT**  
In this work we introduce a new optimisation method called SAGA in the spirit of SAG, SDCA, MISO and SVRG, a set of recently proposed incremental gradient algorithms with fast linear convergence rates. SAGA improves on the theory behind SAG and SVRG, with better theoretical convergence rates, and has support for composite objectives where a proximal operator is used on the regulariser. Unlike SDCA, SAGA supports non-strongly convex problems directly, and is adaptive to any inherent strong convexity of the problem. We give experimental results showing the effectiveness of our method.
{{< /ci-details >}}
{{< ci-details summary="Graph based anomaly detection and description: a survey (L. Akoglu et al., 2014)">}}
L. Akoglu, Hanghang Tong, Danai Koutra. (2014)  
**Graph based anomaly detection and description: a survey**  
Data Mining and Knowledge Discovery  
[Paper Link](https://www.semanticscholar.org/paper/ae74522002f9093fbf63a20efb57d80ee2f4f564)  
Influential Citation Count (43), SS-ID (ae74522002f9093fbf63a20efb57d80ee2f4f564)  
{{< /ci-details >}}
{{< ci-details summary="DeepWalk: online learning of social representations (Bryan Perozzi et al., 2014)">}}
Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2014)  
**DeepWalk: online learning of social representations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fff114cbba4f3ba900f33da574283e3de7f26c83)  
Influential Citation Count (1442), SS-ID (fff114cbba4f3ba900f33da574283e3de7f26c83)  
**ABSTRACT**  
We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.
{{< /ci-details >}}
{{< ci-details summary="Local Learning for Mining Outlier Subgraphs from Network Datasets (Manish Gupta et al., 2014)">}}
Manish Gupta, Arun Mallya, Subhro Roy, Jason H. D. Cho, Jiawei Han. (2014)  
**Local Learning for Mining Outlier Subgraphs from Network Datasets**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/2d7fcca1b0bde8e3f0450f4c8e67e6cbf519bff1)  
Influential Citation Count (2), SS-ID (2d7fcca1b0bde8e3f0450f4c8e67e6cbf519bff1)  
**ABSTRACT**  
In the real world, various systems can be modeled using entity-relationship graphs. Given such a graph, one may be interested in identifying suspicious or anomalous subgraphs. Specifically, a user may want to identify suspicious subgraphs matching a query template. A subgraph can be defined as anomalous based on the connectivity structure within itself as well as with its neighborhood. For example for a co-authorship network, given a subgraph containing three authors, one expects all three authors to be say data mining authors. Also, one expects the neighborhood to mostly consist of data mining authors. But a 3-author clique of data mining authors with all theory authors in the neighborhood clearly seems interesting. Similarly, having one of the authors in the clique as a theory author when all other authors (both in the clique and neighborhood) are data mining authors, is also suspicious. Thus, existence of lowprobability links and absence of high-probability links can be a good indicator of subgraph outlierness. The probability of an edge can in turn be modeled based on the weighted similarity between the attribute values of the nodes linked by the edge. We claim that the attribute weights must be learned locally for accurate link existence probability computations. In this paper, we design a system that finds subgraph outliers given a graph and a query by modeling the problem as a linear optimization. Experimental results on several synthetic and real datasets show the effectiveness of the proposed approach in computing interesting outliers.
{{< /ci-details >}}
{{< ci-details summary="Distributed Representations of Words and Phrases and their Compositionality (Tomas Mikolov et al., 2013)">}}
Tomas Mikolov, Ilya Sutskever, Kai Chen, G. Corrado, J. Dean. (2013)  
**Distributed Representations of Words and Phrases and their Compositionality**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/87f40e6f3022adbc1f1905e3e506abad05a9964f)  
Influential Citation Count (3802), SS-ID (87f40e6f3022adbc1f1905e3e506abad05a9964f)  
**ABSTRACT**  
The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling.    An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of "Canada" and "Air" cannot be easily combined to obtain "Air Canada". Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible.
{{< /ci-details >}}
{{< ci-details summary="Embedding with Autoencoder Regularization (Wenchao Yu et al., 2013)">}}
Wenchao Yu, Guangxiang Zeng, Ping Luo, Fuzhen Zhuang, Qing He, Zhongzhi Shi. (2013)  
**Embedding with Autoencoder Regularization**  
ECML/PKDD  
[Paper Link](https://www.semanticscholar.org/paper/c5db1487d8bb10a6b4c6934ed7d9ec36381c3609)  
Influential Citation Count (3), SS-ID (c5db1487d8bb10a6b4c6934ed7d9ec36381c3609)  
{{< /ci-details >}}
{{< ci-details summary="Anomaly, event, and fraud detection in large network datasets (L. Akoglu et al., 2013)">}}
L. Akoglu, C. Faloutsos. (2013)  
**Anomaly, event, and fraud detection in large network datasets**  
WSDM '13  
[Paper Link](https://www.semanticscholar.org/paper/a520e1ec9cea03e125af46c53f9aea2df9848979)  
Influential Citation Count (7), SS-ID (a520e1ec9cea03e125af46c53f9aea2df9848979)  
**ABSTRACT**  
Detecting anomalies and events in data is a vital task, with numerous applications in security, finance, health care, law enforcement, and many others. While many techniques have been developed in past years for spotting outliers and anomalies in unstructured collections of multi-dimensional points, with graph data becoming ubiquitous, techniques for structured graph data have been of focus recently. As objects in graphs have long-range correlations, novel technology has been developed for abnormality detection in graph data.  The goal of this tutorial is to provide a general, comprehensive overview of the state-of-the-art methods for anomaly, event, and fraud detection in data represented as graphs. As a key contribution, we provide a thorough exploration of both data mining and machine learning algorithms for these detection tasks. We give a general framework for the algorithms, categorized under various settings: unsupervised vs.(semi-)supervised, for static vs. dynamic data. We focus on the scalability and effectiveness aspects of the methods, and highlight results on crucial real-world applications, including accounting fraud and opinion spam detection.
{{< /ci-details >}}
{{< ci-details summary="On Graph Stream Clustering with Side Information (Philip S. Yu et al., 2013)">}}
Philip S. Yu, Yuchen Zhao. (2013)  
**On Graph Stream Clustering with Side Information**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/43cc27f931247bd4d19c5437244ef3a4f3f0aae9)  
Influential Citation Count (0), SS-ID (43cc27f931247bd4d19c5437244ef3a4f3f0aae9)  
**ABSTRACT**  
Graph clustering becomes an important problem due to emerging applications involving the web, social networks and bio-informatics. Recently, many such applications generate data in the form of streams. Clustering massive, dynamic graph streams is significantly challenging because of the complex structures of graphs and computational difficulties of continuous data. Meanwhile, a large volume of side information is associated with graphs, which can be of various types. The examples include the properties of users in social network activities, the meta attributes associated with web click graph streams and the location information in mobile communication networks. Such attributes contain extremely useful information and has the potential to improve the clustering process, but are neglected by most recent graph stream mining techniques. In this paper, we define a unified distance measure on both link structures and side attributes for clustering. In addition, we propose a novel optimization framework DMO, which can dynamically optimize the distance metric and make it adapt to the newly received stream data. We further introduce a carefully designed statistics SGS(C) which consume constant storage spaces with the progression of streams. We demonstrate that the statistics maintained are sufficient for the clustering process as well as the distance optimization and can be scalable to massive graphs with side attributes. We will present experiment results to show the advantages of the approach in graph stream clustering with both links and side information over the baselines.
{{< /ci-details >}}
{{< ci-details summary="Efficient Estimation of Word Representations in Vector Space (Tomas Mikolov et al., 2013)">}}
Tomas Mikolov, Kai Chen, G. Corrado, J. Dean. (2013)  
**Efficient Estimation of Word Representations in Vector Space**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/330da625c15427c6e42ccfa3b747fb29e5835bf0)  
Influential Citation Count (3613), SS-ID (330da625c15427c6e42ccfa3b747fb29e5835bf0)  
**ABSTRACT**  
We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.
{{< /ci-details >}}
{{< ci-details summary="Outlier Analysis (C. Aggarwal, 2013)">}}
C. Aggarwal. (2013)  
**Outlier Analysis**  
Springer New York  
[Paper Link](https://www.semanticscholar.org/paper/1bc042ec7a58ca8040ee08178433752f2c16f25e)  
Influential Citation Count (90), SS-ID (1bc042ec7a58ca8040ee08178433752f2c16f25e)  
{{< /ci-details >}}
{{< ci-details summary="Community Trend Outlier Detection Using Soft Temporal Pattern Mining (Manish Gupta et al., 2012)">}}
Manish Gupta, Jing Gao, Yizhou Sun, Jiawei Han. (2012)  
**Community Trend Outlier Detection Using Soft Temporal Pattern Mining**  
ECML/PKDD  
[Paper Link](https://www.semanticscholar.org/paper/ccdc80952500de68b3f9d105d311ef86c8bd7d4a)  
Influential Citation Count (3), SS-ID (ccdc80952500de68b3f9d105d311ef86c8bd7d4a)  
{{< /ci-details >}}
{{< ci-details summary="Integrating community matching and outlier detection for mining evolutionary community outliers (Manish Gupta et al., 2012)">}}
Manish Gupta, Jing Gao, Yizhou Sun, Jiawei Han. (2012)  
**Integrating community matching and outlier detection for mining evolutionary community outliers**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/8af4dab6c1bf208743966bebe2d88692649d70dc)  
Influential Citation Count (9), SS-ID (8af4dab6c1bf208743966bebe2d88692649d70dc)  
**ABSTRACT**  
Temporal datasets, in which data evolves continuously, exist in a wide variety of applications, and identifying anomalous or outlying objects from temporal datasets is an important and challenging task. Different from traditional outlier detection, which detects objects that have quite different behavior compared with the other objects, temporal outlier detection tries to identify objects that have different evolutionary behavior compared with other objects. Usually objects form multiple communities, and most of the objects belonging to the same community follow similar patterns of evolution. However, there are some objects which evolve in a very different way relative to other community members, and we define such objects as evolutionary community outliers. This definition represents a novel type of outliers considering both temporal dimension and community patterns. We investigate the problem of identifying evolutionary community outliers given the discovered communities from two snapshots of an evolving dataset. To tackle the challenges of community evolution and outlier detection, we propose an integrated optimization framework which conducts outlier-aware community matching across snapshots and identification of evolutionary outliers in a tightly coupled way. A coordinate descent algorithm is proposed to improve community matching and outlier detection performance iteratively. Experimental results on both synthetic and real datasets show that the proposed approach is highly effective in discovering interesting evolutionary community outliers.
{{< /ci-details >}}
{{< ci-details summary="Outlier detection in graph streams (C. Aggarwal et al., 2011)">}}
C. Aggarwal, Yuchen Zhao, Philip S. Yu. (2011)  
**Outlier detection in graph streams**  
2011 IEEE 27th International Conference on Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/f038cb7e8e4349937a4af5894ac565f20afae2f4)  
Influential Citation Count (5), SS-ID (f038cb7e8e4349937a4af5894ac565f20afae2f4)  
**ABSTRACT**  
A number of applications in social networks, telecommunications, and mobile computing create massive streams of graphs. In many such applications, it is useful to detect structural abnormalities which are different from the “typical” behavior of the underlying network. In this paper, we will provide first results on the problem of structural outlier detection in massive network streams. Such problems are inherently challenging, because the problem of outlier detection is specially challenging because of the high volume of the underlying network stream. The stream scenario also increases the computational challenges for the approach. We use a structural connectivity model in order to define outliers in graph streams. In order to handle the sparsity problem of massive networks, we dynamically partition the network in order to construct statistically robust models of the connectivity behavior. We design a reservoir sampling method in order to maintain structural summaries of the underlying network. These structural summaries are designed in order to create robust, dynamic and efficient models for outlier detection in graph streams. We present experimental results illustrating the effectiveness and efficiency of our approach.
{{< /ci-details >}}
{{< ci-details summary="On Clustering Graph Streams (C. Aggarwal et al., 2010)">}}
C. Aggarwal, Yuchen Zhao, Philip S. Yu. (2010)  
**On Clustering Graph Streams**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/c4688c0856f501f4250808bfa80175bfbb039c2d)  
Influential Citation Count (1), SS-ID (c4688c0856f501f4250808bfa80175bfbb039c2d)  
**ABSTRACT**  
In this paper, we will examine the problem of clustering massive graph streams. Graph clustering poses significant challenges because of the complex structures which may be present in the underlying data. The massive size of the underlying graph makes explicit structural enumeration very difficult. Consequently, most techniques for clustering multi-dimensional data are difficult to generalize to the case of massive graphs. Recently, methods have been proposed for clustering graph data, though these methods are designed for static data, and are not applicable to the case of graph streams. Furthermore, these techniques are especially not effective for the case of massive graphs, since a huge number of distinct edges may need to be tracked simultaneously. This results in storage and computational challenges during the clustering process. In order to deal with the natural problems arising from the use of massive disk-resident graphs, we will propose a technique for creating hash-compressed micro-clusters from graph streams. The compressed micro-clusters are designed by using a hash-based compression of the edges onto a smaller domain space. We will provide theoretical results which show that the hash-based compression continues to maintain bounded accuracy in terms of distance computations. We will provide experimental results which illustrate the accuracy and efficiency of the underlying method.
{{< /ci-details >}}
{{< ci-details summary="On community outliers and their efficient detection in information networks (Jing Gao et al., 2010)">}}
Jing Gao, Feng Liang, Wei Fan, Chi Wang, Yizhou Sun, Jiawei Han. (2010)  
**On community outliers and their efficient detection in information networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/27cbe8f134065d3fc61234d822cdd86742471001)  
Influential Citation Count (26), SS-ID (27cbe8f134065d3fc61234d822cdd86742471001)  
**ABSTRACT**  
Linked or networked data are ubiquitous in many applications. Examples include web data or hypertext documents connected via hyperlinks, social networks or user profiles connected via friend links, co-authorship and citation information, blog data, movie reviews and so on. In these datasets (called "information networks"), closely related objects that share the same properties or interests form a community. For example, a community in blogsphere could be users mostly interested in cell phone reviews and news. Outlier detection in information networks can reveal important anomalous and interesting behaviors that are not obvious if community information is ignored. An example could be a low-income person being friends with many rich people even though his income is not anomalously low when considered over the entire population. This paper first introduces the concept of community outliers (interesting points or rising stars for a more positive sense), and then shows that well-known baseline approaches without considering links or community information cannot find these community outliers. We propose an efficient solution by modeling networked data as a mixture model composed of multiple normal communities and a set of randomly generated outliers. The probabilistic model characterizes both data and links simultaneously by defining their joint distribution based on hidden Markov random fields (HMRF). Maximizing the data likelihood and the posterior of the model gives the solution to the outlier inference problem. We apply the model on both synthetic data and DBLP data sets, and the results demonstrate importance of this concept, as well as the effectiveness and efficiency of the proposed approach.
{{< /ci-details >}}
{{< ci-details summary="oddball: Spotting Anomalies in Weighted Graphs (L. Akoglu et al., 2010)">}}
L. Akoglu, Mary McGlohon, C. Faloutsos. (2010)  
**oddball: Spotting Anomalies in Weighted Graphs**  
PAKDD  
[Paper Link](https://www.semanticscholar.org/paper/a8dc47e370b17371e57ad070e669360794473efe)  
Influential Citation Count (39), SS-ID (a8dc47e370b17371e57ad070e669360794473efe)  
{{< /ci-details >}}
{{< ci-details summary="Streaming k-means approximation (Nir Ailon et al., 2009)">}}
Nir Ailon, Ragesh Jaiswal, C. Monteleoni. (2009)  
**Streaming k-means approximation**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/600ab83b9ec48148a3a0428a02c78e47dd742d61)  
Influential Citation Count (17), SS-ID (600ab83b9ec48148a3a0428a02c78e47dd742d61)  
**ABSTRACT**  
We provide a clustering algorithm that approximately optimizes the k-means objective, in the one-pass streaming setting. We make no assumptions about the data, and our algorithm is very light-weight in terms of memory, and computation. This setting is applicable to unsupervised learning on massive data sets, or resource-constrained devices. The two main ingredients of our theoretical work are: a derivation of an extremely simple pseudo-approximation batch algorithm for k-means (based on the recent k-means++), in which the algorithm is allowed to output more than k centers, and a streaming clustering algorithm in which batch clustering algorithms are performed on small inputs (fitting in memory) and combined in a hierarchical manner. Empirical evaluations on real and simulated data reveal the practical utility of our method.
{{< /ci-details >}}
{{< ci-details summary="Clustering in weighted networks (Tore Opsahl et al., 2009)">}}
Tore Opsahl, P. Panzarasa. (2009)  
**Clustering in weighted networks**  
Soc. Networks  
[Paper Link](https://www.semanticscholar.org/paper/95f568b1dfc1078b6a6227f522c5406b2d52e949)  
Influential Citation Count (57), SS-ID (95f568b1dfc1078b6a6227f522c5406b2d52e949)  
{{< /ci-details >}}
{{< ci-details summary="Anomaly detection in data represented as graphs (W. Eberle et al., 2007)">}}
W. Eberle, L. Holder. (2007)  
**Anomaly detection in data represented as graphs**  
Intell. Data Anal.  
[Paper Link](https://www.semanticscholar.org/paper/3041c28370d3ffc7422fd6bc82be5f9e03ada1c9)  
Influential Citation Count (3), SS-ID (3041c28370d3ffc7422fd6bc82be5f9e03ada1c9)  
**ABSTRACT**  
An important area of data mining is anomaly detection, particularly for fraud. However, little work has been done in terms of detecting anomalies in data that is represented as a graph. In this paper we present graph-based approaches to uncovering anomalies in domains where the anomalies consist of unexpected entity/relationship alterations that closely resemble non-anomalous behavior. We have developed three algorithms for the purpose of detecting anomalies in all three types of possible graph changes: label modifications, vertex/edge insertions and vertex/edge deletions. Each of our algorithms focuses on one of these anomalous types, using the minimum description length principle to first discover the normative pattern. Once the common pattern is known, each algorithm then uses a different approach to discover particular anomalous types. In this paper, we validate all three approaches using synthetic data, verifying that each of the algorithms on graphs and anomalies of varying sizes, are able to detect the anomalies with very high detection rates and minimal false positives. We then further validate the algorithms using real-world cargo data and actual fraud scenarios injected into the data set with 100% accuracy and no false positives. Each of these algorithms demonstrates the usefulness of examining a graph-based representation of data for the purposes of detecting fraud.
{{< /ci-details >}}
{{< ci-details summary="A tutorial on spectral clustering (U. V. Luxburg, 2007)">}}
U. V. Luxburg. (2007)  
**A tutorial on spectral clustering**  
Stat. Comput.  
[Paper Link](https://www.semanticscholar.org/paper/eda90bd43f4256986688e525b45b833a3addab97)  
Influential Citation Count (852), SS-ID (eda90bd43f4256986688e525b45b833a3addab97)  
{{< /ci-details >}}
{{< ci-details summary="Graph evolution: Densification and shrinking diameters (J. Leskovec et al., 2006)">}}
J. Leskovec, J. Kleinberg, C. Faloutsos. (2006)  
**Graph evolution: Densification and shrinking diameters**  
TKDD  
[Paper Link](https://www.semanticscholar.org/paper/5929e6031115e3dfa4b4f12071f4e16b24a003e0)  
Influential Citation Count (200), SS-ID (5929e6031115e3dfa4b4f12071f4e16b24a003e0)  
**ABSTRACT**  
How do real graphs evolve over time? What are normal growth patterns in social, technological, and information networks? Many studies have discovered patterns in static graphs, identifying properties in a single snapshot of a large network or in a very small number of snapshots; these include heavy tails for in- and out-degree distributions, communities, small-world phenomena, and others. However, given the lack of information about network evolution over long periods, it has been hard to convert these findings into statements about trends over time.  Here we study a wide range of real graphs, and we observe some surprising phenomena. First, most of these graphs densify over time with the number of edges growing superlinearly in the number of nodes. Second, the average distance between nodes often shrinks over time in contrast to the conventional wisdom that such distance parameters should increase slowly as a function of the number of nodes (like O(log n) or O(log(log n)).  Existing graph generation models do not exhibit these types of behavior even at a qualitative level. We provide a new graph generator, based on a forest fire spreading process that has a simple, intuitive justification, requires very few parameters (like the flammability of nodes), and produces graphs exhibiting the full range of properties observed both in prior work and in the present study.  We also notice that the forest fire model exhibits a sharp transition between sparse graphs and graphs that are densifying. Graphs with decreasing distance between the nodes are generated around this transition point.  Last, we analyze the connection between the temporal evolution of the degree distribution and densification of a graph. We find that the two are fundamentally related. We also observe that real networks exhibit this type of relation between densification and the degree distribution.
{{< /ci-details >}}
{{< ci-details summary="Relevance search and anomaly detection in bipartite graphs (Jimeng Sun et al., 2005)">}}
Jimeng Sun, Huiming Qu, Deepayan Chakrabarti, C. Faloutsos. (2005)  
**Relevance search and anomaly detection in bipartite graphs**  
SKDD  
[Paper Link](https://www.semanticscholar.org/paper/e5f2c6d9ca09bfaf8f83523e47169444d7b1d9d7)  
Influential Citation Count (4), SS-ID (e5f2c6d9ca09bfaf8f83523e47169444d7b1d9d7)  
**ABSTRACT**  
Many real applications can be modeled using bipartite graphs, such as users vs. files in a P2P system, traders vs. stocks in a financial trading system, conferences vs. authors in a scientific publication network, and so on. We introduce two operations on bipartite graphs: 1) identifying similar nodes (relevance search), and 2) finding nodes connecting irrelevant nodes (anomaly detection). And we propose algorithms to compute the relevance score for each node using random walk with restarts and graph partitioning; we also propose algorithms to identify anomalies, using relevance scores. We evaluate the quality of relevance search based on semantics of the datasets, and we also measure the performance of the anomaly detection algorithm with manually injected anomalies. Both effectiveness and efficiency of the methods are confirmed by experiments on several real datasets.
{{< /ci-details >}}
{{< ci-details summary="Neighborhood formation and anomaly detection in bipartite graphs (Jimeng Sun et al., 2005)">}}
Jimeng Sun, Huiming Qu, Deepayan Chakrabarti, C. Faloutsos. (2005)  
**Neighborhood formation and anomaly detection in bipartite graphs**  
Fifth IEEE International Conference on Data Mining (ICDM'05)  
[Paper Link](https://www.semanticscholar.org/paper/e028820ff569c59338fb7df22484d2435e194d0b)  
Influential Citation Count (21), SS-ID (e028820ff569c59338fb7df22484d2435e194d0b)  
**ABSTRACT**  
Many real applications can be modeled using bipartite graphs, such as users vs. files in a P2P system, traders vs. stocks in a financial trading system, conferences vs. authors in a scientific publication network, and so on. We introduce two operations on bipartite graphs: 1) identifying similar nodes (Neighborhood formation), and 2) finding abnormal nodes (Anomaly detection). And we propose algorithms to compute the neighborhood for each node using random walk with restarts and graph partitioning; we also propose algorithms to identify abnormal nodes, using neighborhood information. We evaluate the quality of neighborhoods based on semantics of the datasets, and we also measure the performance of the anomaly detection algorithm with manually injected anomalies. Both effectiveness and efficiency of the methods are confirmed by experiments on several real datasets.
{{< /ci-details >}}
{{< ci-details summary="An improved data stream summary: the count-min sketch and its applications (Graham Cormode et al., 2004)">}}
Graham Cormode, S. Muthukrishnan. (2004)  
**An improved data stream summary: the count-min sketch and its applications**  
J. Algorithms  
[Paper Link](https://www.semanticscholar.org/paper/cd873347660c2af6a70d623a9fb265893e64c98d)  
Influential Citation Count (348), SS-ID (cd873347660c2af6a70d623a9fb265893e64c98d)  
{{< /ci-details >}}
{{< ci-details summary="A Neural Probabilistic Language Model (Yoshua Bengio et al., 2003)">}}
Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Janvin. (2003)  
**A Neural Probabilistic Language Model**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/6c2b28f9354f667cd5bd07afc0471d8334430da7)  
Influential Citation Count (479), SS-ID (6c2b28f9354f667cd5bd07afc0471d8334430da7)  
**ABSTRACT**  
A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language. This is intrinsically difficult because of the curse of dimensionality: a word sequence on which the model will be tested is likely to be different from all the word sequences seen during training. Traditional but very successful approaches based on n-grams obtain generalization by concatenating very short overlapping sequences seen in the training set. We propose to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences. The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations. Generalization is obtained because a sequence of words that has never been seen before gets high probability if it is made of words that are similar (in the sense of having a nearby representation) to words forming an already seen sentence. Training such large models (with millions of parameters) within a reasonable time is itself a significant challenge. We report on experiments using neural networks for the probability function, showing on two text corpora that the proposed approach significantly improves on state-of-the-art n-gram models, and that the proposed approach allows to take advantage of longer contexts.
{{< /ci-details >}}
{{< ci-details summary="Approximate nearest neighbors: towards removing the curse of dimensionality (P. Indyk et al., 1998)">}}
P. Indyk, R. Motwani. (1998)  
**Approximate nearest neighbors: towards removing the curse of dimensionality**  
STOC '98  
[Paper Link](https://www.semanticscholar.org/paper/1955266a8a58d94e41ad0efe20d707c92a069e95)  
Influential Citation Count (624), SS-ID (1955266a8a58d94e41ad0efe20d707c92a069e95)  
**ABSTRACT**  
We present two algorithms for the approximate nearest neighbor problem in high-dimensional spaces. For data sets of size n living in R d , the algorithms require space that is only polynomial in n and d, while achieving query times that are sub-linear in n and polynomial in d. We also show applications to other high-dimensional geometric problems, such as the approximate minimum spanning tree. The article is based on the material from the authors' STOC'98 and FOCS'01 papers. It unifies, generalizes and simplifies the results from those papers.
{{< /ci-details >}}
{{< ci-details summary="Learning representations by back-propagating errors (D. Rumelhart et al., 1986)">}}
D. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams. (1986)  
**Learning representations by back-propagating errors**  
Nature  
[Paper Link](https://www.semanticscholar.org/paper/052b1d8ce63b07fec3de9dbb583772d860b7c769)  
Influential Citation Count (733), SS-ID (052b1d8ce63b07fec3de9dbb583772d860b7c769)  
{{< /ci-details >}}
{{< ci-details summary="An Information Flow Model for Conflict and Fission in Small Groups (W. Zachary, 1977)">}}
W. Zachary. (1977)  
**An Information Flow Model for Conflict and Fission in Small Groups**  
Journal of Anthropological Research  
[Paper Link](https://www.semanticscholar.org/paper/0de728ad1b67221d7a7302f809f987bb926f4504)  
Influential Citation Count (325), SS-ID (0de728ad1b67221d7a7302f809f987bb926f4504)  
**ABSTRACT**  
Data from a voluntary association are used to construct a new formal model for a traditional anthropological problem, fission in small groups. The process leading to fission is viewed as an unequal flow of sentiments and information across the ties in a social network. This flow is unequal because it is uniquely constrained by the contextual range and sensitivity of each relationship in the network. The subsequent differential sharing of sentiments leads to the formation of subgroups with more internal stability than the group as a whole, and results in fission. The Ford-Fulkerson labeling algorithm allows an accurate prediction of membership in the subgroups and of the locus of the fission to be made from measurements of the potential for information flow across each edge in the network. Methods for measurement of potential information flow are discussed, and it is shown that all appropriate techniques will generate the same predictions.
{{< /ci-details >}}
