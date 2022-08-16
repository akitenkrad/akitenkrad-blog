---
draft: false
title: "High-order Proximity Preserved Embedding for Dynamic Networks"
date: 2022-06-18
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:20XX"]
menu:
  sidebar:
    name: 2022.06.18
    identifier: 20220618
    parent: 202206
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Zhu, D., Cui, P., Zhang, Z., Pei, J., & Zhu, W. (2018).  
High-Order Proximity Preserved Embedding for Dynamic Networks.  
IEEE Transactions on Knowledge and Data Engineering, 30(11), 2134–2144.  
https://doi.org/10.1109/TKDE.2018.2822283
{{< /citation >}}

## Abstract
> Network embedding, aiming to embed a network into a low dimensional vector space while preserving the inherent structural properties of the network, has attracted considerable attention. However, most existing embedding methods focus on the static network while neglecting the evolving characteristic of real-world networks. Meanwhile, most of previous methods cannot well preserve the high-order proximity, which is a critical structural property of networks. These problems motivate us to seek an effective and efficient way to preserve the high-order proximity in embedding vectors when the networks evolve over time. In this paper, we propose a novel method of Dynamic High-order Proximity preserved Embedding (DHPE). Specifically, we adopt the generalized SVD (GSVD) to preserve the high-order proximity. Then, by transforming the GSVD problem to a generalized eigenvalue problem, we propose a generalized eigen perturbation to incrementally update the results of GSVD to incorporate the changes of dynamic networks. Further, we propose an accelerated solution to the DHPE model so that it achieves a linear time complexity with respect to the number of nodes and number of changed edges in the network. Our empirical experiments on one synthetic network and several real-world networks demonstrate the effectiveness and efficiency of the proposed method.

## Background & What's New

- 動的なグラフ構造データにおいて，**High-order Proximity** はグラフの構造的な特徴を捉えるための重要な特徴量であるが，既存手法の多くは High-order Proximity をうまく扱えない
    - 要因の一つは，既存手法の多くが Static Graph を対象としたものでありノードやエッジが追加・削除された場合の High-order Proximity の変動を捉えることができないことである
- グラフ構造データにおいては，そもそも High-order Proximity の計算自体が負荷の高い処理である
    - 近年，Ou et al. (2016) において，Generalized SVD(GSVD) を用いることによって High-order Proximity を陽に計算することなく特徴量を保持したままベクトルに変換する方法が提案された
    - しかし，動的なグラフにおいて GSVD をどのように活用することができるか，という点については研究の余地が残されていた
- 動的な無向グラフにおいて，High-order Proximity を保持したEmbeddingを算出できる手法として，**DHPE** を提案した
    - GSVDを適用した後，問題を一般化された固有値問題に変換し，行列摂動理論を用いてグラフの動的な構造変化に対応した
- 行列摂動理論を用いた計算は複雑になりがちで効率性のボトルネックになりうるが，この点を改善する方法を提案し，計算効率を大きく向上させた


{{< ci-details summary="Asymmetric Transitivity Preserving Graph Embedding (Mingdong Ou et al., 2016)">}}

Mingdong Ou, Peng Cui, J. Pei, Ziwei Zhang, Wenwu Zhu. (2016)  
**Asymmetric Transitivity Preserving Graph Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  
Influential Citation Count (117), SS-ID (07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  

**ABSTRACT**  
Graph embedding algorithms embed a graph into a vector space where the structure and the inherent properties of the graph are preserved. The existing graph embedding methods cannot preserve the asymmetric transitivity well, which is a critical property of directed graphs. Asymmetric transitivity depicts the correlation among directed edges, that is, if there is a directed path from u to v, then there is likely a directed edge from u to v. Asymmetric transitivity can help in capturing structures of graphs and recovering from partially observed graphs. To tackle this challenge, we propose the idea of preserving asymmetric transitivity by approximating high-order proximity which are based on asymmetric transitivity. In particular, we develop a novel graph embedding algorithm, High-Order Proximity preserved Embedding (HOPE for short), which is scalable to preserve high-order proximities of large scale graphs and capable of capturing the asymmetric transitivity. More specifically, we first derive a general formulation that cover multiple popular high-order proximity measurements, then propose a scalable embedding algorithm to approximate the high-order proximity measurements based on their general formulation. Moreover, we provide a theoretical upper bound on the RMSE (Root Mean Squared Error) of the approximation. Our empirical experiments on a synthetic dataset and three real-world datasets demonstrate that HOPE can approximate the high-order proximities significantly better than the state-of-art algorithms and outperform the state-of-art algorithms in tasks of reconstruction, link prediction and vertex recommendation.

{{< /ci-details >}}


## Dataset

## Model Description

ステップ $t$ における Dynamic Network を

$$
\begin{array}{l}
    G^{(t)} & = \lbrace V^{(t)}, E^{(t)} \rbrace \\\\
    & \text{where} \\\\
    & \hspace{10pt} \begin{array}{l}
        V^{(t)} = \left\lbrace v\_1^{(t)}, v\_2^{(t)}, \ldots, v\_N^{(t)} \right\rbrace
    \end{array}
\end{array}
$$

と表すこととし，$G^{(t)}$ の Embedding と high-order proximity をそれぞれ

$$
\begin{array}{l}
    \text{Embedding} & \mapsto U^{(t)} \in \mathbb{R}^{N \times d} \\\\
    \text{High-order Proximity} & \mapsto S^{(t)} \hspace{10pt} (S\_{ij}^{(t)} \text{ is the proximity between }v\_i^{(t)}\text{ and }v\_j^{(t)}) \\\\
    & \text{where} \\\\
    & \hspace{10pt} \begin{array}{l}
        d & \mapsto \text{embedding dimension} \\\\
        S^{(t)} & \in \mathbb{R}^N
    \end{array}
\end{array}
$$

と表す．

Dynamic Network Embedding の計算プロセスは次の2つのステップに分けられる．

{{< box-with-title title="Step 1. Static network embedding" >}}
given adjacency matrix $A^{(t)}$, at time step $t$;  
output the embedding matrix
$$
U^{(t)}
$$
using static model.
{{< /box-with-title >}}

{{< box-with-title title="Step 2. Dynamic network embedding" >}}
given adjacency matrix $\lbrace A^{(t+1)}, A^{(t+2)}, \ldots, A^{(t+i)} \rbrace$, at time steps $\lbrace t+1, t+2, \ldots, t+i \rbrace$ and the embedding matrix $U^{(t)}$ at time step $t$;  
output the embedding matrix
$$
\lbrace U^{(t+1)}, U^{(t+2)}, \ldots, U^{(t+i)} \rbrace
$$
at time steps $\lbrace t+1, t+2, \ldots, t+i \rbrace$.
{{< /box-with-title >}}

#### GSVD-Based Static Model

Ou et al. (2016) において提案されている手法にしたがって，次の目的関数を考える．

$$
\begin{array}{l}
    \min \left\lVert S - U{U^\prime}^\mathsf{T} \right\rVert\_F^2 \\\\
    \text{where} \\\\
    \hspace{10pt} \begin{array}{c}
        U, U^\prime &\in& \mathbb{R}^{N \times d} \\\\
        S &\in& \mathbb{R}^N
    \end{array}
\end{array}
$$

{{< ci-details summary="Asymmetric Transitivity Preserving Graph Embedding (Mingdong Ou et al., 2016)">}}

Mingdong Ou, Peng Cui, J. Pei, Ziwei Zhang, Wenwu Zhu. (2016)  
**Asymmetric Transitivity Preserving Graph Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  
Influential Citation Count (117), SS-ID (07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  

**ABSTRACT**  
Graph embedding algorithms embed a graph into a vector space where the structure and the inherent properties of the graph are preserved. The existing graph embedding methods cannot preserve the asymmetric transitivity well, which is a critical property of directed graphs. Asymmetric transitivity depicts the correlation among directed edges, that is, if there is a directed path from u to v, then there is likely a directed edge from u to v. Asymmetric transitivity can help in capturing structures of graphs and recovering from partially observed graphs. To tackle this challenge, we propose the idea of preserving asymmetric transitivity by approximating high-order proximity which are based on asymmetric transitivity. In particular, we develop a novel graph embedding algorithm, High-Order Proximity preserved Embedding (HOPE for short), which is scalable to preserve high-order proximities of large scale graphs and capable of capturing the asymmetric transitivity. More specifically, we first derive a general formulation that cover multiple popular high-order proximity measurements, then propose a scalable embedding algorithm to approximate the high-order proximity measurements based on their general formulation. Moreover, we provide a theoretical upper bound on the RMSE (Root Mean Squared Error) of the approximation. Our empirical experiments on a synthetic dataset and three real-world datasets demonstrate that HOPE can approximate the high-order proximities significantly better than the state-of-art algorithms and outperform the state-of-art algorithms in tasks of reconstruction, link prediction and vertex recommendation.

{{< /ci-details >}}

<br/>

High-order Proximity の指標として広く用いられている Katz Index を $S$ として採用する．

$$
\begin{array}{l}
    S^{\text{Katz}} & = M\_a^{-1} M\_b \\\\
    M\_a & = (I - \beta A) \\\\
    M\_b & = \beta A \\\\
    & \text{where} \\\\
    & \hspace{10pt} \begin{array}{l}
        I & \mapsto \text{Identity Matrix} \\\\
        \beta & \mapsto \text{determines how fast the weight of a path decays when the length of path grows}
    \end{array}
\end{array}
$$

$\beta$ は収束に大きく影響するので，適切に設定する必要がある．

Ou et al. (2016) で提案されている通り，ここで Generalized SVD (GSVD) を適用することによって，$S$ を計算することなく $S$ の特異値と特異ベクトルを得ることができる．  
すなわち，目的関数における最適なEmbeddingは以下で与えられる．

$$
\begin{array}{l}
    U & = \left[ \sqrt{\sigma\_1} \boldsymbol{v}\_1^l, \ldots, \sqrt{\sigma\_d}\boldsymbol{v}\_d^l\right] \\\\
    U^\prime & = \left[ \sqrt{\sigma\_1} \boldsymbol{v}\_1^r, \ldots, \sqrt{\sigma\_d}\boldsymbol{v}\_d^r\right] \\\\
    & \text{where} \\\\
    & \hspace{10pt} \begin{array}{l}
        \lbrace \sigma\_1, \ldots, \sigma\_N \rbrace & \mapsto \text{the singular values of }S \text{ sorted in descending order} \\\\
        \boldsymbol{v}\_i^l, \boldsymbol{v}\_i^r & \mapsto \text{corresponding left and right singular vectors of }\sigma\_i
    \end{array}
\end{array}
$$

また，Ou et al. (2016) によれば，GSVD-Based Static Model のエラーバウンドは

$$
\left\lVert S - U{U^\prime}^\mathsf{T} \right\rVert \_F^2 = \sum\_{i=d+1}^N \sigma\_i^2
$$

である．

#### Problem Transformation for Dynamic Model

$\Delta A$ および $U^{(t)}$ が与えられたときに， $U^{(t)}$ を $U^{(t+1)}$ へ更新する方法について検討する．  
GSVDのアウトプットは下記で与えられる．

$$
\begin{array}{l}
    S^{(t)} & = {M\_a^{(t)}}^{-1} M\_b^{(t)} = V^{l(t)} \Sigma^{(t)} {V^{r(t)}}^\mathsf{T} \\\\
    \Sigma^{(t)} &= \text{diag} \left( \sigma\_1^{(t)}, \ldots, \sigma\_N^{(t)} \right) \\\\
    & \text{where} \\\\
    & \begin{array}{l}
        V^{l(t)}, V^{r(t)} & \mapsto \text{singular vectors in matrices}
    \end{array}
\end{array}
$$

上式を直接計算して $S^{(t+1)}$ を算出するのは非常に計算コストが高いため，GSVDを次のように一般化固有値問題へ変形する．  
無向グラフにおいては，$A$ および $S$ は対称行列となるので，

$$
\begin{array}{l}
    M_a^{-1} M\_b X & = \Lambda X \\\\
    \Lambda & = \text{diag} \left( \lambda\_1, \ldots, \lambda\_N \right) \\\\
    \lambda\_i & = \sigma\_i \cdot \text{sgn} \left( \boldsymbol{v}\_i^l \cdot \boldsymbol{v}\_i^r \right) \\\\
    X & = V^l \\\\
    & \text{where} \\\\
    & \begin{array}{l}
        \lbrace \lambda\_i \rbrace & \mapsto \text{eigenvalues of }S\text{ in descending order} \\\\
        X & \mapsto \text{a matrix which contains the corresponding eigen vectors of }\lambda\_i \\\\
        \text{sgn} & \mapsto \text{Sign function}
    \end{array}
\end{array} \tag{1}
$$

となり，

$$
M\_b X = M\_a \Lambda X
$$

を得る．  
上式は明らかに固有値問題の一般化された形式であり，その計算結果をGSVDに逆変形することができる．すなわち，

$$
\begin{array}{l}
    \boldsymbol{v}\_i^l & = \boldsymbol{x}\_i \sigma\_i \\\\
    & = \lvert \lambda\_i \rvert \boldsymbol{v}\_i^r \\\\
    & = \boldsymbol{x}\_i \cdot \text{sgn} \left( \lambda\_i \right)
\end{array} \tag{2}
$$

となる．したがって，$\Sigma^{(t)}$，$V^{l(t)}$， $V^{r(t)}$ が与えられれば，(1)によって $X^{(t)}$ および $\Lambda^{(t)}$ を計算することができ，$X^{(t+1)}$ および $\Lambda^{(t+1)}$ が得られれば，(2)によって $\Sigma^{(t)}$，$V^{l(t)}$， $V^{r(t)}$ を計算することができる．

よって，次の問題は $X^{(t)}$ を $X^{(t+1)}$ に効率よく更新する計算をどのように実装するか，というものとなる．

#### Generalized Eigen Perturbation

### Training Settings

## Results

## References


{{< ci-details summary="Asymmetric Transitivity Preserving Graph Embedding (Mingdong Ou et al., 2016)">}}

Mingdong Ou, Peng Cui, J. Pei, Ziwei Zhang, Wenwu Zhu. (2016)  
**Asymmetric Transitivity Preserving Graph Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  
Influential Citation Count (117), SS-ID (07627bf7eb649220ffbcdf6bf233e3a4a76e8590)  

**ABSTRACT**  
Graph embedding algorithms embed a graph into a vector space where the structure and the inherent properties of the graph are preserved. The existing graph embedding methods cannot preserve the asymmetric transitivity well, which is a critical property of directed graphs. Asymmetric transitivity depicts the correlation among directed edges, that is, if there is a directed path from u to v, then there is likely a directed edge from u to v. Asymmetric transitivity can help in capturing structures of graphs and recovering from partially observed graphs. To tackle this challenge, we propose the idea of preserving asymmetric transitivity by approximating high-order proximity which are based on asymmetric transitivity. In particular, we develop a novel graph embedding algorithm, High-Order Proximity preserved Embedding (HOPE for short), which is scalable to preserve high-order proximities of large scale graphs and capable of capturing the asymmetric transitivity. More specifically, we first derive a general formulation that cover multiple popular high-order proximity measurements, then propose a scalable embedding algorithm to approximate the high-order proximity measurements based on their general formulation. Moreover, we provide a theoretical upper bound on the RMSE (Root Mean Squared Error) of the approximation. Our empirical experiments on a synthetic dataset and three real-world datasets demonstrate that HOPE can approximate the high-order proximities significantly better than the state-of-art algorithms and outperform the state-of-art algorithms in tasks of reconstruction, link prediction and vertex recommendation.

{{< /ci-details >}}

{{< ci-details summary="LINE: Large-scale Information Network Embedding (Jian Tang et al., 2015)">}}

Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Q. Mei. (2015)  
**LINE: Large-scale Information Network Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
Influential Citation Count (867), SS-ID (0834e74304b547c9354b6d7da6fa78ef47a48fa8)  

**ABSTRACT**  
This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for real world information networks which usually contain millions of nodes. In this paper, we propose a novel network embedding method called the ``LINE,'' which is suitable for arbitrary types of information networks: undirected, directed, and/or weighted. The method optimizes a carefully designed objective function that preserves both the local and global network structures. An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the effectiveness and the efficiency of the inference. Empirical experiments prove the effectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very efficient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online\footnote{\url{https://github.com/tangjianpku/LINE}}.

{{< /ci-details >}}

{{< ci-details summary="Scalable learning of collective behavior based on sparse social dimensions (Lei Tang et al., 2009)">}}

Lei Tang, Huan Liu. (2009)  
**Scalable learning of collective behavior based on sparse social dimensions**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/094f9616e15f4e64e7afd9d7f5a1b092bbc83738)  
Influential Citation Count (33), SS-ID (094f9616e15f4e64e7afd9d7f5a1b092bbc83738)  

**ABSTRACT**  
The study of collective behavior is to understand how individuals behave in a social network environment. Oceans of data generated by social media like Facebook, Twitter, Flickr and YouTube present opportunities and challenges to studying collective behavior in a large scale. In this work, we aim to learn to predict collective behavior in social media. In particular, given information about some individuals, how can we infer the behavior of unobserved individuals in the same network? A social-dimension based approach is adopted to address the heterogeneity of connections presented in social media. However, the networks in social media are normally of colossal size, involving hundreds of thousands or even millions of actors. The scale of networks entails scalable learning of models for collective behavior prediction. To address the scalability issue, we propose an edge-centric clustering scheme to extract sparse social dimensions. With sparse social dimensions, the social-dimension based approach can efficiently handle networks of millions of actors while demonstrating comparable prediction performance as other non-scalable methods.

{{< /ci-details >}}

{{< ci-details summary="A General Framework for Content-enhanced Network Representation Learning (Xiaofei Sun et al., 2016)">}}

Xiaofei Sun, Jiang Guo, Xiao Ding, Ting Liu. (2016)  
**A General Framework for Content-enhanced Network Representation Learning**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/1268d2ae95f0b128678d6ce033ba8ea7f0d98be1)  
Influential Citation Count (12), SS-ID (1268d2ae95f0b128678d6ce033ba8ea7f0d98be1)  

**ABSTRACT**  
This paper investigates the problem of network embedding, which aims at learning low-dimensional vector representation of nodes in networks. Most existing network embedding methods rely solely on the network structure, i.e., the linkage relationships between nodes, but ignore the rich content information associated with it, which is common in real world networks and beneficial to describing the characteristics of a node. In this paper, we propose content-enhanced network embedding (CENE), which is capable of jointly leveraging the network structure and the content information. Our approach integrates text modeling and structure modeling in a general framework by treating the content information as a special kind of node. Experiments on several real world net- works with application to node classification show that our models outperform all existing network embedding methods, demonstrating the merits of content information and joint learning.

{{< /ci-details >}}

{{< ci-details summary="Model‐based clustering for social networks (M. Handcock et al., 2007)">}}

M. Handcock, A. Raftery, J. Tantrum. (2007)  
**Model‐based clustering for social networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/157c24a3df3203622cfc2ffd514e6ea10b019a14)  
Influential Citation Count (78), SS-ID (157c24a3df3203622cfc2ffd514e6ea10b019a14)  

**ABSTRACT**  
Summary.  Network models are widely used to represent relations between interacting units or actors. Network data often exhibit transitivity, meaning that two actors that have ties to a third actor are more likely to be tied than actors that do not, homophily by attributes of the actors or dyads, and clustering. Interest often focuses on finding clusters of actors or ties, and the number of groups in the data is typically unknown. We propose a new model, the latent position cluster model, under which the probability of a tie between two actors depends on the distance between them in an unobserved Euclidean ‘social space’, and the actors’ locations in the latent social space arise from a mixture of distributions, each corresponding to a cluster. We propose two estimation methods: a two‐stage maximum likelihood method and a fully Bayesian method that uses Markov chain Monte Carlo sampling. The former is quicker and simpler, but the latter performs better. We also propose a Bayesian way of determining the number of clusters that are present by using approximate conditional Bayes factors. Our model represents transitivity, homophily by attributes and clustering simultaneously and does not require the number of clusters to be known. The model makes it easy to simulate realistic networks with clustering, which are potentially useful as inputs to models of more complex systems of which the network is part, such as epidemic models of infectious disease. We apply the model to two networks of social relations. A free software package in the R statistical language, latentnet, is available to analyse data by using the model.

{{< /ci-details >}}

{{< ci-details summary="Visualizing Data using t-SNE (L. V. D. Maaten et al., 2008)">}}

L. V. D. Maaten, Geoffrey E. Hinton. (2008)  
**Visualizing Data using t-SNE**  
  
[Paper Link](https://www.semanticscholar.org/paper/1c46943103bd7b7a2c7be86859995a4144d1938b)  
Influential Citation Count (873), SS-ID (1c46943103bd7b7a2c7be86859995a4144d1938b)  

**ABSTRACT**  
We present a new technique called “t-SNE” that visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large datasets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of datasets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the datasets.

{{< /ci-details >}}

{{< ci-details summary="TIMERS: Error-Bounded SVD Restart on Dynamic Networks (Ziwei Zhang et al., 2017)">}}

Ziwei Zhang, Peng Cui, J. Pei, Xiao Wang, Wenwu Zhu. (2017)  
**TIMERS: Error-Bounded SVD Restart on Dynamic Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/1ca27cb4d43e7f3fbc1c4196252168702d7b3b3e)  
Influential Citation Count (10), SS-ID (1ca27cb4d43e7f3fbc1c4196252168702d7b3b3e)  

**ABSTRACT**  
Singular Value Decomposition (SVD) is a popular approach in various network applications, such as link prediction and network parameter characterization. Incremental SVD approaches are proposed to process newly changed nodes and edges in dynamic networks. However, incremental SVD approaches suffer from serious error accumulation inevitably due to approximation on incremental updates. SVD restart is an effective approach to reset the aggregated error, but when to restart SVD for dynamic networks is not addressed in literature. In this paper, we propose TIMERS, Theoretically Instructed Maximum-Error-bounded Restart of SVD, a novel approach which optimally sets the restart time in order to reduce error accumulation in time. Specifically, we monitor the margin between reconstruction loss of incremental updates and the minimum loss in SVD model. To reduce the complexity of monitoring, we theoretically develop a lower bound of SVD minimum loss for dynamic networks and use the bound to replace the minimum loss in monitoring. By setting a maximum tolerated error as a threshold, we can trigger SVD restart automatically when the margin exceeds this threshold. We prove that the time complexity of our method is linear with respect to the number of local dynamic changes, and our method is general across different types of dynamic networks. We conduct extensive experiments on several synthetic and real dynamic networks. The experimental results demonstrate that our proposed method significantly outperforms the existing methods by reducing 27% to 42% in terms of the maximum error for dynamic network reconstruction when fixing the number of restarts. Our method reduces the number of restarts by 25% to 50% when fixing the maximum error tolerated.   

{{< /ci-details >}}

{{< ci-details summary="LIBLINEAR: A Library for Large Linear Classification (Rong-En Fan et al., 2008)">}}

Rong-En Fan, Kai-Wei Chang, Cho-Jui Hsieh, Xiang-Rui Wang, Chih-Jen Lin. (2008)  
**LIBLINEAR: A Library for Large Linear Classification**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/268a4f8da15a42f3e0e71691f760ff5edbf9cec8)  
Influential Citation Count (907), SS-ID (268a4f8da15a42f3e0e71691f760ff5edbf9cec8)  

**ABSTRACT**  
LIBLINEAR is an open source library for large-scale linear classification. It supports logistic regression and linear support vector machines. We provide easy-to-use command-line tools and library calls for users and developers. Comprehensive documents are available for both beginners and advanced users. Experiments demonstrate that LIBLINEAR is very efficient on large sparse data sets.

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

{{< ci-details summary="A global geometric framework for nonlinear dimensionality reduction. (J. Tenenbaum et al., 2000)">}}

J. Tenenbaum, V. De Silva, J. Langford. (2000)  
**A global geometric framework for nonlinear dimensionality reduction.**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/3537fcd0ff99a3b3cb3d279012df826358420556)  
Influential Citation Count (1200), SS-ID (3537fcd0ff99a3b3cb3d279012df826358420556)  

**ABSTRACT**  
Scientists working with large volumes of high-dimensional data, such as global climate patterns, stellar spectra, or human gene distributions, regularly confront the problem of dimensionality reduction: finding meaningful low-dimensional structures hidden in their high-dimensional observations. The human brain confronts the same problem in everyday perception, extracting from its high-dimensional sensory inputs-30,000 auditory nerve fibers or 10(6) optic nerve fibers-a manageably small number of perceptually relevant features. Here we describe an approach to solving dimensionality reduction problems that uses easily measured local metric information to learn the underlying global geometry of a data set. Unlike classical techniques such as principal component analysis (PCA) and multidimensional scaling (MDS), our approach is capable of discovering the nonlinear degrees of freedom that underlie complex natural observations, such as human handwriting or images of a face under different viewing conditions. In contrast to previous algorithms for nonlinear dimensionality reduction, ours efficiently computes a globally optimal solution, and, for an important class of data manifolds, is guaranteed to converge asymptotically to the true structure.

{{< /ci-details >}}

{{< ci-details summary="node2vec: Scalable Feature Learning for Networks (Aditya Grover et al., 2016)">}}

Aditya Grover, J. Leskovec. (2016)  
**node2vec: Scalable Feature Learning for Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/36ee2c8bd605afd48035d15fdc6b8c8842363376)  
Influential Citation Count (1164), SS-ID (36ee2c8bd605afd48035d15fdc6b8c8842363376)  

**ABSTRACT**  
Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

{{< /ci-details >}}

{{< ci-details summary="Revisiting Semi-Supervised Learning with Graph Embeddings (Zhilin Yang et al., 2016)">}}

Zhilin Yang, William W. Cohen, R. Salakhutdinov. (2016)  
**Revisiting Semi-Supervised Learning with Graph Embeddings**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/3d846cb01f6a975554035d2210b578ca61344b22)  
Influential Citation Count (186), SS-ID (3d846cb01f6a975554035d2210b578ca61344b22)  

**ABSTRACT**  
We present a semi-supervised learning framework based on graph embeddings. Given a graph between instances, we train an embedding for each instance to jointly predict the class label and the neighborhood context in the graph. We develop both transductive and inductive variants of our method. In the transductive variant of our method, the class labels are determined by both the learned embeddings and input feature vectors, while in the inductive variant, the embeddings are defined as a parametric function of the feature vectors, so predictions can be made on instances not seen during training. On a large and diverse set of benchmark tasks, including text classification, distantly supervised entity extraction, and entity classification, we show improved performance over many of the existing models.

{{< /ci-details >}}

{{< ci-details summary="Fast, Warped Graph Embedding: Unifying Framework and One-Click Algorithm (Siheng Chen et al., 2017)">}}

Siheng Chen, Sufeng Niu, L. Akoglu, J. Kovacevic, C. Faloutsos. (2017)  
**Fast, Warped Graph Embedding: Unifying Framework and One-Click Algorithm**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/406b5b1f350a4f51391da29fcfa2a6800dc33973)  
Influential Citation Count (4), SS-ID (406b5b1f350a4f51391da29fcfa2a6800dc33973)  

**ABSTRACT**  
What is the best way to describe a user in a social network with just a few numbers? Mathematically, this is equivalent to assigning a vector representation to each node in a graph, a process called graph embedding. We propose a novel framework, GEM-D that unifies most of the past algorithms such as LapEigs, DeepWalk and node2vec. GEM-D achieves its goal by decomposing any graph embedding algorithm into three building blocks: node proximity function, warping function and loss function. Based on thorough analysis of GEM-D, we propose a novel algorithm, called UltimateWalk, which outperforms the most-recently proposed state-of-the-art DeepWalk and node2vec. The contributions of this work are: (1) The proposed framework, GEM-D unifies the past graph embedding algorithms and provides a general recipe of how to design a graph embedding; (2) the nonlinearlity in the warping function contributes significantly to the quality of embedding and the exponential function is empirically optimal; (3) the proposed algorithm, UltimateWalk is one-click (no user-defined parameters), scalable and has a closed-form solution.

{{< /ci-details >}}

{{< ci-details summary="Matrix Computations (A. Chrzȩszczyk et al., 2011)">}}

A. Chrzȩszczyk, J. Kochanowski. (2011)  
**Matrix Computations**  
Encyclopedia of Parallel Computing  
[Paper Link](https://www.semanticscholar.org/paper/444d70e3331b5083b40ef32e49390ef683a65e67)  
Influential Citation Count (569), SS-ID (444d70e3331b5083b40ef32e49390ef683a65e67)  

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

{{< ci-details summary="Combining content and link for classification using matrix factorization (Shenghuo Zhu et al., 2007)">}}

Shenghuo Zhu, Kai Yu, Yun Chi, Yihong Gong. (2007)  
**Combining content and link for classification using matrix factorization**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/5c58ad9a6c09782814a7d048bebd6ef1609c0fb4)  
Influential Citation Count (19), SS-ID (5c58ad9a6c09782814a7d048bebd6ef1609c0fb4)  

**ABSTRACT**  
The world wide web contains rich textual contents that areinterconnected via complex hyperlinks. This huge database violates the assumption held by most of conventional statistical methods that each web page is considered as an independent and identical sample. It is thus difficult to apply traditional mining or learning methods for solving web mining problems, e.g., web page classification, by exploiting both the content and the link structure. The research in this direction has recently received considerable attention but are still in an early stage. Though a few methods exploit both the link structure or the content information, some of them combine the only authority information with the content information, and the others first decompose the link structure into hub and authority features, then apply them as additional document features. Being practically attractive for its great simplicity, this paper aims to design an algorithm that exploits both the content and linkage information, by carrying out a joint factorization on both the linkage adjacency matrix and the document-term matrix, and derives a new representation for web pages in a low-dimensional factor space, without explicitly separating them as content, hub or authority factors. Further analysis can be performed based on the compact representation of web pages. In the experiments, the proposed method is compared with state-of-the-art methods and demonstrates an excellent accuracy in hypertext classification on the WebKB and Cora benchmarks.

{{< /ci-details >}}

{{< ci-details summary="Graph Embedding and Extensions: A General Framework for Dimensionality Reduction (Shuicheng Yan et al., 2007)">}}

Shuicheng Yan, Dong Xu, Benyu Zhang, HongJiang Zhang, Qiang Yang, Stephen Lin. (2007)  
**Graph Embedding and Extensions: A General Framework for Dimensionality Reduction**  
IEEE Transactions on Pattern Analysis and Machine Intelligence  
[Paper Link](https://www.semanticscholar.org/paper/69381b5efd97e7c55f51c2730caccab3d632d4d2)  
Influential Citation Count (311), SS-ID (69381b5efd97e7c55f51c2730caccab3d632d4d2)  

**ABSTRACT**  
A large family of algorithms - supervised or unsupervised; stemming from statistics or geometry theory - has been designed to provide different solutions to the problem of dimensionality reduction. Despite the different motivations of these algorithms, we present in this paper a general formulation known as graph embedding to unify them within a common framework. In graph embedding, each algorithm can be considered as the direct graph embedding or its linear/kernel/tensor extension of a specific intrinsic graph that describes certain desired statistical or geometric properties of a data set, with constraints from scale normalization or a penalty graph that characterizes a statistical or geometric property that should be avoided. Furthermore, the graph embedding framework can be used as a general platform for developing new dimensionality reduction algorithms. By utilizing this framework as a tool, we propose a new supervised dimensionality reduction algorithm called marginal Fisher analysis in which the intrinsic graph characterizes the intraclass compactness and connects each data point with its neighboring points of the same class, while the penalty graph connects the marginal points and characterizes the interclass separability. We show that MFA effectively overcomes the limitations of the traditional linear discriminant analysis algorithm due to data distribution assumptions and available projection directions. Real face recognition experiments show the superiority of our proposed MFA in comparison to LDA, also for corresponding kernel and tensor extensions

{{< /ci-details >}}

{{< ci-details summary="Task-Guided and Path-Augmented Heterogeneous Network Embedding for Author Identification (Ting Chen et al., 2016)">}}

Ting Chen, Yizhou Sun. (2016)  
**Task-Guided and Path-Augmented Heterogeneous Network Embedding for Author Identification**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/6b183d2297cb493a57dbc875689ab2430d870043)  
Influential Citation Count (14), SS-ID (6b183d2297cb493a57dbc875689ab2430d870043)  

**ABSTRACT**  
In this paper, we study the problem of author identification under double-blind review setting, which is to identify potential authors given information of an anonymized paper. Different from existing approaches that rely heavily on feature engineering, we propose to use network embedding approach to address the problem, which can automatically represent nodes into lower dimensional feature vectors. However, there are two major limitations in recent studies on network embedding: (1) they are usually general-purpose embedding methods, which are independent of the specific tasks; and (2) most of these approaches can only deal with homogeneous networks, where the heterogeneity of the network is ignored. Hence, challenges faced here are two folds: (1) how to embed the network under the guidance of the author identification task, and (2) how to select the best type of information due to the heterogeneity of the network. To address the challenges, we propose a task-guided and path-augmented heterogeneous network embedding model. In our model, nodes are first embedded as vectors in latent feature space. Embeddings are then shared and jointly trained according to task-specific and network-general objectives. We extend the existing unsupervised network embedding to incorporate meta paths in heterogeneous networks, and select paths according to the specific task. The guidance from author identification task for network embedding is provided both explicitly in joint training and implicitly during meta path selection. Our experiments demonstrate that by using path-augmented network embedding with task guidance, our model can obtain significantly better accuracy at identifying the true authors comparing to existing methods.

{{< /ci-details >}}

{{< ci-details summary="Graphs over time: densification laws, shrinking diameters and possible explanations (J. Leskovec et al., 2005)">}}

J. Leskovec, J. Kleinberg, C. Faloutsos. (2005)  
**Graphs over time: densification laws, shrinking diameters and possible explanations**  
KDD '05  
[Paper Link](https://www.semanticscholar.org/paper/788b6f36a2b7cab86a5a29000e8b7cde25b85e73)  
Influential Citation Count (182), SS-ID (788b6f36a2b7cab86a5a29000e8b7cde25b85e73)  

**ABSTRACT**  
How do real graphs evolve over time? What are "normal" growth patterns in social, technological, and information networks? Many studies have discovered patterns in static graphs, identifying properties in a single snapshot of a large network, or in a very small number of snapshots; these include heavy tails for in- and out-degree distributions, communities, small-world phenomena, and others. However, given the lack of information about network evolution over long periods, it has been hard to convert these findings into statements about trends over time.Here we study a wide range of real graphs, and we observe some surprising phenomena. First, most of these graphs densify over time, with the number of edges growing super-linearly in the number of nodes. Second, the average distance between nodes often shrinks over time, in contrast to the conventional wisdom that such distance parameters should increase slowly as a function of the number of nodes (like O(log n) or O(log(log n)).Existing graph generation models do not exhibit these types of behavior, even at a qualitative level. We provide a new graph generator, based on a "forest fire" spreading process, that has a simple, intuitive justification, requires very few parameters (like the "flammability" of nodes), and produces graphs exhibiting the full range of properties observed both in prior work and in the present study.

{{< /ci-details >}}

{{< ci-details summary="Structural Deep Embedding for Hyper-Networks (Ke Tu et al., 2017)">}}

Ke Tu, Peng Cui, Xiao Wang, Fei Wang, Wenwu Zhu. (2017)  
**Structural Deep Embedding for Hyper-Networks**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/7b9cdf953223aa27ea548fa3a62d77d67723b0e2)  
Influential Citation Count (12), SS-ID (7b9cdf953223aa27ea548fa3a62d77d67723b0e2)  

**ABSTRACT**  
Network embedding has recently attracted lots of attentions in data mining. Existing network embedding methods mainly focus on networks with pairwise relationships. In real world, however, the relationships among data points could go beyond pairwise, i.e., three or more objects are involved in each relationship represented by a hyperedge, thus forming hyper-networks. These hyper-networks pose great challenges to existing network embedding methods when the hyperedges are indecomposable, that is to say, any subset of nodes in a hyperedge cannot form another hyperedge. These indecomposable hyperedges are especially common in heterogeneous networks. In this paper, we propose a novel Deep Hyper-Network Embedding (DHNE) model to embed hyper-networks with indecomposable hyperedges. More specifically, we theoretically prove that any linear similarity metric in embedding space commonly used in existing methods cannot maintain the indecomposibility property in hyper-networks, and thus propose a new deep model to realize a non-linear tuplewise similarity function while preserving both local and global proximities in the formed embedding space. We conduct extensive experiments on four different types of hyper-networks, including a GPS network, an online social network, a drug network and a semantic network. The empirical results demonstrate that our method can significantly and consistently outperform the state-of-the-art algorithms.   

{{< /ci-details >}}

{{< ci-details summary="Latent Space Approaches to Social Network Analysis (Peter D. Hoff et al., 2002)">}}

Peter D. Hoff, A. Raftery, M. Handcock. (2002)  
**Latent Space Approaches to Social Network Analysis**  
  
[Paper Link](https://www.semanticscholar.org/paper/82e4390c043754d5af22d48964a42a891f81e8b3)  
Influential Citation Count (177), SS-ID (82e4390c043754d5af22d48964a42a891f81e8b3)  

**ABSTRACT**  
Network models are widely used to represent relational information among interacting units. In studies of social networks, recent emphasis has been placed on random graph models where the nodes usually represent individual social actors and the edges represent the presence of a specified relation between actors. We develop a class of models where the probability of a relation between actors depends on the positions of individuals in an unobserved “social space.” We make inference for the social space within maximum likelihood and Bayesian frameworks, and propose Markov chain Monte Carlo procedures for making inference on latent positions and the effects of observed covariates. We present analyses of three standard datasets from the social networks literature, and compare the method to an alternative stochastic blockmodeling approach. In addition to improving on model fit for these datasets, our method provides a visual and interpretable model-based spatial representation of social relationships and improves on existing methods by allowing the statistical uncertainty in the social space to be quantified and graphically represented.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Large Graph Embedding (F. Nie et al., 2017)">}}

F. Nie, Wei Zhu, Xuelong Li. (2017)  
**Unsupervised Large Graph Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/9ad503ff70a2b3a1ddebc96683ed73c7fcd0840b)  
Influential Citation Count (6), SS-ID (9ad503ff70a2b3a1ddebc96683ed73c7fcd0840b)  

**ABSTRACT**  
There are many successful spectral based unsupervised dimensionality reduction methods, including Laplacian Eigenmap (LE), Locality Preserving Projection (LPP), Spectral Regression (SR), etc. LPP and SR are two different linear spectral based methods, however, we discover that LPP and SR are equivalent, if the symmetric similarity matrix is doubly stochastic, Positive Semi-Definite (PSD) and with rank p, where p is the reduced dimension. The discovery promotes us to seek low-rank and doubly stochastic similarity matrix, we then propose an unsupervised linear dimensionality reduction method, called Unsupervised Large Graph Embedding (ULGE). ULGE starts with similar idea as LPP, it adopts an efficient approach to construct similarity matrix and then performs spectral analysis efficiently, the computational complexity can reduce to O(ndm), which is a significant improvement compared to conventional spectral based methods which need O(n^2d) at least, where n, d and m are the number of samples, dimensions and anchors, respectively. Extensive experiments on several public available data sets demonstrate the efficiency and effectiveness of the proposed method.   

{{< /ci-details >}}

{{< ci-details summary="Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering (Mikhail Belkin et al., 2001)">}}

Mikhail Belkin, P. Niyogi. (2001)  
**Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/9d16c547d15a08091e68c86a99731b14366e3f0d)  
Influential Citation Count (370), SS-ID (9d16c547d15a08091e68c86a99731b14366e3f0d)  

**ABSTRACT**  
Drawing on the correspondence between the graph Laplacian, the Laplace-Beltrami operator on a manifold, and the connections to the heat equation, we propose a geometrically motivated algorithm for constructing a representation for data sampled from a low dimensional manifold embedded in a higher dimensional space. The algorithm provides a computationally efficient approach to nonlinear dimensionality reduction that has locality preserving properties and a natural connection to clustering. Several applications are considered.

{{< /ci-details >}}

{{< ci-details summary="Towards a Generalized Singular Value Decomposition (C. Paige et al., 1981)">}}

C. Paige, M. Saunders. (1981)  
**Towards a Generalized Singular Value Decomposition**  
  
[Paper Link](https://www.semanticscholar.org/paper/9d2e9f807557b5cb50340402a0df45da53e5ba13)  
Influential Citation Count (59), SS-ID (9d2e9f807557b5cb50340402a0df45da53e5ba13)  

**ABSTRACT**  
We suggest a form for, and give a constructive derivation of, the generalized singular value decomposition of any two matrices having the same number of columns. We outline its desirable characteristics and compare it to an earlier suggestion by Van Loan [SIAM J. Numer. Anal., 13 (1976), pp. 76–83]. The present form largely follows from the work of Van Loan, but is slightly more general and computationally more amenable than that in the paper cited. We also prove a useful extension of a theorem of Stewart [SIAM Rev. 19 (1977), pp. 634–662] on unitary decompositions of submatrices of a unitary matrix.

{{< /ci-details >}}

{{< ci-details summary="Relational learning via latent social dimensions (Lei Tang et al., 2009)">}}

Lei Tang, Huan Liu. (2009)  
**Relational learning via latent social dimensions**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/a505e4c2bf30cd88afe483f7541409e2ba5ab3d4)  
Influential Citation Count (74), SS-ID (a505e4c2bf30cd88afe483f7541409e2ba5ab3d4)  

**ABSTRACT**  
Social media such as blogs, Facebook, Flickr, etc., presents data in a network format rather than classical IID distribution. To address the interdependency among data instances, relational learning has been proposed, and collective inference based on network connectivity is adopted for prediction. However, connections in social media are often multi-dimensional. An actor can connect to another actor for different reasons, e.g., alumni, colleagues, living in the same city, sharing similar interests, etc. Collective inference normally does not differentiate these connections. In this work, we propose to extract latent social dimensions based on network information, and then utilize them as features for discriminative learning. These social dimensions describe diverse affiliations of actors hidden in the network, and the discriminative learning can automatically determine which affiliations are better aligned with the class labels. Such a scheme is preferred when multiple diverse relations are associated with the same network. We conduct extensive experiments on social media data (one from a real-world blog site and the other from a popular content sharing site). Our model outperforms representative relational learning methods based on collective inference, especially when few labeled data are available. The sensitivity of this model and its connection to existing methods are also examined.

{{< /ci-details >}}

{{< ci-details summary="Nonlinear dimensionality reduction by locally linear embedding. (S. Roweis et al., 2000)">}}

S. Roweis, L. Saul. (2000)  
**Nonlinear dimensionality reduction by locally linear embedding.**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/afcd6da7637ddeef6715109aca248da7a24b1c65)  
Influential Citation Count (1579), SS-ID (afcd6da7637ddeef6715109aca248da7a24b1c65)  

**ABSTRACT**  
Many areas of science depend on exploratory data analysis and visualization. The need to analyze large amounts of multivariate data raises the fundamental problem of dimensionality reduction: how to discover compact representations of high-dimensional data. Here, we introduce locally linear embedding (LLE), an unsupervised learning algorithm that computes low-dimensional, neighborhood-preserving embeddings of high-dimensional inputs. Unlike clustering methods for local dimensionality reduction, LLE maps its inputs into a single global coordinate system of lower dimensionality, and its optimizations do not involve local minima. By exploiting the local symmetries of linear reconstructions, LLE is able to learn the global structure of nonlinear manifolds, such as those generated by images of faces or documents of text.

{{< /ci-details >}}

{{< ci-details summary="GraRep: Learning Graph Representations with Global Structural Information (Shaosheng Cao et al., 2015)">}}

Shaosheng Cao, Wei Lu, Qiongkai Xu. (2015)  
**GraRep: Learning Graph Representations with Global Structural Information**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  
Influential Citation Count (142), SS-ID (c2fd72cb2a77941e655b5d949d0d59b01e173c3b)  

**ABSTRACT**  
In this paper, we present {GraRep}, a novel model for learning vertex representations of weighted graphs. This model learns low dimensional vectors to represent vertices appearing in a graph and, unlike existing work, integrates global structural information of the graph into the learning process. We also formally analyze the connections between our work and several previous research efforts, including the DeepWalk model of Perozzi et al. as well as the skip-gram model with negative sampling of Mikolov et al. We conduct experiments on a language network, a social network as well as a citation network and show that our learned global representations can be effectively used as features in tasks such as clustering, classification and visualization. Empirical results demonstrate that our representation significantly outperforms other state-of-the-art methods in such tasks.

{{< /ci-details >}}

{{< ci-details summary="A Survey on Network Embedding (Peng Cui et al., 2017)">}}

Peng Cui, Xiao Wang, J. Pei, Wenwu Zhu. (2017)  
**A Survey on Network Embedding**  
IEEE Transactions on Knowledge and Data Engineering  
[Paper Link](https://www.semanticscholar.org/paper/ce840188f3395815201b7da49f9bb40d24fc046a)  
Influential Citation Count (31), SS-ID (ce840188f3395815201b7da49f9bb40d24fc046a)  

**ABSTRACT**  
Network embedding assigns nodes in a network to low-dimensional representations and effectively preserves the network structure. Recently, a significant amount of progresses have been made toward this emerging network analysis paradigm. In this survey, we focus on categorizing and then reviewing the current development on network embedding methods, and point out its future research directions. We first summarize the motivation of network embedding. We discuss the classical graph embedding algorithms and their relationship with network embedding. Afterwards and primarily, we provide a comprehensive overview of a large number of network embedding methods in a systematic manner, covering the structure- and property-preserving network embedding methods, the network embedding methods with side information, and the advanced information preserving network embedding methods. Moreover, several evaluation approaches for network embedding and some useful online resources, including the network data sets and softwares, are reviewed, too. Finally, we discuss the framework of exploiting these network embedding methods to build an effective system and point out some potential future directions.

{{< /ci-details >}}

{{< ci-details summary="Structural Deep Network Embedding (Daixin Wang et al., 2016)">}}

Daixin Wang, Peng Cui, Wenwu Zhu. (2016)  
**Structural Deep Network Embedding**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/d0b7c8828f0fca4dd901674e8fb5bd464a187664)  
Influential Citation Count (241), SS-ID (d0b7c8828f0fca4dd901674e8fb5bd464a187664)  

**ABSTRACT**  
Network embedding is an important method to learn low-dimensional representations of vertexes in networks, aiming to capture and preserve the network structure. Almost all the existing network embedding methods adopt shallow models. However, since the underlying network structure is complex, shallow models cannot capture the highly non-linear network structure, resulting in sub-optimal network representations. Therefore, how to find a method that is able to effectively capture the highly non-linear network structure and preserve the global and local structure is an open yet important problem. To solve this problem, in this paper we propose a Structural Deep Network Embedding method, namely SDNE. More specifically, we first propose a semi-supervised deep model, which has multiple layers of non-linear functions, thereby being able to capture the highly non-linear network structure. Then we propose to exploit the first-order and second-order proximity jointly to preserve the network structure. The second-order proximity is used by the unsupervised component to capture the global network structure. While the first-order proximity is used as the supervised information in the supervised component to preserve the local network structure. By jointly optimizing them in the semi-supervised deep model, our method can preserve both the local and global network structure and is robust to sparse networks. Empirically, we conduct the experiments on five real-world networks, including a language network, a citation network and three social networks. The results show that compared to the baselines, our method can reconstruct the original network significantly better and achieves substantial gains in three applications, i.e. multi-label classification, link prediction and visualization.

{{< /ci-details >}}

{{< ci-details summary="An Introduction to Linear Algebra (J. Esser, 2006)">}}

J. Esser. (2006)  
**An Introduction to Linear Algebra**  
  
[Paper Link](https://www.semanticscholar.org/paper/d2af753f6ae3711ed6fcd6fdac5bba70b7dae3a8)  
Influential Citation Count (66), SS-ID (d2af753f6ae3711ed6fcd6fdac5bba70b7dae3a8)  

**ABSTRACT**  
Class notes on vectors, linear combination, basis, span. 1 Vectors Vectors on the plane are ordered pairs of real numbers (a, b) such as (0, 1), (1, 0), (1, 2), (−1, 1). The plane is denoted by R, also known as Euclidean 2-space. Vectors in our physical space are ordered triples (a, b, c) such as (1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 1, 2), (1,−1, 2). All such vectors form Euclidean 3-space, or R. In general, the set of all ordered n-tuples (x1, x2, · · · , xn) is Euclidean n-space. 1.1 Addition and Scalar Multiplication Let v = [v1, v2, · · · , vn] and w = [w1, w2, · · · , wn] be two vectors in R. Then vector addition is: v + w = [v1 + w1, v2 + w2, · · · , vn + wn], vector subtraction is: v − w = [v1 − w1, v2 − w2, · · · , vn − wn]. For any number r (scalar), vector scalar multiplication is: r v = [r v1, r v2, · · · , r vn]. Two nonzero vectors v and w in R are said to be parallel if one is a scalar mulplication of the other, v = r w. If r > 0 (r < 0), they point in the same (opposite) direction. Example 1: are the two vectors v = [2, 1, 3,−5], and w = [6, 3, 9,−15] parallel ? Yes, w = 3 v. ∗Department of Mathematics, UCI, Irvine, CA 92617. 1 1.2 Linear Combination Given vectors v, v, · · · , v in R, and scalars r1, r2, · · · , rk in R, the vector: r1 v 1 + r2 v 2 + · · ·+ rk v, (1.1) is called a linear combination of the vectors v, v, · · · , v with scalar coefficients r1, r2, · · · , rk. Example 2: any vector [a1, a2] in R 2 can be expressed as a unique linear combination of the two vectors [1, 0] and [0, 1]: [a1, a2] = r1[1, 0] + r2[0, 1], (1.2) if and only if r1 = a1, r2 = a2. We call [1, 0] and [0, 1] standard basis vectors of R , often denoted by e and e. Similarly, standard basis vectors in R are: e = [1, 0, 0], e = [0, 1, 0], e = [0, 0, 1]. Standard basis vectors of R are e, e, · · · , e, where e, 1 ≤ j ≤ n, is the vector with zero components except that the j-th component equals 1. Each vector in R is a unique linear combination of the standard basis vectors. 1.3 Span Let v, v, · · · , v be vectors in R. The span of these vectors is the set of all linear combinations of them, and is denoted by sp(v, v, · · · , v). Example 3: the span of [1, 0] and [0, 1] is R. The span of v = [1,−2] and v = [7,−14] is the line along [1,−2] instead of R, because v is a scalar multiple of v (or v is in sp(v)). Example 4: Let v = [1, 3], w = [−2, 5], determine if [−1, 19] is in sp(v,w) and if so the coefficients of linear combination. The problem is same as finding a solution to [1,−19] = r v + s w = r[1, 3] + s[−2, 5] for two real numbers r and s. It follows from comparing the components that: r − 2s = −1, 3r + 5s = 19. Eliminating the r variable gives: 0 + 11s = 22, s = 2, then r = 3. Solution is unique. 2 1.4 Matlab Exercises Here are hands-on Matlab Exercises. Exercise 1: enter vectors in Matlab: v = [1 2 3 4]

{{< /ci-details >}}

{{< ci-details summary="Community Preserving Network Embedding (Xiao Wang et al., 2017)">}}

Xiao Wang, Peng Cui, Jing Wang, J. Pei, Wenwu Zhu, Shiqiang Yang. (2017)  
**Community Preserving Network Embedding**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/d3e0d596efd9d19b93d357565a68dfa925dce2bb)  
Influential Citation Count (56), SS-ID (d3e0d596efd9d19b93d357565a68dfa925dce2bb)  

**ABSTRACT**  
Network embedding, aiming to learn the low-dimensional representations of nodes in networks, is of paramount importance in many real applications. One basic requirement of network embedding is to preserve the structure and inherent properties of the networks. While previous network embedding methods primarily preserve the microscopic structure, such as the first- and second-order proximities of nodes, the mesoscopic community structure, which is one of the most prominent feature of networks, is largely ignored. In this paper, we propose a novel Modularized Nonnegative Matrix Factorization (M-NMF) model to incorporate the community structure into network embedding. We exploit the consensus relationship between the representations of nodes and community structure, and then jointly optimize NMF based representation learning model and modularity based community detection model in a unified framework, which enables the learned representations of nodes to preserve both of the microscopic and community structures. We also provide efficient updating rules to infer the parameters of our model, together with the correctness and convergence guarantees. Extensive experimental results on a variety of real-world networks show the superior performance of the proposed method over the state-of-the-arts.   

{{< /ci-details >}}

{{< ci-details summary="HINE: Heterogeneous Information Network Embedding (Yuxin Chen et al., 2017)">}}

Yuxin Chen, Chenguang Wang. (2017)  
**HINE: Heterogeneous Information Network Embedding**  
DASFAA  
[Paper Link](https://www.semanticscholar.org/paper/d8fbd38ec3c8aedf9964882a73e84d8a540de4b9)  
Influential Citation Count (0), SS-ID (d8fbd38ec3c8aedf9964882a73e84d8a540de4b9)  

{{< /ci-details >}}

{{< ci-details summary="Multiplicative latent factor models for description and prediction of social networks (Peter D. Hoff, 2009)">}}

Peter D. Hoff. (2009)  
**Multiplicative latent factor models for description and prediction of social networks**  
Comput. Math. Organ. Theory  
[Paper Link](https://www.semanticscholar.org/paper/dbe30a96b7db2df4e8f6c3492e2092c68feedcd6)  
Influential Citation Count (19), SS-ID (dbe30a96b7db2df4e8f6c3492e2092c68feedcd6)  

{{< /ci-details >}}

{{< ci-details summary="Fast Eigen-Functions Tracking on Dynamic Graphs (C. Chen et al., 2015)">}}

C. Chen, Hanghang Tong. (2015)  
**Fast Eigen-Functions Tracking on Dynamic Graphs**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/ddc3dc43095173a6b5f0a6f5fbdde79e1abac34d)  
Influential Citation Count (3), SS-ID (ddc3dc43095173a6b5f0a6f5fbdde79e1abac34d)  

**ABSTRACT**  
Many important graph parameters can be expressed as eigenfunctions of its adjacency matrix. Examples include epidemic threshold, graph robustness, etc. It is often of key importance to accurately monitor these parameters. For example, knowing that Ebola virus has already been brought to the US continent, to avoid the virus from spreading away, it is important to know which emerging connections among related people would cause great reduction on the epidemic threshold of the network. However, most, if not all, of the existing algorithms computing these measures assume that the input graph is static, despite the fact that almost all real graphs are evolving over time. In this paper, we propose two online algorithms to track the eigen-functions of a dynamic graph with linear complexity wrt the number of nodes and number of changed edges in the graph. The key idea is to leverage matrix perturbation theory to efficiently update the top eigen-pairs of the underlying graph without recomputing them from scratch at each time stamp. Experiment results demonstrate that our methods can reach up to 20× speedup with precision more than 80% for fairly long period of time.

{{< /ci-details >}}

{{< ci-details summary="A new status index derived from sociometric analysis (L. Katz, 1953)">}}

L. Katz. (1953)  
**A new status index derived from sociometric analysis**  
  
[Paper Link](https://www.semanticscholar.org/paper/f1d40a639d7f83d373f07b2bf4a96f0313b584d8)  
Influential Citation Count (240), SS-ID (f1d40a639d7f83d373f07b2bf4a96f0313b584d8)  

{{< /ci-details >}}

{{< ci-details summary="DeepWalk: online learning of social representations (Bryan Perozzi et al., 2014)">}}

Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2014)  
**DeepWalk: online learning of social representations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fff114cbba4f3ba900f33da574283e3de7f26c83)  
Influential Citation Count (1396), SS-ID (fff114cbba4f3ba900f33da574283e3de7f26c83)  

**ABSTRACT**  
We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

{{< /ci-details >}}

