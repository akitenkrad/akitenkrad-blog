---
draft: false
title: "Attributed Network Embedding for Learning in a Dynamic Environment"
date: 2022-06-12
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2017", "DS:BlogCatalog", "DS:Flickr", "DS:Epinions", "DS:DBLP"]
menu:
  sidebar:
    name: "Attributed Network Embedding for Learning in a Dynamic Environment"
    identifier: 20220612
    parent: 202206
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation
{{< citation >}}
Jundong Li, Harsh Dani, Xia Hu, Jiliang Tang, Yi Chang, and Huan Liu. 2017.  
Attributed Network Embedding for Learning in a Dynamic Environment.  
In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management (CIKM '17). Association for Computing Machinery, New York, NY, USA, 387–396.  
https://doi.org/10.1145/3132847.3132919
{{< /citation >}}

## Abstract
> Network embedding leverages the node proximity manifested to learn a low-dimensional node vector representation for each node in the network. The learned embeddings could advance various learning tasks such as node classification, network clustering, and link prediction. Most, if not all, of the existing works, are overwhelmingly performed in the context of plain and static networks. Nonetheless, in reality, network structure often evolves over time with addition/deletion of links and nodes. Also, a vast majority of real-world networks are associated with a rich set of node attributes, and their attribute values are also naturally changing, with the emerging of new content patterns and the fading of old content patterns. These changing characteristics motivate us to seek an effective embedding representation to capture network and attribute evolving patterns, which is of fundamental importance for learning in a dynamic environment. To our best knowledge, we are the first to tackle this problem with the following two challenges: (1) the inherently correlated network and node attributes could be noisy and incomplete, it necessitates a robust consensus representation to capture their individual properties and correlations; (2) the embedding learning needs to be performed in an online fashion to adapt to the changes accordingly. In this paper, we tackle this problem by proposing a novel dynamic attributed network embedding framework - DANE. In particular, DANE first provides an offline method for a consensus embedding and then leverages matrix perturbation theory to maintain the freshness of the end embedding results in an online manner. We perform extensive experiments on both synthetic and real attributed networks to corroborate the effectiveness and efficiency of the proposed framework.

## Background & Wat's New
- Dynamic Graphを扱うときの難しさ
  1. ネットワークのトポロジーとノードごとの特徴量は密接に結びついており，さらに生データはノイズが多く不完全である場合がほとんどであるため，そうした不完全性に対して頑健なモデルが求められる
  2. 各タイムステップごとにグラフをゼロから構築するのは効率が悪いため，オンラインで動作する効率的なアルゴリズムを開発する必要がある
- Contributions
  - Problem Formulations  
    - 動的グラフに対応したタスクの定式化を考案した
    - オフラインモデルは最初に初期化し，それをベースとしてオンラインモデルが最新の特徴を保持していく仕組みが主なアイディアである
  - Algorithms and Analysis
    - 新しいフレームワーク **DANE (Dynamic Attributed Network Embedding)** を提案した
    - ベースモデルとしてオフラインのEmbeddingモデルを採用し，Matrix Perturbation Theoryに基づいて時系列に沿ってベースモデルを更新していく
  - Evaluations
    - 実データを用いて，教師あり/なし両方のタスクにおいて精度評価を実施した
    - 既存手法と比較して，ClusteringおよびClassificationのタスクにおいて優れた精度を発揮した
    - 既存手法と比較して，実行速度を大きく改善した

## Dataset

- BlogCatalog
- Flickr
- Epinions
- DBLP

<figure>
  <img src="datasets.png" width="100%" />
  <figcaption>Detailed information of the datasets</figcaption>
</figure>

## Model Description

### Problem Formulation

$$
\begin{align*}
  \mathcal{U}^{(t)} &= \lbrace u\_1, u\_2, \ldots, u\_n \rbrace & \hspace{10pt} & \text{a set of }n\text{ nodes in the attributed network }\mathcal{G}^{(t)} \\\\
  A^{(t)} &\in \mathbb{R}^{n \times n} & \hspace{10pt} & \text{adjacency matrix to represent the network structure of }\mathcal{U}^{(t)} \\\\
  X^{(t)} &\in \mathbb{R}^{n \times d} & \hspace{10pt} & d\text{-dimentional attributes} \\\\
  &\text{where} \\\\
  &\mathcal{G}^{(t)} \mapsto \text{network at time step }t \\\\
  &\mathcal{F} = \lbrace f\_1, f\_2, \ldots, f\_d \rbrace & \hspace{10pt} & \text{list of attributes}
\end{align*}
$$

{{< box-with-title title="Problem 1. The offline model of DANE at time step $t$" >}}
given network topology $A^{(t)}$ and node attributes $X^{(t)}$, output attributed network embedding $Y^{(t)}$ for all nodes.
{{< /box-with-title >}}

{{< box-with-title title="Problem 2. The online model of DANE at time step $t + 1$" >}}
given network topology $A^{(t + 1)}$ and node attributes $X^{(t + 1)}$, and intermediate embedding results at time stemp $t$, output attributed network embedding $Y^{(t + 1)}$ for all nodes.
{{< /box-with-title >}}

### DANE: Offline Model

ノイズの多いネットワークのトポロジーと特徴量の情報を考慮する頑健なモデルを設計する  
Offline Modelは隣接行列を入力として，接続されたノード同士の距離がEmbedding空間において近くなるように，以下のかたちで定式化できる

$$
\begin{align*}
  f(A^{(t)}) &= Y\_A^{(t)} = [\boldsymbol{y}\_1, \boldsymbol{y}\_2, \ldots, \boldsymbol{y}\_n]^\mathsf{T} \hspace{10pt} (y\_i \in \mathbb{R}^k \hspace{5pt} (k \ll n))\\\\
   f &: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times k} \\\\
   & \text{where} \\\\
   & \text{minimize } \frac{1}{2} \sum\_{i,j} A^{(t)}\_{i,j} \lVert \boldsymbol{y}\_i - \boldsymbol{y}\_j \rVert \_2^2
\end{align*}
$$

上記の定式化は，次数行列 $D^{(t)}_A \hspace{5pt} (D\_{A\_{i,i}}^{(t)} = \sum\_{j=1}^n A^{(t)})$ およびラプラシアン $L\_A^{(t)} = D\_A^{(t)} - A^{(t)}$ を用いて，以下の固有方程式を解く問題に一般化することができる

$$
L\_A^{(t)} \boldsymbol{a} = \lambda D\_A^{(t)} \boldsymbol{a}
$$

固有ベクトル $\boldsymbol{1}$ に対応する固有値は $\lambda \_1 = 0$ であり，$0 = \lambda\_1 \leqq \lambda\_2 \leqq \ldots \leqq \lambda\_n$ となる  
したがって，$\boldsymbol{a}\_2$ から上位 $k$ 個の固有ベクトルをとってきて

$$
Y\_A^{(t)} = [\boldsymbol{a}\_2, \ldots, \boldsymbol{a}\_k, \boldsymbol{a}\_{k + 1}]
$$

となる

特徴量についても同様に，

$$
\begin{align*}
  X^{(t)}\_{\text{norm}} &= \text{normalize}(X^{(t)}) \\\\
  W^{(t)}\_{i, j} &= \frac{X^{(t)}\_i \cdot X^{(t)}\_j}{\lVert X^{(t)}\_i \rVert \lVert X^{(t)}\_j \rVert} \hspace{10pt} \text{(cosine similarity matrix)}\\\\
  L\_W^{(t)} \boldsymbol{b} &= \lambda D\_W^{(t)} \boldsymbol{b} \\\\
  Y\_X^{(t)} &= [\boldsymbol{b}\_2, \ldots, \boldsymbol{b}\_{k + 1}]
\end{align*}
$$

として，$\text{top-}k$ の固有ベクトルを得る  

$A^{(t)}$ と $X^{(t)}$ の相互依存関係を捉えるために，$A^{(t)}$ と $X^{(t)}$ の相関関係を最大化するような射影ベクトルを学習する

$$
\begin{align*}
  \max\_{\boldsymbol{p}\_A^{(t)}, \boldsymbol{p}\_X^{(t)}} & {\boldsymbol{p}\_A^{(t)}}^\mathsf{T} {Y\_A^{(t)}}^\mathsf{T} Y\_A^{(t)} \boldsymbol{p}\_A^{(t)} + {\boldsymbol{p}\_A^{(t)}}^\mathsf{T} {Y\_A^{(t)}}^\mathsf{T} Y\_X^{(t)} \boldsymbol{p}\_X^{(t)} + {\boldsymbol{p}\_X^{(t)}}^\mathsf{T} {Y\_X^{(t)}}^\mathsf{T} Y\_A^{(t)} \boldsymbol{p}\_A^{(t)} + {\boldsymbol{p}\_X^{(t)}}^\mathsf{T} {Y\_X^{(t)}}^\mathsf{T} Y\_X^{(t)} \boldsymbol{p}\_X^{(t)} \\\\
  & \text{where} \\\\
  & {\boldsymbol{p}\_A^{(t)}}^\mathsf{T} {Y\_A^{(t)}}^\mathsf{T} Y\_A^{(t)} \boldsymbol{p}\_A^{(t)} + {\boldsymbol{p}\_X^{(t)}}^\mathsf{T} {Y\_X^{(t)}}^\mathsf{T} Y\_X^{(t)} \boldsymbol{p}\_X^{(t)} = 1
\end{align*}
$$

上記の式は，ラグランジュの未定乗数法を用いることによって固有値問題に変換できる

$$
\begin{align*}
  \Leftrightarrow & \left[\begin{array}{c}
    {Y\_A^{(t)}}^\mathsf{T} Y\_A^{(t)} & {Y\_A^{(t)}}^\mathsf{T} Y\_X^{(t)} \\\\
    {Y\_X^{(t)}}^\mathsf{T} Y\_A^{(t)} & {Y\_X^{(t)}}^\mathsf{T} Y\_X^{(t)}
  \end{array}\right]\left[\begin{array}{c}
  \boldsymbol{p}\_A^{(t)} \\\\
  \boldsymbol{p}\_X^{(t)} \\\\
  \end{array}\right] \\\\
  \Leftrightarrow & \hspace{5pt} \gamma \left[\begin{array}{c}
    {Y\_A^{(t)}}^\mathsf{T} Y\_A^{(t)} & 0 \\\\
    0 & {Y\_A^{(t)}}^\mathsf{T} Y\_X^{(t)} \\\\
  \end{array}\right]\left[\begin{array}{c}
  \boldsymbol{p}\_A^{(t)} \\\\
  \boldsymbol{p}\_X^{(t)} \\\\
  \end{array}\right] \hspace{10pt} (\gamma \text{ はラグランジュ乗数})
\end{align*}
$$

上記の固有値から $\text{top-}l$ の固有ベクトルを取り出すとして，最終的にOffline ModelのEmbeddingを求めるには，以下を計算すれば良い

$$
\begin{align*}
  Y^{(t)} &= \left[ Y\_A^{(t)}, Y\_X^{(t)}\right] \times P^{(t)} \\\\
  & \text{where} \\\\
  & P^{(t)} \in \mathbb{R}^{2k \times l}
\end{align*}
$$

### Online Model of DANE

実際のネットワークは，これまでのところ例外なく"**スムーズに**"変化する  
そこで，タイムステップ $t$ と $t + 1$ の間の微小な変化を $\Delta A$，$\Delta X$ と表すこととする  
次数行列とラプラシアンは以下のようになる

$$
\begin{align*}
  D\_A^{(t + 1)} = D\_A^{(t)} + \Delta D\_A, & \hspace{10pt} & L\_A^{(t + 1)} = L\_A^{(t)} + \Delta L\_A \\\\
  D\_X^{(t + 1)} = D\_X^{(t)} + \Delta D\_X, & \hspace{10pt} & L\_X^{(t + 1)} = L\_X^{(t)} + \Delta L\_X
\end{align*}
$$

Offline Modelは隣接行列と特徴量行列に関して，$\text{top-}k$ の固有値ベクトルを求める問題に帰着した  
Online Modelは $text{top-}k$ の固有値ベクトルを効率的にアップデートする

Matrix Perturbation Theoryに基づき，以下の式を得る

$$
(L\_A^{(t)} + \Delta L\_A)(\boldsymbol{a} + \Delta \boldsymbol{a}) = (\lambda + \Delta \lambda)(D\_A^{(t)} + \Delta D\_A)(\boldsymbol{a} + \Delta \boldsymbol{a})
$$

{{< ci-details summary="Matrix Perturbation Theory (V. N. Bogaevski et al., 1991)">}}
V. N. Bogaevski, A. Povzner. (1991)  
**Matrix Perturbation Theory**  
[Paper Link](https://www.semanticscholar.org/paper/6c09c25131ac2e7f01fd14ce2a576c209f8ad23e)  
Influential Citation Count (88), SS-ID (6c09c25131ac2e7f01fd14ce2a576c209f8ad23e)  
{{< /ci-details >}}
<br/>

ある固有値と固有ベクトルのペア $(\lambda\_i, \boldsymbol{a}\_i)$ について見れば，

$$
(L\_A^{(t)} + \Delta L\_A)(\boldsymbol{a}\_i + \Delta \boldsymbol{a}\_i) = (\lambda\_i + \Delta \lambda\_i)(D\_A^{(t)} + \Delta D\_A)(\boldsymbol{a}\_i + \Delta \boldsymbol{a}\_i)
$$

となルので，次は $(\Delta \boldsymbol{a}\_i, \Delta \lambda\_i)$ の計算方法が課題となる

#### Computgin the change of eigenvalue $\Delta \lambda\_i$

$$
\begin{align*}
  & L\_A^{(t)} \boldsymbol{a}\_i + \Delta L\_A \boldsymbol{a}\_i + L\_A^{(t)} \Delta \boldsymbol{a}\_i + \Delta L\_A \Delta \boldsymbol{a}\_i \\\\
  = & \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i + \lambda\_i \Delta D\_A \boldsymbol{a}\_i + \Delta \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i + \Delta \lambda\_i \Delta D\_A \boldsymbol{a}\_i + (\lambda\_i D\_A^{(t)} + \lambda\_i \Delta D\_A + \Delta \lambda\_i D\_A^{(t)} + \Delta \lambda\_i \Delta D\_A) \Delta \boldsymbol{a}\_i
\end{align*}
$$

$\Delta \lambda\_i \Delta D\_A \boldsymbol{a}\_i, \lambda\_i \Delta D\_A \Delta \boldsymbol{a}\_i, \Delta \lambda\_i D\_A^{(t)} \Delta \boldsymbol{a}\_i, \Delta \lambda\_i \Delta D\_A \Delta \boldsymbol{a}\_i$ は計算結果に与える影響が小さいので除外して考えて良い  
$L\_A^{(t)} \boldsymbol{a}\_i = \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i$ であるから

$$
\begin{equation}
\Delta L\_A \boldsymbol{a}\_i + L\_A^{(t)} \Delta \boldsymbol{a}\_i = \lambda\_i \Delta D\_A \boldsymbol{a}\_i + \Delta \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i + \lambda\_i D\_A^{(t)} \Delta \boldsymbol{a}\_i
\end{equation}
$$

となる  
両辺に $\boldsymbol{a}^\mathsf{T}$ をかけ，また $L\_A^{(t)}$ および $D\_A^{(t)}$ は対称行列であることから，

$$
\begin{align*}
  {\boldsymbol{a}\_i}^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i + {\boldsymbol{a}\_i}^\mathsf{T} L\_A^{(t)} \Delta \boldsymbol{a}\_i & = {\boldsymbol{a}\_i}^\mathsf{T} \lambda\_i \Delta D\_A \boldsymbol{a}\_i + {\boldsymbol{a}\_i}^\mathsf{T} \Delta \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i + {\boldsymbol{a}\_i}^\mathsf{T} \lambda\_i D\_A^{(t)} \Delta \boldsymbol{a}\_i \\\\
  \boldsymbol{a}^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i & = \lambda\_i {\boldsymbol{a}\_i}^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i + \Delta \lambda\_i {\boldsymbol{a}\_i}^\mathsf{T} D\_A^{(t)} \boldsymbol{a}\_i \\\\
  & \because {\boldsymbol{a}\_i}^\mathsf{T}L\_A^{(t)} \Delta \boldsymbol{a}\_i = \lambda\_i {\boldsymbol{a}\_i}^\mathsf{T} D\_A^{(t)} \Delta \boldsymbol{a}\_i
\end{align*}
$$

したがって，

$$
\Delta \lambda\_i = \frac{{\boldsymbol{a}\_i}^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i - \lambda\_i {\boldsymbol{a}\_i}^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i}{{\boldsymbol{a}\_i}^\mathsf{T} D\_A^{(t)} \boldsymbol{a}\_i }
$$

を得る

ここで，

{{< box-with-title title="Theorem 1.1" >}}
In the generalized eigen-problem

$$
A\boldsymbol{v} = \lambda B \boldsymbol{v}
$$

if $A$ and $B$ are both **Hermitian matrices** and $B$ is a **positive-semidefinite matrix**, 

1. the eigenvalue $\lambda$ are real
2. eigenvectors $\boldsymbol{v}\_j (i \neq j)$ are $B$-orthogonal such that $\boldsymbol{v}\_i^\mathsf{T} B \boldsymbol{v}\_j = 0$ and $\boldsymbol{v}\_i^\mathsf{T} B \boldsymbol{v}\_i = 1$
{{< /box-with-title>}}

より，

{{< box-with-title title="Corollary 1.2" >}}
$$
\boldsymbol{a}\_i^\mathsf{T} D\_A^{(t)} \boldsymbol{a}\_j = \left\lbrace \begin{array}{c}
1 & (i = j) \\\\
0 & (i \neq j) \\\\
\end{array}\right.
$$
{{</ box-with-title >}}

を得るので，最終的に

$$
\Delta \lambda\_i = {\boldsymbol{a}\_i}^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i - \lambda\_i {\boldsymbol{a}\_i}^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i
$$

となる．

#### Computgin the change of eigenvector $\Delta \boldsymbol{a}\_i$

2時点間における固有値ベクトルの列方向の微小な変化 $\Delta \boldsymbol{a}\_i$ は以下のように表される．

$$
  \Delta \boldsymbol{a}\_i = \sum\_{j=2}^{k+1} \alpha\_{ij} \boldsymbol{a}\_j
$$

$\alpha\_{ij}$ は $j$ 番目の固有値ベクトルから $i$ 番目の固有値ベクトルを推定する場合の重みである．  
この重みは次のようにして計算される．

(1)式より，

$$
\begin{align*}
  \Delta L\_A \boldsymbol{a}\_i + D\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_ij \lambda\_j \boldsymbol{a}\_j &= \lambda\_i \Delta D\_A \boldsymbol{a}\_i + \Delta \lambda\_i D\_A^{(t)} \boldsymbol{a}\_i + \lambda\_i D\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_{ij} \boldsymbol{a}\_j \\\\
  & \begin{array}{l} 
  \because \hspace{5pt} \Delta \boldsymbol{a}\_i = \sum\_{j=2}^{k+1} \alpha\_{ij} \boldsymbol{a}\_j \\\\
  \because \hspace{5pt} L\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_{ij} \boldsymbol{a}\_j = D\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_{ij} \lambda\_j \boldsymbol{a}\_j
  \end{array}
\end{align*}
$$

$\boldsymbol{a}\_p^\mathsf{T} \hspace{5pt} (2 \leqq p \leqq k + 1, p \neq i)$ を両辺にかけることで，

$$
\begin{align*}
  & \boldsymbol{a}\_p^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i + \boldsymbol{a}\_p^\mathsf{T} D\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_ij \lambda\_j \boldsymbol{a}\_j \\\\
  = \hspace{5pt} &\lambda\_i \boldsymbol{a}\_p^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i + \Delta \lambda\_i \boldsymbol{a}\_p^\mathsf{T} D\_A^{(t)} \boldsymbol{a}\_i + \lambda\_i \boldsymbol{a}\_p^\mathsf{T} D\_A^{(t)} \sum\_{j=2}^{k+1} \alpha\_{ij} \boldsymbol{a}\_j \\\\
  \Rightarrow \hspace{10pt} & \boldsymbol{a}\_p^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i + \alpha\_{ip} \lambda\_p = \lambda\_i \boldsymbol{a}\_p^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i + \alpha\_{ip} \lambda\_i
\end{align*}
$$

したがって，

$$
\alpha\_{ip} = \frac{\boldsymbol{a}\_p^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i - \lambda\_i \boldsymbol{a}\_p ^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i}{\lambda\_i - \lambda\_p}
$$

を得る．  
正規直交条件より $(\boldsymbol{a}\_i + \Delta \boldsymbol{a}\_i)^\mathsf{T} (D\_A + \Delta D\_A) (\boldsymbol{a}\_i + \Delta \boldsymbol{a}\_i) = 1$ であるので，

$$
2 \boldsymbol{a}\_i^\mathsf{T} D\_A^{(t)} \Delta \boldsymbol{a}\_i + \boldsymbol{a}\_i^\mathsf{T} \Delta D\_A^{(t)} \boldsymbol{a}\_i = 0
$$

より，

$$
\alpha\_{ii} = -\frac{1}{2} \boldsymbol{a}\_i^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i
$$

となる．  
以上より，

$$
\Delta \boldsymbol{a}\_i = -\frac{1}{2} \boldsymbol{a}\_i^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i \boldsymbol{a}\_i + \sum\_{j=2, j \neq i}^{k+1} \left( \frac{\boldsymbol{a}\_p^\mathsf{T} \Delta L\_A \boldsymbol{a}\_i - \lambda\_i \boldsymbol{a}\_p ^\mathsf{T} \Delta D\_A \boldsymbol{a}\_i}{\lambda\_i - \lambda\_p} \right) \boldsymbol{a}\_j
$$

を得る．

$(\Delta \lambda\_i, \Delta \boldsymbol{a}\_i)$ が求まったので，Embeddingを更新するアルゴリズムは以下の疑似コードで表現することができる．

<figure>
  <img src="algorithm-1.png" width="100%" />
  <figcaption>Algorithm 1</figcaption>
</figure>

## Results

### Unsupervised Network Clustering

<figure>
  <img src="results-1.png" width="100%" />
  <figcaption>Clustering results (%)</figcaption>
</figure>

### Supervised Node Classification

<figure>
  <img src="results-2.png" width="100%" />
  <figcaption>Classification results (%)</figcaption>
</figure>

## References


{{< ci-details summary="Learning latent representations of nodes for classifying in heterogeneous social networks (Yann Jacob et al., 2014)">}}

Yann Jacob, Ludovic Denoyer, P. Gallinari. (2014)  
**Learning latent representations of nodes for classifying in heterogeneous social networks**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/030d436cb0465fd6cec0d5140b2534a8f1b8aeca)  
Influential Citation Count (6), SS-ID (030d436cb0465fd6cec0d5140b2534a8f1b8aeca)  

**ABSTRACT**  
Social networks are heterogeneous systems composed of different types of nodes (e.g. users, content, groups, etc.) and relations (e.g. social or similarity relations). While learning and performing inference on homogeneous networks have motivated a large amount of research, few work exists on heterogeneous networks and there are open and challenging issues for existing methods that were previously developed for homogeneous networks. We address here the specific problem of nodes classification and tagging in heterogeneous social networks, where different types of nodes are considered, each type with its own label or tag set. We propose a new method for learning node representations onto a latent space, common to all the different node types. Inference is then performed in this latent space. In this framework, two nodes connected in the network will tend to share similar representations regardless of their types. This allows bypassing limitations of the methods based on direct extensions of homogenous frameworks and exploiting the dependencies and correlations between the different node types. The proposed method is tested on two representative datasets and compared to state-of-the-art methods and to baselines.

{{< /ci-details >}}

{{< ci-details summary="A Survey on Multi-view Learning (Chang Xu et al., 2013)">}}

Chang Xu, D. Tao, Chao Xu. (2013)  
**A Survey on Multi-view Learning**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/032d67d27ecacbf6c5b82eb67e5d02d81fb43a7a)  
Influential Citation Count (53), SS-ID (032d67d27ecacbf6c5b82eb67e5d02d81fb43a7a)  

**ABSTRACT**  
In recent years, a great many methods of learning from multi-view data by considering the diversity of different views have been proposed. These views may be obtained from multiple sources or different feature subsets. In trying to organize and highlight similarities and differences between the variety of multi-view learning approaches, we review a number of representative multi-view learning algorithms in different areas and classify them into three groups: 1) co-training, 2) multiple kernel learning, and 3) subspace learning. Notably, co-training style algorithms train alternately to maximize the mutual agreement on two distinct views of the data; multiple kernel learning algorithms exploit kernels that naturally correspond to different views and combine kernels either linearly or non-linearly to improve learning performance; and subspace learning algorithms aim to obtain a latent subspace shared by multiple views by assuming that the input views are generated from this latent subspace. Though there is significant variance in the approaches to integrating multiple views to improve learning performance, they mainly exploit either the consensus principle or the complementary principle to ensure the success of multi-view learning. Since accessing multiple views is the fundament of multi-view learning, with the exception of study on learning a model from multiple views, it is also valuable to study how to construct multiple views and how to evaluate these views. Overall, by exploring the consistency and complementary properties of different views, multi-view learning is rendered more effective, more promising, and has better generalization ability than single-view learning.

{{< /ci-details >}}

{{< ci-details summary="Colibri: fast mining of large static and dynamic graphs (Hanghang Tong et al., 2008)">}}

Hanghang Tong, S. Papadimitriou, Jimeng Sun, Philip S. Yu, C. Faloutsos. (2008)  
**Colibri: fast mining of large static and dynamic graphs**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/0637aede560d3670dd8b31df81e39fce0de552d7)  
Influential Citation Count (17), SS-ID (0637aede560d3670dd8b31df81e39fce0de552d7)  

**ABSTRACT**  
Low-rank approximations of the adjacency matrix of a graph are essential in finding patterns (such as communities) and detecting anomalies. Additionally, it is desirable to track the low-rank structure as the graph evolves over time, efficiently and within limited storage. Real graphs typically have thousands or millions of nodes, but are usually very sparse. However, standard decompositions such as SVD do not preserve sparsity. This has led to the development of methods such as CUR and CMD, which seek a non-orthogonal basis by sampling the columns and/or rows of the sparse matrix.  However, these approaches will typically produce overcomplete bases, which wastes both space and time. In this paper we propose the family of Colibri methods to deal with these challenges. Our version for static graphs, Colibri-S, iteratively finds a non-redundant basis and we prove that it has no loss of accuracy compared to the best competitors (CUR and CMD), while achieving significant savings in space and time: on real data, Colibri-S requires much less space and is orders of magnitude faster (in proportion to the square of the number of non-redundant columns). Additionally, we propose an efficient update algorithm for dynamic, time-evolving graphs, Colibri-D. Our evaluation on a large, real network traffic dataset shows that Colibri-D is over 100 times faster than the best published competitor (CMD).

{{< /ci-details >}}

{{< ci-details summary="LINE: Large-scale Information Network Embedding (Jian Tang et al., 2015)">}}

Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, Q. Mei. (2015)  
**LINE: Large-scale Information Network Embedding**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/0834e74304b547c9354b6d7da6fa78ef47a48fa8)  
Influential Citation Count (864), SS-ID (0834e74304b547c9354b6d7da6fa78ef47a48fa8)  

**ABSTRACT**  
This paper studies the problem of embedding very large information networks into low-dimensional vector spaces, which is useful in many tasks such as visualization, node classification, and link prediction. Most existing graph embedding methods do not scale for real world information networks which usually contain millions of nodes. In this paper, we propose a novel network embedding method called the ``LINE,'' which is suitable for arbitrary types of information networks: undirected, directed, and/or weighted. The method optimizes a carefully designed objective function that preserves both the local and global network structures. An edge-sampling algorithm is proposed that addresses the limitation of the classical stochastic gradient descent and improves both the effectiveness and the efficiency of the inference. Empirical experiments prove the effectiveness of the LINE on a variety of real-world information networks, including language networks, social networks, and citation networks. The algorithm is very efficient, which is able to learn the embedding of a network with millions of vertices and billions of edges in a few hours on a typical single machine. The source code of the LINE is available online\footnote{\url{https://github.com/tangjianpku/LINE}}.

{{< /ci-details >}}

{{< ci-details summary="Toward Time-Evolving Feature Selection on Dynamic Networks (Liu Jundong et al., 2016)">}}

Liu Jundong, Huchuan Xia, Jian Ling, Liu Huan. (2016)  
**Toward Time-Evolving Feature Selection on Dynamic Networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/0c524fc7efd5bfa82a711734c8d70f17158518fb)  
Influential Citation Count (0), SS-ID (0c524fc7efd5bfa82a711734c8d70f17158518fb)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Robust Unsupervised Feature Selection on Networked Data (Jundong Li et al., 2016)">}}

Jundong Li, Xia Hu, Liang Wu, Huan Liu. (2016)  
**Robust Unsupervised Feature Selection on Networked Data**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/0dd8ce441c90280637d6d2b6df24747a0ff6ae11)  
Influential Citation Count (4), SS-ID (0dd8ce441c90280637d6d2b6df24747a0ff6ae11)  

**ABSTRACT**  
Feature selection has shown its effectiveness to prepare high-dimensional data for many data mining and machine learning tasks. Traditional feature selection algorithms are mainly based on the assumption that data instances are independent and identically distributed. However, this assumption is invalid in networked data since instances are not only associated with high dimensional features but also inherently interconnected with each other. In addition, obtaining label information for networked data is time consuming and labor intensive. Without label information to direct feature selection, it is difficult to assess the feature relevance. In contrast to the scarce label information, link information in networks are abundant and could help select relevant features. However, most networked data has a lot of noisy links, resulting in the feature selection algorithms to be less effective. To address the above mentioned issues, we propose a robust unsupervised feature selection framework NetFS for networked data, which embeds the latent representation learning into feature selection. Therefore, content information is able to help mitigate the negative effects from noisy links in learning latent representations, while good latent representations in turn can contribute to extract more meaningful features. In other words, both phases could cooperate and boost each other. Experimental results on realworld datasets demonstrate the effectiveness of the proposed

{{< /ci-details >}}

{{< ci-details summary="Accelerated Attributed Network Embedding (Xiao Huang et al., 2017)">}}

Xiao Huang, Jundong Li, Xia Hu. (2017)  
**Accelerated Attributed Network Embedding**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/0efb659b15737c76a2fc50010a694123f6c45f64)  
Influential Citation Count (37), SS-ID (0efb659b15737c76a2fc50010a694123f6c45f64)  

**ABSTRACT**  
Network embedding is to learn low-dimensional vector representations for nodes in a network. It has shown to be effective in a variety of tasks such as node classification and link prediction. While embedding algorithms on pure networks have been intensively studied, in many real-world applications, nodes are often accompanied with a rich set of attributes or features, aka attributed networks. It has been observed that network topological structure and node attributes are often strongly correlated with each other. Thus modeling and incorporating node attribute proximity into network embedding could be potentially helpful, though non-trivial, in learning better vector representations. Meanwhile, real-world networks often contain a large number of nodes and features, which put demands on the scalability of embedding algorithms. To bridge the gap, in this paper, we propose an accelerated attributed network embedding algorithm AANE, which enables the joint learning process to be done in a distributed manner by decomposing the complex modeling and optimization into many sub-problems. Experimental results on several real-world datasets demonstrate the effectiveness and efficiency of the proposed algorithm.

{{< /ci-details >}}

{{< ci-details summary="Human mobility, social ties, and link prediction (Dashun Wang et al., 2011)">}}

Dashun Wang, D. Pedreschi, Chaoming Song, F. Giannotti, A. Barabasi. (2011)  
**Human mobility, social ties, and link prediction**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/117fc73987a2808a90bf093be5730e64f7514d30)  
Influential Citation Count (47), SS-ID (117fc73987a2808a90bf093be5730e64f7514d30)  

**ABSTRACT**  
Our understanding of how individual mobility patterns shape and impact the social network is limited, but is essential for a deeper understanding of network dynamics and evolution. This question is largely unexplored, partly due to the difficulty in obtaining large-scale society-wide data that simultaneously capture the dynamical information on individual movements and social interactions. Here we address this challenge for the first time by tracking the trajectories and communication records of 6 Million mobile phone users. We find that the similarity between two individuals' movements strongly correlates with their proximity in the social network. We further investigate how the predictive power hidden in such correlations can be exploited to address a challenging problem: which new links will develop in a social network. We show that mobility measures alone yield surprising predictive power, comparable to traditional network-based measures. Furthermore, the prediction accuracy can be significantly improved by learning a supervised classifier based on combined mobility and network measures. We believe our findings on the interplay of mobility patterns and social ties offer new perspectives on not only link prediction but also network dynamics.

{{< /ci-details >}}

{{< ci-details summary="On Node Classification in Dynamic Content-based Networks (C. Aggarwal et al., 2011)">}}

C. Aggarwal, Nan Li. (2011)  
**On Node Classification in Dynamic Content-based Networks**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/1c7ec5f28d034953edbd1040e46a1e1b8a90aac2)  
Influential Citation Count (3), SS-ID (1c7ec5f28d034953edbd1040e46a1e1b8a90aac2)  

**ABSTRACT**  
In recent years, a large amount of information has become available online in the form of web documents, social networks, blogs, or other kinds of social entities. Such networks are large, heterogeneous, and often contain a huge number of links. This linkage structure encodes rich structural information about the underlying topical behavior of the network. Such networks are often dynamic and evolve rapidly over time. Much of the work in the literature has focussed either on the problem of classification with purely text behavior, or on the problem of classification with purely the linkage behavior of the underlying graph. Furthermore, the work in the literature is mostly designed for the problem of static networks. However, a given network may be quite diverse, and the use of either content or structure could be more or less effective in different parts of the network. In this paper, we examine the problem of node classification in dynamic information networks with both text content and links. Our techniques use a random walk approach in conjunction with the content of the network in order to facilitate an effective classification process. This results in an effective approach which is more robust to variations in content and linkage structure. Our approach is dynamic, and can be applied to networks which are updated incrementally. Our results suggest that an approach which is based on a combination of content and links is extremely robust and effective. We present experimental results illustrating the effectiveness and efficiency of our approach.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Streaming Feature Selection in Social Media (Jundong Li et al., 2015)">}}

Jundong Li, Xia Hu, Jiliang Tang, Huan Liu. (2015)  
**Unsupervised Streaming Feature Selection in Social Media**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/1faabf9f6657cc2d295b3cf0545699168591f6d0)  
Influential Citation Count (5), SS-ID (1faabf9f6657cc2d295b3cf0545699168591f6d0)  

**ABSTRACT**  
The explosive growth of social media sites brings about massive amounts of high-dimensional data. Feature selection is effective in preparing high-dimensional data for data analytics. The characteristics of social media present novel challenges for feature selection. First, social media data is not fully structured and its features are usually not predefined, but are generated dynamically. For example, in Twitter, slang words (features) are created everyday and quickly become popular within a short period of time. It is hard to directly apply traditional batch-mode feature selection methods to find such features. Second, given the nature of social media, label information is costly to collect. It exacerbates the problem of feature selection without knowing feature relevance. On the other hand, opportunities are also unequivocally present with additional data sources; for example, link information is ubiquitous in social media and could be helpful in selecting relevant features. In this paper, we study a novel problem to conduct unsupervised streaming feature selection for social media data. We investigate how to exploit link information in streaming feature selection, resulting in a novel unsupervised streaming feature selection framework USFS. Experimental results on two real-world social media datasets show the effectiveness and efficiency of the proposed framework comparing with the state-of-the-art unsupervised feature selection algorithms.

{{< /ci-details >}}

{{< ci-details summary="Birds of a Feather: Homophily in Social Networks (M. McPherson et al., 2001)">}}

M. McPherson, L. Smith-Lovin, J. Cook. (2001)  
**Birds of a Feather: Homophily in Social Networks**  
  
[Paper Link](https://www.semanticscholar.org/paper/228bafce55e6f1cbe2c1df75b1949a1fb9c93eb3)  
Influential Citation Count (699), SS-ID (228bafce55e6f1cbe2c1df75b1949a1fb9c93eb3)  

**ABSTRACT**  
Similarity breeds connection. This principle—the homophily principle—structures network ties of every type, including marriage, friendship, work, advice, support, information transfer, exchange, comembership, and other types of relationship. The result is that people's personal networks are homogeneous with regard to many sociodemographic, behavioral, and intrapersonal characteristics. Homophily limits people's social worlds in a way that has powerful implications for the information they receive, the attitudes they form, and the interactions they experience. Homophily in race and ethnicity creates the strongest divides in our personal environments, with age, religion, education, occupation, and gender following in roughly that order. Geographic propinquity, families, organizations, and isomorphic positions in social systems all create contexts in which homophilous relations form. Ties between nonsimilar individuals also dissolve at a higher rate, which sets the stage for the formation of niches (localize...

{{< /ci-details >}}

{{< ci-details summary="ReAct: Online Multimodal Embedding for Recency-Aware Spatiotemporal Activity Modeling (Chao Zhang et al., 2017)">}}

Chao Zhang, Keyang Zhang, Quan Yuan, Fangbo Tao, Luming Zhang, T. Hanratty, Jiawei Han. (2017)  
**ReAct: Online Multimodal Embedding for Recency-Aware Spatiotemporal Activity Modeling**  
SIGIR  
[Paper Link](https://www.semanticscholar.org/paper/3094fbbea48a5794a07d071a346ef836c0ae10ac)  
Influential Citation Count (5), SS-ID (3094fbbea48a5794a07d071a346ef836c0ae10ac)  

**ABSTRACT**  
Spatiotemporalactivity modeling is an important task for applications like tour recommendation and place search. The recently developed geographical topic models have demonstrated compelling results in using geo-tagged social media (GTSM) for spatiotemporal activity modeling. Nevertheless, they all operate in batch and cannot dynamically accommodate the latest information in the GTSM stream to reveal up-to-date spatiotemporal activities. We propose ReAct, a method that processes continuous GTSM streams and obtains recency-aware spatiotemporal activity models on the fly. Distinguished from existing topic-based methods, ReAct embeds all the regions, hours, and keywords into the same latent space to capture their correlations. To generate high-quality embeddings, it adopts a novel semi-supervised multimodal embedding paradigm that leverages the activity category information to guide the embedding process. Furthermore, as new records arrive continuously, it employs strategies to effectively incorporate the new information while preserving the knowledge encoded in previous embeddings. Our experiments on the geo-tagged tweet streams in two major cities have shown that ReAct significantly outperforms existing methods for location and activity retrieval tasks.

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
Influential Citation Count (1162), SS-ID (36ee2c8bd605afd48035d15fdc6b8c8842363376)  

**ABSTRACT**  
Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

{{< /ci-details >}}

{{< ci-details summary="On the eigen‐functions of dynamic graphs: Fast tracking and attribution algorithms (C. Chen et al., 2017)">}}

C. Chen, Hanghang Tong. (2017)  
**On the eigen‐functions of dynamic graphs: Fast tracking and attribution algorithms**  
Stat. Anal. Data Min.  
[Paper Link](https://www.semanticscholar.org/paper/3ae8f410d9838d995b5f5ac5d62e65038c712aee)  
Influential Citation Count (1), SS-ID (3ae8f410d9838d995b5f5ac5d62e65038c712aee)  

**ABSTRACT**  
Eigen‐functions are of key importance in graph mining since they can be used to approximate many graph parameters, such as node centrality, epidemic threshold, graph robustness, with high accuracy. As real‐world graphs are changing over time, those parameters may get sharp changes correspondingly. Taking virus propagation network for example, new connections between infected and susceptible people appear all the time, and some of the crucial infections may lead to large decreasing on the epidemic threshold of the network. As a consequence, the virus would spread around the network quickly. However, if we can keep track of the epidemic threshold as the graph structure changes, those crucial infections would be identified timely so that counter measures can be taken proactively to contain the spread process. In our paper, we propose two online eigen‐functions tracking algorithms which can effectively monitor those key parameters with linear complexity. Furthermore, we propose a general attribution analysis framework which can be used to identify important structural changes in the evolving process. In addition, we introduce an error estimation method for the proposed eigen‐functions tracking algorithms to estimate the tracking error at each time stamp. Finally, extensive evaluations are conducted to validate the effectiveness and efficiency of the proposed algorithms. © 2016 Wiley Periodicals, Inc. Statistical Analysis and Data Mining: The ASA Data Science Journal, 2016

{{< /ci-details >}}

{{< ci-details summary="Revisiting Semi-Supervised Learning with Graph Embeddings (Zhilin Yang et al., 2016)">}}

Zhilin Yang, William W. Cohen, R. Salakhutdinov. (2016)  
**Revisiting Semi-Supervised Learning with Graph Embeddings**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/3d846cb01f6a975554035d2210b578ca61344b22)  
Influential Citation Count (185), SS-ID (3d846cb01f6a975554035d2210b578ca61344b22)  

**ABSTRACT**  
We present a semi-supervised learning framework based on graph embeddings. Given a graph between instances, we train an embedding for each instance to jointly predict the class label and the neighborhood context in the graph. We develop both transductive and inductive variants of our method. In the transductive variant of our method, the class labels are determined by both the learned embeddings and input feature vectors, while in the inductive variant, the embeddings are defined as a parametric function of the feature vectors, so predictions can be made on instances not seen during training. On a large and diverse set of benchmark tasks, including text classification, distantly supervised entity extraction, and entity classification, we show improved performance over many of the existing models.

{{< /ci-details >}}

{{< ci-details summary="Label Informed Attributed Network Embedding (Xiao Huang et al., 2017)">}}

Xiao Huang, Jundong Li, Xia Hu. (2017)  
**Label Informed Attributed Network Embedding**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/44044556dae0e21cab058c18f704b15d33bd17c5)  
Influential Citation Count (47), SS-ID (44044556dae0e21cab058c18f704b15d33bd17c5)  

**ABSTRACT**  
Attributed network embedding aims to seek low-dimensional vector representations for nodes in a network, such that original network topological structure and node attribute proximity can be preserved in the vectors. These learned representations have been demonstrated to be helpful in many learning tasks such as network clustering and link prediction. While existing algorithms follow an unsupervised manner, nodes in many real-world attributed networks are often associated with abundant label information, which is potentially valuable in seeking more effective joint vector representations. In this paper, we investigate how labels can be modeled and incorporated to improve attributed network embedding. This is a challenging task since label information could be noisy and incomplete. In addition, labels are completely distinct with the geometrical structure and node attributes. The bewildering combination of heterogeneous information makes the joint vector representation learning more difficult. To address these issues, we propose a novel Label informed Attributed Network Embedding (LANE) framework. It can smoothly incorporate label information into the attributed network embedding while preserving their correlations. Experiments on real-world datasets demonstrate that the proposed framework achieves significantly better performance compared with the state-of-the-art embedding algorithms.

{{< /ci-details >}}

{{< ci-details summary="Community evolution in dynamic multi-mode networks (Lei Tang et al., 2008)">}}

Lei Tang, Huan Liu, Jianping Zhang, Z. Nazeri. (2008)  
**Community evolution in dynamic multi-mode networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/4bdebe5ea5d75741d598eba16c2c715147229225)  
Influential Citation Count (9), SS-ID (4bdebe5ea5d75741d598eba16c2c715147229225)  

**ABSTRACT**  
A multi-mode network typically consists of multiple heterogeneous social actors among which various types of interactions could occur. Identifying communities in a multi-mode network can help understand the structural properties of the network, address the data shortage and unbalanced problems, and assist tasks like targeted marketing and finding influential actors within or between groups. In general, a network and the membership of groups often evolve gradually. In a dynamic multi-mode network, both actor membership and interactions can evolve, which poses a challenging problem of identifying community evolution. In this work, we try to address this issue by employing the temporal information to analyze a multi-mode network. A spectral framework and its scalability issue are carefully studied. Experiments on both synthetic data and real-world large scale networks demonstrate the efficacy of our algorithm and suggest its generality in solving problems with complex relationships.

{{< /ci-details >}}

{{< ci-details summary="Social Spammer Detection in Microblogging (Xia Hu et al., 2013)">}}

Xia Hu, Jiliang Tang, Yanchao Zhang, Huan Liu. (2013)  
**Social Spammer Detection in Microblogging**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/5146b393523d2b5ec24b000d2b776a961524eb64)  
Influential Citation Count (11), SS-ID (5146b393523d2b5ec24b000d2b776a961524eb64)  

**ABSTRACT**  
The availability of microblogging, like Twitter and Sina Weibo, makes it a popular platform for spammers to unfairly overpower normal users with unwanted content via social networks, known as social spamming. The rise of social spamming can significantly hinder the use of microblogging systems for effective information dissemination and sharing. Distinct features of microblogging systems present new challenges for social spammer detection. First, unlike traditional social networks, microblogging allows to establish some connections between two parties without mutual consent, which makes it easier for spammers to imitate normal users by quickly accumulating a large number of "human" friends. Second, microblogging messages are short, noisy, and unstructured. Traditional social spammer detection methods are not directly applicable to microblogging. In this paper, we investigate how to collectively use network and content information to perform effective social spammer detection in microblogging. In particular, we present an optimization formulation that models the social network and content information in a unified framework. Experiments on a real-world Twitter dataset demonstrate that our proposed method can effectively utilize both kinds of information for social spammer detection.

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

{{< ci-details summary="Toward Personalized Relational Learning (Jundong Li et al., 2017)">}}

Jundong Li, Liang Wu, Osmar R Zaiane, Huan Liu. (2017)  
**Toward Personalized Relational Learning**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/665c6e93c3777a33598b04d47eea87028f755bab)  
Influential Citation Count (0), SS-ID (665c6e93c3777a33598b04d47eea87028f755bab)  

**ABSTRACT**  
Relational learning exploits relationships among instances manifested in a network to improve the predictive performance of many network mining tasks. Due to its empirical success, it has been widely applied in myriad domains. In many cases, individuals in a network are highly idiosyncratic. They not only connect to each other with a composite of factors but also are often described by some content information of high dimensionality specific to each individual. For example in social media, as user interests are quite diverse and personal; posts by different users could differ significantly. Moreover, social content of users is often of high dimensionality which may negatively degrade the learning performance. Therefore, it would be more appealing to tailor the prediction for each individual while alleviating the issue related to the curse of dimensionality. In this paper, we study a novel problem of Personalized Relational Learning and propose a principled framework PRL to personalize the prediction for each individual in a network. Specifically, we perform personalized feature selection and employ a small subset of discriminative features customized for each individual and some common features shared by all to build a predictive model. On this account, the proposed personalized model is more human interpretable. Experiments on realworld datasets show the superiority of the proposed PRL framework over traditional relational learning methods.

{{< /ci-details >}}

{{< ci-details summary="Network Studies of Social Influence (P. Marsden et al., 1993)">}}

P. Marsden, Noah E. Friedkin. (1993)  
**Network Studies of Social Influence**  
  
[Paper Link](https://www.semanticscholar.org/paper/6678fd2d27ef1263a97823193ea1add21a82844c)  
Influential Citation Count (25), SS-ID (6678fd2d27ef1263a97823193ea1add21a82844c)  

**ABSTRACT**  
Network analysts interested in social influence examine the social foundations for influence—the social relations that provide a basis for the alteration of an attitude or behavior by one network actor in response to another. This article contrasts two empirical accounts of social influence (structural cohesion and equivalence) and describes the social processes (e.g., identification, competition, and authority) presumed to undergird them. It then reviews mathematical models of influence processes involving networks and related statistical models used in data analysis. Particular attention is given to the “network effects” model. A number of empirical studies of social influence are reviewed. The article concludes by identifying several problems of specification, research design, and measurement and suggesting some research that would help to resolve these problems.

{{< /ci-details >}}

{{< ci-details summary="Directed Graph Embedding (Mo Chen et al., 2007)">}}

Mo Chen, Qiong Yang, Xiaoou Tang. (2007)  
**Directed Graph Embedding**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/6951786736a30f77a803c00e19cb2b848f56c85c)  
Influential Citation Count (3), SS-ID (6951786736a30f77a803c00e19cb2b848f56c85c)  

**ABSTRACT**  
In this paper, we propose the Directed Graph Embedding (DGE) method that embeds vertices on a directed graph into a vector space by considering the link structure of graphs. The basic idea is to preserve the locality property of vertices on a directed graph in the embedded space. We use the transition probability together with the stationary distribution of Markov random walks to measure such locality property. It turns out that by exploring the directed links of the graph using random walks, we can get an optimal embedding on the vector space that preserves the local affinity which is inherent in the directed graph. Experiments on both synthetic data and real-world Web page data are considered. The application of our method to Web page classification problems gets a significant improvement comparing with state-of-art methods.

{{< /ci-details >}}

{{< ci-details summary="Matrix Perturbation Theory (V. N. Bogaevski et al., 1991)">}}

V. N. Bogaevski, A. Povzner. (1991)  
**Matrix Perturbation Theory**  
  
[Paper Link](https://www.semanticscholar.org/paper/6c09c25131ac2e7f01fd14ce2a576c209f8ad23e)  
Influential Citation Count (88), SS-ID (6c09c25131ac2e7f01fd14ce2a576c209f8ad23e)  

{{< /ci-details >}}

{{< ci-details summary="Node Classification in Social Networks (Smriti Bhagat et al., 2011)">}}

Smriti Bhagat, Graham Cormode, S. Muthukrishnan. (2011)  
**Node Classification in Social Networks**  
Social Network Data Analytics  
[Paper Link](https://www.semanticscholar.org/paper/6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  
Influential Citation Count (12), SS-ID (6e45220c1f3a8a8cbf176a2fc722c7e8380d5dd4)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Recommending Groups to Users Using User-Group Engagement and Time-Dependent Matrix Factorization (X. Wang et al., 2016)">}}

X. Wang, R. Donaldson, C. Nell, Peter Gorniak, M. Ester, Jiajun Bu. (2016)  
**Recommending Groups to Users Using User-Group Engagement and Time-Dependent Matrix Factorization**  
AAAI  
[Paper Link](https://www.semanticscholar.org/paper/720e17a617602403d64167fcb3857035b8201373)  
Influential Citation Count (2), SS-ID (720e17a617602403d64167fcb3857035b8201373)  

**ABSTRACT**  
    Social networks often provide group features to help users with similar interests associate and consume content together. Recommending groups to users poses challenges due to their complex relationship: user-group affinity is typically measured implicitly and varies with time; similarly, group characteristics change as users join and leave. To tackle these challenges, we adapt existing matrix factorization techniques to learn user-group affinity based on two different implicit engagement metrics: (i) which group-provided content users consume; and (ii) which content users provide to groups. To capture the temporally extended nature of group engagement we implement a time-varying factorization. We test the assertion that latent preferences for groups and users are sparse in investigating elastic-net regularization. Experiments using data from DeviantArt indicate that the time-varying implicit engagement-based model provides the best top-K group recommendations, illustrating the benefit of the added model complexity.   

{{< /ci-details >}}

{{< ci-details summary="An Attention-based Collaboration Framework for Multi-View Network Representation Learning (Meng Qu et al., 2017)">}}

Meng Qu, Jian Tang, Jingbo Shang, Xiang Ren, Ming Zhang, Jiawei Han. (2017)  
**An Attention-based Collaboration Framework for Multi-View Network Representation Learning**  
CIKM  
[Paper Link](https://www.semanticscholar.org/paper/73d9ee3238a872af94d5a03f4d951234c90037ac)  
Influential Citation Count (19), SS-ID (73d9ee3238a872af94d5a03f4d951234c90037ac)  

**ABSTRACT**  
Learning distributed node representations in networks has been attracting increasing attention recently due to its effectiveness in a variety of applications. Existing approaches usually study networks with a single type of proximity between nodes, which defines a single view of a network. However, in reality there usually exists multiple types of proximities between nodes, yielding networks with multiple views. This paper studies learning node representations for networks with multiple views, which aims to infer robust node representations across different views. We propose a multi-view representation learning approach, which promotes the collaboration of different views and lets them vote for the robust representations. During the voting process, an attention mechanism is introduced, which enables each node to focus on the most informative views. Experimental results on real-world networks show that the proposed approach outperforms existing state-of-the-art approaches for network representation learning with a single view and other competitive approaches with multiple views.

{{< /ci-details >}}

{{< ci-details summary="The Symmetric Eigenvalue Problem (B. Parlett, 1981)">}}

B. Parlett. (1981)  
**The Symmetric Eigenvalue Problem**  
  
[Paper Link](https://www.semanticscholar.org/paper/785f6308eae0b2075162c1d3d47ff392f71537db)  
Influential Citation Count (254), SS-ID (785f6308eae0b2075162c1d3d47ff392f71537db)  

**ABSTRACT**  
According to Parlett, 'Vibrations are everywhere, and so too are the eigenvalues associated with them. As mathematical models invade more and more disciplines, we can anticipate a demand for eigenvalue calculations in an ever richer variety of contexts.' Anyone who performs these calculations will welcome the reprinting of Parlett's book (originally published in 1980). In this unabridged, amended version, Parlett covers aspects of the problem that are not easily found elsewhere. The chapter titles convey the scope of the material succinctly. The aim of the book is to present mathematical knowledge that is needed in order to understand the art of computing eigenvalues of real symmetric matrices, either all of them or only a few. The author explains why the selected information really matters and he is not shy about making judgments. The commentary is lively but the proofs are terse.

{{< /ci-details >}}

{{< ci-details summary="Rare Category Detection on Time-Evolving Graphs (Dawei Zhou et al., 2015)">}}

Dawei Zhou, Kangyang Wang, Nan Cao, Jingrui He. (2015)  
**Rare Category Detection on Time-Evolving Graphs**  
2015 IEEE International Conference on Data Mining  
[Paper Link](https://www.semanticscholar.org/paper/8221b9493b029deac55123aa50aa086a57dee05c)  
Influential Citation Count (0), SS-ID (8221b9493b029deac55123aa50aa086a57dee05c)  

**ABSTRACT**  
Rare category detection(RCD) is an important topicin data mining, focusing on identifying the initial examples fromrare classes in imbalanced data sets. This problem becomes more challenging when the data is presented as time-evolving graphs, as used in synthetic ID detection and insider threat detection. Most existing techniques for RCD are designed for static data sets, thus not suitable for time-evolving RCD applications. To address this challenge, in this paper, we first proposetwo incremental RCD algorithms, SIRD and BIRD. They arebuilt upon existing density-based techniques for RCD, andincrementally update the detection models, which provide 'timeflexible' RCD. Furthermore, based on BIRD, we propose amodified version named BIRD-LI to deal with the cases wherethe exact priors of the minority classes are not available. Wealso identify a critical task in RCD named query distribution. Itaims to allocate the limited budget into multiple time steps, suchthat the initial examples from the rare classes are detected asearly as possible with the minimum labeling cost. The proposedincremental RCD algorithms and various query distributionstrategies are evaluated empirically on both synthetic and real data.

{{< /ci-details >}}

{{< ci-details summary="Who to follow and why: link prediction with explanations (Nicola Barbieri et al., 2014)">}}

Nicola Barbieri, F. Bonchi, G. Manco. (2014)  
**Who to follow and why: link prediction with explanations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/8cedc418ded7c4f5b3a10d6e551478c7084940af)  
Influential Citation Count (10), SS-ID (8cedc418ded7c4f5b3a10d6e551478c7084940af)  

**ABSTRACT**  
User recommender systems are a key component in any on-line social networking platform: they help the users growing their network faster, thus driving engagement and loyalty. In this paper we study link prediction with explanations for user recommendation in social networks. For this problem we propose WTFW ("Who to Follow and Why"), a stochastic topic model for link prediction over directed and nodes-attributed graphs. Our model not only predicts links, but for each predicted link it decides whether it is a "topical" or a "social" link, and depending on this decision it produces a different type of explanation. A topical link is recommended between a user interested in a topic and a user authoritative in that topic: the explanation in this case is a set of binary features describing the topic responsible of the link creation. A social link is recommended between users which share a large social neighborhood: in this case the explanation is the set of neighbors which are more likely to be responsible for the link creation. Our experimental assessment on real-world data confirms the accuracy of WTFW in the link prediction and the quality of the associated explanations.

{{< /ci-details >}}

{{< ci-details summary="Statistical Comparisons of Classifiers over Multiple Data Sets (J. Demšar, 2006)">}}

J. Demšar. (2006)  
**Statistical Comparisons of Classifiers over Multiple Data Sets**  
J. Mach. Learn. Res.  
[Paper Link](https://www.semanticscholar.org/paper/8f1408d33858a78f90f9000a34856664fc639ae4)  
Influential Citation Count (909), SS-ID (8f1408d33858a78f90f9000a34856664fc639ae4)  

**ABSTRACT**  
While methods for comparing two learning algorithms on a single data set have been scrutinized for quite some time already, the issue of statistical tests for comparisons of more algorithms on multiple data sets, which is even more essential to typical machine learning studies, has been all but ignored. This article reviews the current practice and then theoretically and empirically examines several suitable tests. Based on that, we recommend a set of simple, yet safe and robust non-parametric tests for statistical comparisons of classifiers: the Wilcoxon signed ranks test for comparison of two classifiers and the Friedman test with the corresponding post-hoc tests for comparison of more classifiers over multiple data sets. Results of the latter can also be neatly presented with the newly introduced CD (critical difference) diagrams.

{{< /ci-details >}}

{{< ci-details summary="The link-prediction problem for social networks (D. Liben-Nowell et al., 2007)">}}

D. Liben-Nowell, J. Kleinberg. (2007)  
**The link-prediction problem for social networks**  
J. Assoc. Inf. Sci. Technol.  
[Paper Link](https://www.semanticscholar.org/paper/996dfa43f6982bcbff862276ef80cbca7515985a)  
Influential Citation Count (278), SS-ID (996dfa43f6982bcbff862276ef80cbca7515985a)  

**ABSTRACT**  
Given a snapshot of a social network, can we infer which new interactions among its members are likely to occur in the near future? We formalize this question as the link-prediction problem, and we develop approaches to link prediction based on measures for analyzing the “proximity” of nodes in a network. Experiments on large coauthorship networks suggest that information about future interactions can be extracted from network topology alone, and that fairly subtle measures for detecting node proximity can outperform more direct measures. © 2007 Wiley Periodicals, Inc.

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

{{< ci-details summary="Evolutionary Network Analysis (C. Aggarwal et al., 2014)">}}

C. Aggarwal, Karthik Subbian. (2014)  
**Evolutionary Network Analysis**  
ACM Comput. Surv.  
[Paper Link](https://www.semanticscholar.org/paper/a304fb9a1037bbc1038fa60de6917269c9941ca2)  
Influential Citation Count (7), SS-ID (a304fb9a1037bbc1038fa60de6917269c9941ca2)  

**ABSTRACT**  
Evolutionary network analysis has found an increasing interest in the literature because of the importance of different kinds of dynamic social networks, email networks, biological networks, and social streams. When a network evolves, the results of data mining algorithms such as community detection need to be correspondingly updated. Furthermore, the specific kinds of changes to the structure of the network, such as the impact on community structure or the impact on network structural parameters, such as node degrees, also needs to be analyzed. Some dynamic networks have a much faster rate of edge arrival and are referred to as network streams or graph streams. The analysis of such networks is especially challenging, because it needs to be performed with an online approach, under the one-pass constraint of data streams. The incorporation of content can add further complexity to the evolution analysis process. This survey provides an overview of the vast literature on graph evolution analysis and the numerous applications that arise in different contexts.

{{< /ci-details >}}

{{< ci-details summary="Radar: Residual Analysis for Anomaly Detection in Attributed Networks (Jundong Li et al., 2017)">}}

Jundong Li, Harsh Dani, Xia Hu, Huan Liu. (2017)  
**Radar: Residual Analysis for Anomaly Detection in Attributed Networks**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/a5160db2d31d47545fb68a4a17580969e1e02f80)  
Influential Citation Count (16), SS-ID (a5160db2d31d47545fb68a4a17580969e1e02f80)  

**ABSTRACT**  
Attributed networks are pervasive in different domains, ranging from social networks, gene regulatory networks to financial transaction networks. This kind of rich network representation presents challenges for anomaly detection due to the heterogeneity of two data representations. A vast majority of existing algorithms assume certain properties of anomalies are given a prior. Since various types of anomalies in real-world attributed networks coexist, the assumption that priori knowledge regarding anomalies is available does not hold. In this paper, we investigate the problem of anomaly detection in attributed networks generally from a residual analysis perspective, which has been shown to be effective in traditional anomaly detection problems. However, it is a non-trivial task in attributed networks as interactions among instances complicate the residual modeling process. Methodologically, we propose a learning framework to characterize the residuals of attribute information and its coherence with network information for anomaly detection. By learning and analyzing the residuals, we detect anomalies whose behaviors are singularly different from the majority. Experiments on real datasets show the effectiveness and generality of the proposed framework.

{{< /ci-details >}}

{{< ci-details summary="Canonical Correlation Analysis: An Overview with Application to Learning Methods (D. Hardoon et al., 2004)">}}

D. Hardoon, S. Szedmák, J. Shawe-Taylor. (2004)  
**Canonical Correlation Analysis: An Overview with Application to Learning Methods**  
Neural Computation  
[Paper Link](https://www.semanticscholar.org/paper/a6b5b20151c752beb74508f813699fa5216dedfa)  
Influential Citation Count (464), SS-ID (a6b5b20151c752beb74508f813699fa5216dedfa)  

**ABSTRACT**  
We present a general method using kernel canonical correlation analysis to learn a semantic representation to web images and their associated text. The semantic space provides a common representation and enables a comparison between the text and images. In the experiments, we look at two approaches of retrieving images based on only their content from a text query. We compare orthogonalization approaches against a standard cross-representation retrieval technique known as the generalized vector space model.

{{< /ci-details >}}

{{< ci-details summary="Toward online node classification on streaming networks (Ling Jian et al., 2017)">}}

Ling Jian, Jundong Li, Huan Liu. (2017)  
**Toward online node classification on streaming networks**  
Data Mining and Knowledge Discovery  
[Paper Link](https://www.semanticscholar.org/paper/a84f089e8a25c75ceff26bca6bbe1040dfecb16a)  
Influential Citation Count (1), SS-ID (a84f089e8a25c75ceff26bca6bbe1040dfecb16a)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Modeling Document Networks with Tree-Averaged Copula Regularization (Yuan He et al., 2017)">}}

Yuan He, Cheng Wang, Changjun Jiang. (2017)  
**Modeling Document Networks with Tree-Averaged Copula Regularization**  
WSDM  
[Paper Link](https://www.semanticscholar.org/paper/ac60fb937b13c6764c7e7297ef209552ff4b5b6f)  
Influential Citation Count (0), SS-ID (ac60fb937b13c6764c7e7297ef209552ff4b5b6f)  

**ABSTRACT**  
Document network is a kind of intriguing dataset which provides both topical (texts) and topological (links) information. Most previous work assumes that documents closely linked with each other share common topics. However, the associations among documents are usually complex, which are not limited to the homophily (i.e., tendency to link to similar others). Actually, the heterophily (i.e., tendency to link to different others) is another pervasive phenomenon in social networks. In this paper, we introduce a new tool, called copula, to separately model the documents and links, so that different copula functions can be applied to capture different correlation patterns. In statistics, a copula is a powerful framework for explicitly modeling the dependence of random variables by separating the marginals and their correlations. Though widely used in Economics, copulas have not been paid enough attention to by researchers in machine learning field. Besides, to further capture the potential associations among the unconnected documents, we apply the tree-averaged copula instead of a single copula function. This improvement makes our model achieve better expressive power, and also more elegant in algebra. We derive efficient EM algorithms to estimate the model parameters, and evaluate the performance of our model on three different datasets. Experimental results show that our approach achieves significant improvements on both topic and link modeling compared with the current state of the art.

{{< /ci-details >}}

{{< ci-details summary="User-level sentiment analysis incorporating social networks (Chenhao Tan et al., 2011)">}}

Chenhao Tan, Lillian Lee, Jie Tang, Long Jiang, M. Zhou, Ping Li. (2011)  
**User-level sentiment analysis incorporating social networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/ad22a4d103510b07f2b9114b371e591c1a09383f)  
Influential Citation Count (30), SS-ID (ad22a4d103510b07f2b9114b371e591c1a09383f)  

**ABSTRACT**  
We show that information about social relationships can be used to improve user-level sentiment analysis. The main motivation behind our approach is that users that are somehow "connected" may be more likely to hold similar opinions; therefore, relationship information can complement what we can extract about a user's viewpoints from their utterances. Employing Twitter as a source for our experimental data, and working within a semi-supervised framework, we propose models that are induced either from the Twitter follower/followee network or from the network in Twitter formed by users referring to each other using "@" mentions. Our transductive learning results reveal that incorporating social-network information can indeed lead to statistically significant sentiment classification improvements over the performance of an approach based on Support Vector Machines having access only to textual features.

{{< /ci-details >}}

{{< ci-details summary="Homophily and Contagion Are Generically Confounded in Observational Social Network Studies (C. Shalizi et al., 2010)">}}

C. Shalizi, Andrew C. Thomas. (2010)  
**Homophily and Contagion Are Generically Confounded in Observational Social Network Studies**  
Sociological methods & research  
[Paper Link](https://www.semanticscholar.org/paper/afcb4870d7751e90cc52e6d5a794bb1530fd33d7)  
Influential Citation Count (46), SS-ID (afcb4870d7751e90cc52e6d5a794bb1530fd33d7)  

**ABSTRACT**  
The authors consider processes on social networks that can potentially involve three factors: homophily, or the formation of social ties due to matching individual traits; social contagion, also known as social influence; and the causal effect of an individual’s covariates on his or her behavior or other measurable responses. The authors show that generically, all of these are confounded with each other. Distinguishing them from one another requires strong assumptions on the parametrization of the social process or on the adequacy of the covariates used (or both). In particular the authors demonstrate, with simple examples, that asymmetries in regression coefficients cannot identify causal effects and that very simple models of imitation (a form of social contagion) can produce substantial correlations between an individual’s enduring traits and his or her choices, even when there is no intrinsic affinity between them. The authors also suggest some possible constructive responses to these results.

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

{{< ci-details summary="Co-regularized Multi-view Spectral Clustering (Abhishek Kumar et al., 2011)">}}

Abhishek Kumar, Piyush Rai, Hal Daumé. (2011)  
**Co-regularized Multi-view Spectral Clustering**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/ba9233dd7c81765b7afb2cc1a6e5e9a075518d8c)  
Influential Citation Count (147), SS-ID (ba9233dd7c81765b7afb2cc1a6e5e9a075518d8c)  

**ABSTRACT**  
In many clustering problems, we have access to multiple views of the data each of which could be individually used for clustering. Exploiting information from multiple views, one can hope to find a clustering that is more accurate than the ones obtained using the individual views. Often these different views admit same underlying clustering of the data, so we can approach this problem by looking for clusterings that are consistent across the views, i.e., corresponding data points in each view should have same cluster membership. We propose a spectral clustering framework that achieves this goal by co-regularizing the clustering hypotheses, and propose two co-regularization schemes to accomplish this. Experimental comparisons with a number of baselines on two synthetic and three real-world datasets establish the efficacy of our proposed approaches.

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

{{< ci-details summary="Evolutionary spectral clustering by incorporating temporal smoothness (Yun Chi et al., 2007)">}}

Yun Chi, Xiaodan Song, Dengyong Zhou, K. Hino, B. Tseng. (2007)  
**Evolutionary spectral clustering by incorporating temporal smoothness**  
KDD '07  
[Paper Link](https://www.semanticscholar.org/paper/ca122cd2ba92e722bc4255c456129d9daaabe535)  
Influential Citation Count (31), SS-ID (ca122cd2ba92e722bc4255c456129d9daaabe535)  

**ABSTRACT**  
Evolutionary clustering is an emerging research area essential to important applications such as clustering dynamic Web and blog contents and clustering data streams. In evolutionary clustering, a good clustering result should fit the current data well, while simultaneously not deviate too dramatically from the recent history. To fulfill this dual purpose, a measure of temporal smoothness is integrated in the overall measure of clustering quality. In this paper, we propose two frameworks that incorporate temporal smoothness in evolutionary spectral clustering. For both frameworks, we start with intuitions gained from the well-known k-means clustering problem, and then propose and solve corresponding cost functions for the evolutionary spectral clustering problems. Our solutions to the evolutionary spectral clustering problems provide more stable and consistent clustering results that are less sensitive to short-term noises while at the same time are adaptive to long-term cluster drifts. Furthermore, we demonstrate that our methods provide the optimal solutions to the relaxed versions of the corresponding evolutionary k-means clustering problems. Performance experiments over a number of real and synthetic data sets illustrate our evolutionary spectral clustering methods provide more robust clustering results that are not sensitive to noise and can adapt to data drifts.

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

{{< ci-details summary="Combining link and content for community detection: a discriminative approach (Tianbao Yang et al., 2009)">}}

Tianbao Yang, Rong Jin, Yun Chi, Shenghuo Zhu. (2009)  
**Combining link and content for community detection: a discriminative approach**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/d15aef555481ed5cd59474e9e2efbc1ee37d5e94)  
Influential Citation Count (25), SS-ID (d15aef555481ed5cd59474e9e2efbc1ee37d5e94)  

**ABSTRACT**  
In this paper, we consider the problem of combining link and content analysis for community detection from networked data, such as paper citation networks and Word Wide Web. Most existing approaches combine link and content information by a generative model that generates both links and contents via a shared set of community memberships. These generative models have some shortcomings in that they failed to consider additional factors that could affect the community memberships and isolate the contents that are irrelevant to community memberships. To explicitly address these shortcomings, we propose a discriminative model for combining the link and content analysis for community detection. First, we propose a conditional model for link analysis and in the model, we introduce hidden variables to explicitly model the popularity of nodes. Second, to alleviate the impact of irrelevant content attributes, we develop a discriminative model for content analysis. These two models are unified seamlessly via the community memberships. We present efficient algorithms to solve the related optimization problems based on bound optimization and alternating projection. Extensive experiments with benchmark data sets show that the proposed framework significantly outperforms the state-of-the-art approaches for combining link and content analysis for community detection.

{{< /ci-details >}}

{{< ci-details summary="Unsupervised Feature Selection in Signed Social Networks (Kewei Cheng et al., 2017)">}}

Kewei Cheng, Jundong Li, Huan Liu. (2017)  
**Unsupervised Feature Selection in Signed Social Networks**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/d1f4460091422514b590bea1247f0ca197133b76)  
Influential Citation Count (1), SS-ID (d1f4460091422514b590bea1247f0ca197133b76)  

**ABSTRACT**  
The rapid growth of social media services brings a large amount of high-dimensional social media data at an unprecedented rate. Feature selection is powerful to prepare high-dimensional data by finding a subset of relevant features. A vast majority of existing feature selection algorithms for social media data exclusively focus on positive interactions among linked instances such as friendships and user following relations. However, in many real-world social networks, instances may also be negatively interconnected. Recent work shows that negative links have an added value over positive links in advancing many learning tasks. In this paper, we study a novel problem of unsupervised feature selection in signed social networks and propose a novel framework SignedFS. In particular, we provide a principled way to model positive and negative links for user latent representation learning. Then we embed the user latent representations into feature selection when label information is not available. Also, we revisit the principle of homophily and balance theory in signed social networks and incorporate the signed graph regularization into the feature selection framework to capture the first-order and the second-order proximity among users in signed social networks. Experiments on two real-world signed social networks demonstrate the effectiveness of our proposed framework. Further experiments are conducted to understand the impacts of different components of SignedFS.

{{< /ci-details >}}

{{< ci-details summary="Attributed graph models: modeling network structure with correlated attributes (Joseph J. Pfeiffer et al., 2014)">}}

Joseph J. Pfeiffer, Sebastián Moreno, T. L. Fond, Jennifer Neville, B. Gallagher. (2014)  
**Attributed graph models: modeling network structure with correlated attributes**  
WWW  
[Paper Link](https://www.semanticscholar.org/paper/d96bd17bfda35c9f38f6562eaacd4d54a29434ce)  
Influential Citation Count (7), SS-ID (d96bd17bfda35c9f38f6562eaacd4d54a29434ce)  

**ABSTRACT**  
Online social networks have become ubiquitous to today's society and the study of data from these networks has improved our understanding of the processes by which relationships form. Research in statistical relational learning focuses on methods to exploit correlations among the attributes of linked nodes to predict user characteristics with greater accuracy. Concurrently, research on generative graph models has primarily focused on modeling network structure without attributes, producing several models that are able to replicate structural characteristics of networks such as power law degree distributions or community structure. However, there has been little work on how to generate networks with real-world structural properties and correlated attributes. In this work, we present the Attributed Graph Model (AGM) framework to jointly model network structure and vertex attributes. Our framework learns the attribute correlations in the observed network and exploits a generative graph model, such as the Kronecker Product Graph Model (KPGM) and Chung Lu Graph Model (CL), to compute structural edge probabilities. AGM then combines the attribute correlations with the structural probabilities to sample networks conditioned on attribute values, while keeping the expected edge probabilities and degrees of the input graph model. We outline an efficient method for estimating the parameters of AGM, as well as a sampling method based on Accept-Reject sampling to generate edges with correlated attributes. We demonstrate the efficiency and accuracy of our AGM framework on two large real-world networks, showing that AGM scales to networks with hundreds of thousands of vertices, as well as having high attribute correlation.

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

{{< ci-details summary="Power-Law Distribution of the World Wide Web (Lada A. Adamic et al., 2000)">}}

Lada A. Adamic, B. Huberman. (2000)  
**Power-Law Distribution of the World Wide Web**  
Science  
[Paper Link](https://www.semanticscholar.org/paper/e50b64b496f3b5bc6eaf50a7f5d53fdae16bb946)  
Influential Citation Count (33), SS-ID (e50b64b496f3b5bc6eaf50a7f5d53fdae16bb946)  

**ABSTRACT**  
Barabasi and Albert ([1][1]) propose an improved version of the Erdos-Renyi (ER) theory of random networks to account for the scaling properties of a number of systems, including the link structure of the World Wide Web (WWW). The theory they present, however, is inconsistent with empirically

{{< /ci-details >}}

{{< ci-details summary="Unsupervised feature selection for linked social media data (Jiliang Tang et al., 2012)">}}

Jiliang Tang, Huan Liu. (2012)  
**Unsupervised feature selection for linked social media data**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/ec8c47ef5797976594c7b784dcad6776743ef014)  
Influential Citation Count (16), SS-ID (ec8c47ef5797976594c7b784dcad6776743ef014)  

**ABSTRACT**  
The prevalent use of social media produces mountains of unlabeled, high-dimensional data. Feature selection has been shown effective in dealing with high-dimensional data for efficient data mining. Feature selection for unlabeled data remains a challenging task due to the absence of label information by which the feature relevance can be assessed. The unique characteristics of social media data further complicate the already challenging problem of unsupervised feature selection, (e.g., part of social media data is linked, which makes invalid the independent and identically distributed assumption), bringing about new challenges to traditional unsupervised feature selection algorithms. In this paper, we study the differences between social media data and traditional attribute-value data, investigate if the relations revealed in linked data can be used to help select relevant features, and propose a novel unsupervised feature selection framework, LUFS, for linked social media data. We perform experiments with real-world social media datasets to evaluate the effectiveness of the proposed framework and probe the working of its key components.

{{< /ci-details >}}

{{< ci-details summary="A tutorial on spectral clustering (U. V. Luxburg, 2007)">}}

U. V. Luxburg. (2007)  
**A tutorial on spectral clustering**  
Stat. Comput.  
[Paper Link](https://www.semanticscholar.org/paper/eda90bd43f4256986688e525b45b833a3addab97)  
Influential Citation Count (818), SS-ID (eda90bd43f4256986688e525b45b833a3addab97)  

**ABSTRACT**  


{{< /ci-details >}}

{{< ci-details summary="Incremental Spectral Clustering With Application to Monitoring of Evolving Blog Communities (Huazhong Ning et al., 2007)">}}

Huazhong Ning, W. Xu, Yun Chi, Yihong Gong, Thomas S. Huang. (2007)  
**Incremental Spectral Clustering With Application to Monitoring of Evolving Blog Communities**  
SDM  
[Paper Link](https://www.semanticscholar.org/paper/f03f6675d82536aa00fc0040dbe37723217a2dd8)  
Influential Citation Count (7), SS-ID (f03f6675d82536aa00fc0040dbe37723217a2dd8)  

**ABSTRACT**  
In recent years, spectral clustering method has gained attentions because of its superior performance compared to other traditional clustering algorithms such as K-means algorithm. The existing spectral clustering algorithms are all off-line algorithms, i.e., they can not incrementally update the clustering result given a small change of the data set. However, the capability of incrementally updating is essential to some applications such as real time monitoring of the evolving communities of websphere or blogsphere. Unlike traditional stream data, these applications require incremental algorithms to handle not only insertion/deletion of data points but also similarity changes between existing items. This paper extends the standard spectral clustering to such evolving data by introducing the incidence vector/matrix to represent two kinds of dynamics in the same framework and by incrementally updating the eigenvalue system. Our incremental algorithm, initialized by a standard spectral clustering, continuously and efficiently updates the eigenvalue system and generates instant cluster labels, as the data set is evolving. The algorithm is applied to a blog data set. Compared with recomputation of the solution by standard spectral clustering, it achieves similar accuracy but with much lower computational cost. Close inspection into the blog content shows that the incremental approach can discover not only the stable blog communities but also the evolution of the individual multi-topic blogs.

{{< /ci-details >}}

{{< ci-details summary="Heterogeneous Network Embedding via Deep Architectures (Shiyu Chang et al., 2015)">}}

Shiyu Chang, Wei Han, Jiliang Tang, Guo-Jun Qi, C. Aggarwal, Thomas S. Huang. (2015)  
**Heterogeneous Network Embedding via Deep Architectures**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  
Influential Citation Count (37), SS-ID (f7172f95a3c0aa4fddfaadbce9908ce20cbf50ef)  

**ABSTRACT**  
Data embedding is used in many machine learning applications to create low-dimensional feature representations, which preserves the structure of data points in their original space. In this paper, we examine the scenario of a heterogeneous network with nodes and content of various types. Such networks are notoriously difficult to mine because of the bewildering combination of heterogeneous contents and structures. The creation of a multidimensional embedding of such data opens the door to the use of a wide variety of off-the-shelf mining techniques for multidimensional data. Despite the importance of this problem, limited efforts have been made on embedding a network of scalable, dynamic and heterogeneous data. In such cases, both the content and linkage structure provide important cues for creating a unified feature representation of the underlying network. In this paper, we design a deep embedding algorithm for networked data. A highly nonlinear multi-layered embedding function is used to capture the complex interactions between the heterogeneous data in a network. Our goal is to create a multi-resolution deep embedding function, that reflects both the local and global network structures, and makes the resulting embedding useful for a variety of data mining tasks. In particular, we demonstrate that the rich content and linkage information in a heterogeneous network can be captured by such an approach, so that similarities among cross-modal data can be measured directly in a common embedding space. Once this goal has been achieved, a wide variety of data mining problems can be solved by applying off-the-shelf algorithms designed for handling vector representations. Our experiments on real-world network datasets show the effectiveness and scalability of the proposed algorithm as compared to the state-of-the-art embedding methods.

{{< /ci-details >}}

{{< ci-details summary="Toward Time-Evolving Feature Selection on Dynamic Networks (Jundong Li et al., 2016)">}}

Jundong Li, Xia Hu, Ling Jian, Huan Liu. (2016)  
**Toward Time-Evolving Feature Selection on Dynamic Networks**  
2016 IEEE 16th International Conference on Data Mining (ICDM)  
[Paper Link](https://www.semanticscholar.org/paper/f831be455cba1db826715e3caf7957fae1e5169d)  
Influential Citation Count (1), SS-ID (f831be455cba1db826715e3caf7957fae1e5169d)  

**ABSTRACT**  
Recent years have witnessed the prevalence of networked data in various domains. Among them, a large number of networks are not only topologically structured but also have a rich set of features on nodes. These node features are usually of high dimensionality with noisy, irrelevant and redundant information, which may impede the performance of other learning tasks. Feature selection is useful to alleviate these critical issues. Nonetheless, a vast majority of existing feature selection algorithms are predominantly designed in a static setting. In reality, real-world networks are naturally dynamic, characterized by both topology and content changes. It is desirable to capture these changes to find relevant features tightly hinged with network structure continuously, which is of fundamental importance for many applications such as disaster relief and viral marketing. In this paper, we study a novel problem of time-evolving feature selection for dynamic networks in an unsupervised scenario. Specifically, we propose a TeFS framework by leveraging the temporal evolution property of dynamic networks to update the feature selection results incrementally. Experimental results show the superiority of TeFS over the state-of-the-art batch-mode unsupervised feature selection algorithms.

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

{{< ci-details summary="Network Representation Learning with Rich Text Information (Cheng Yang et al., 2015)">}}

Cheng Yang, Zhiyuan Liu, Deli Zhao, Maosong Sun, Edward Y. Chang. (2015)  
**Network Representation Learning with Rich Text Information**  
IJCAI  
[Paper Link](https://www.semanticscholar.org/paper/fce14c6aa64e888456256ac6796744683165a0ff)  
Influential Citation Count (172), SS-ID (fce14c6aa64e888456256ac6796744683165a0ff)  

**ABSTRACT**  
Representation learning has shown its effectiveness in many tasks such as image classification and text mining. Network representation learning aims at learning distributed vector representation for each vertex in a network, which is also increasingly recognized as an important aspect for network analysis. Most network representation learning methods investigate network structures for learning. In reality, network vertices contain rich information (such as text), which cannot be well applied with algorithmic frameworks of typical representation learning methods. By proving that DeepWalk, a state-of-the-art network representation method, is actually equivalent to matrix factorization (MF), we propose text-associated DeepWalk (TADW). TADW incorporates text features of vertices into network representation learning under the framework of matrix factorization. We evaluate our method and various baseline methods by applying them to the task of multi-class classification of vertices. The experimental results show that, our method outperforms other baselines on all three datasets, especially when networks are noisy and training ratio is small. The source code of this paper can be obtained from https://github.com/albertyang33/TADW.

{{< /ci-details >}}

{{< ci-details summary="DeepWalk: online learning of social representations (Bryan Perozzi et al., 2014)">}}

Bryan Perozzi, Rami Al-Rfou, S. Skiena. (2014)  
**DeepWalk: online learning of social representations**  
KDD  
[Paper Link](https://www.semanticscholar.org/paper/fff114cbba4f3ba900f33da574283e3de7f26c83)  
Influential Citation Count (1393), SS-ID (fff114cbba4f3ba900f33da574283e3de7f26c83)  

**ABSTRACT**  
We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs. DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk's latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk's representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk's representations are able to outperform all baseline methods while using 60% less training data. DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

{{< /ci-details >}}

