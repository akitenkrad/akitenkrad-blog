---
draft: true 
title: "Dilated Recurrent Neural Networks"
date: 2022-09-14
author: "akitenkrad"
description: ""
tags: ["At:Round-2", "Published:2017", "RNN"]
menu:
  sidebar:
    name: "Dilated Recurrent Neural Networks"
    identifier: 20220914
    parent: 202209
    weight: 10
math: true
---

- [x] Round-1: Overview
- [x] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

{{< citation >}}
Chang, S., Zhang, Y., Han, W., Yu, M., Guo, X., Tan, W., Cui, X., Witbrock, M., Hasegawa-Johnson, M., & Huang, T. S. (2017).  
Dilated Recurrent Neural Networks.  
https://doi.org/10.48550/arxiv.1710.02224
{{< /citation >}}

## Abstract
> Learning with recurrent neural networks (RNNs) on long sequences is a notoriously difficult task. There are three major challenges: 1) complex dependencies, 2) vanishing and exploding gradients, and 3) efficient parallelization. In this paper, we introduce a simple yet effective RNN connection structure, the DilatedRNN, which simultaneously tackles all of these challenges. The proposed architecture is characterized by multi-resolution dilated recurrent skip connections and can be combined flexibly with diverse RNN cells. Moreover, the DilatedRNN reduces the number of parameters needed and enhances training efficiency significantly, while matching state-of-the-art performance (even with standard RNN cells) in tasks involving very long-term dependencies. To provide a theory-based quantification of the architecture's advantages, we introduce a memory capacity measure, the mean recurrent length, which is more suitable for RNNs with long skip connections than existing measures. We rigorously prove the advantages of the DilatedRNN over other recurrent neural architectures. The code for our method is publicly available at https://github.com/code-terminator/DilatedRNN

## Background & Wat's New

- RNN系列のモデルにおける3つの課題
  - 短期的・中期的な情報を維持しながら長期的な情報を伝播させることが困難
  - BPTT (Back Propagation Through Time)の間に勾配が消失または爆発する
  - シーケンシャルな処理であるため，処理に時間がかかる
- Main Contributions
  - RNNに対してdilated recurrent skip connectionを導入
    - 勾配の問題を緩和し，長期的な情報の保持を可能にする
    - パラメータが少なくなるため，学習速度の向上にも寄与する
  - dilated recurrent layersを複数重ねることでDILATED RNNを提案
    - 時系列の依存関係を複数のスケールで学習することを可能にする
  - 新しい指標としてmean recurrent lengthを提案

{{< figure src="Dilated RNN.png" width="100%" caption="Dependencies" >}}

## Dataset

## Model Description

時刻 $t$ におけるレイヤ $l$ のセルを $c\_{l, t}$ としたとき，**delated skip connection** は以下のように表される．

$$
\begin{array}{ll}
  c\_{l,t} = f\left( x\_{l,t}, c\_{l, t-s\_l}\right)  \tag{1} \\\\ \\\\
  \text{where} \hspace{10pt} \left \lbrace \begin{array}{ll}
    s\_l \mapsto \text{skip length/dilation of layer }l \\\\
    x\_{l,t} \mapsto \text{the input to layer }l \\\\
    f(\cdot) \mapsto \text{any RNN cell e.g. Vanilla RNN, LSTM, GRU etc.}
  \end{array} \right .
\end{array}
$$

{{< figure src="figure-1.png" width="60%" caption="" >}}

## Results

### Settings

## References


{{< ci-details summary="Dilated Residual Networks (F. Yu et al., 2017)">}}
F. Yu, V. Koltun, T. Funkhouser. (2017)  
**Dilated Residual Networks**  
2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)  
[Paper Link](https://www.semanticscholar.org/paper/6f34b9a4a0e2ee90e86ed720dc26cc6ba9da8df0)  
Influential Citation Count (130), SS-ID (6f34b9a4a0e2ee90e86ed720dc26cc6ba9da8df0)  
**ABSTRACT**  
Convolutional networks for image classification progressively reduce resolution until the image is represented by tiny feature maps in which the spatial structure of the scene is no longer discernible. Such loss of spatial acuity can limit image classification accuracy and complicate the transfer of the model to downstream applications that require detailed scene understanding. These problems can be alleviated by dilation, which increases the resolution of output feature maps without reducing the receptive field of individual neurons. We show that dilated residual networks (DRNs) outperform their non-dilated counterparts in image classification without increasing the models depth or complexity. We then study gridding artifacts introduced by dilation, develop an approach to removing these artifacts (degridding), and show that this further increases the performance of DRNs. In addition, we show that the accuracy advantage of DRNs is further magnified in downstream applications such as object localization and semantic segmentation.
{{< /ci-details >}}
{{< ci-details summary="Learning to Skim Text (Adams Wei Yu et al., 2017)">}}
Adams Wei Yu, Hongrae Lee, Quoc V. Le. (2017)  
**Learning to Skim Text**  
ACL  
[Paper Link](https://www.semanticscholar.org/paper/c25a67ad7e8629a9d12b9e2fc356cd73af99a060)  
Influential Citation Count (17), SS-ID (c25a67ad7e8629a9d12b9e2fc356cd73af99a060)  
**ABSTRACT**  
Recurrent Neural Networks are showing much promise in many sub-areas of natural language processing, ranging from document classification to machine translation to automatic question answering. Despite their promise, many recurrent models have to read the whole text word by word, making it slow to handle long documents. For example, it is difficult to use a recurrent network to read a book and answer questions about it. In this paper, we present an approach of reading text while skipping irrelevant information if needed. The underlying model is a recurrent network that learns how far to jump after reading a few words of the input text. We employ a standard policy gradient method to train the model to make discrete jumping decisions. In our benchmarks on four different tasks, including number prediction, sentiment analysis, news article classification and automatic Q&A, our proposed model, a modified LSTM with jumping, is up to 6 times faster than the standard sequential LSTM, while maintaining the same or even better accuracy.
{{< /ci-details >}}
{{< ci-details summary="FeUdal Networks for Hierarchical Reinforcement Learning (A. Vezhnevets et al., 2017)">}}
A. Vezhnevets, Simon Osindero, T. Schaul, N. Heess, Max Jaderberg, David Silver, K. Kavukcuoglu. (2017)  
**FeUdal Networks for Hierarchical Reinforcement Learning**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/049c6e5736313374c6e594c34b9be89a3a09dced)  
Influential Citation Count (50), SS-ID (049c6e5736313374c6e594c34b9be89a3a09dced)  
**ABSTRACT**  
We introduce FeUdal Networks (FuNs): a novel architecture for hierarchical reinforcement learning. Our approach is inspired by the feudal reinforcement learning proposal of Dayan and Hinton, and gains power and efficacy by decoupling end-to-end learning across multiple levels -- allowing it to utilise different resolutions of time. Our framework employs a Manager module and a Worker module. The Manager operates at a lower temporal resolution and sets abstract goals which are conveyed to and enacted by the Worker. The Worker generates primitive actions at every tick of the environment. The decoupled structure of FuN conveys several benefits -- in addition to facilitating very long timescale credit assignment it also encourages the emergence of sub-policies associated with different goals set by the Manager. These properties allow FuN to dramatically outperform a strong baseline agent on tasks that involve long-term credit assignment or memorisation. We demonstrate the performance of our proposed system on a range of tasks from the ATARI suite and also from a 3D DeepMind Lab environment.
{{< /ci-details >}}
{{< ci-details summary="Hierarchical Multiscale Recurrent Neural Networks (Junyoung Chung et al., 2016)">}}
Junyoung Chung, Sungjin Ahn, Yoshua Bengio. (2016)  
**Hierarchical Multiscale Recurrent Neural Networks**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/65eee67dee969fdf8b44c87c560d66ad4d78e233)  
Influential Citation Count (72), SS-ID (65eee67dee969fdf8b44c87c560d66ad4d78e233)  
**ABSTRACT**  
Learning both hierarchical and temporal representation has been among the long-standing challenges of recurrent neural networks. Multiscale recurrent neural networks have been considered as a promising approach to resolve this issue, yet there has been a lack of empirical evidence showing that this type of models can actually capture the temporal dependencies by discovering the latent hierarchical structure of the sequence. In this paper, we propose a novel multiscale approach, called the hierarchical multiscale recurrent neural networks, which can capture the latent hierarchical structure in the sequence by encoding the temporal dependencies with different timescales using a novel update mechanism. We show some evidence that our proposed multiscale architecture can discover underlying hierarchical structure in the sequences without using explicit boundary information. We evaluate our proposed model on character-level language modelling and handwriting sequence modelling.
{{< /ci-details >}}
{{< ci-details summary="Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations (David Krueger et al., 2016)">}}
David Krueger, Tegan Maharaj, J'anos Kram'ar, M. Pezeshki, Nicolas Ballas, Nan Rosemary Ke, Anirudh Goyal, Yoshua Bengio, H. Larochelle, Aaron C. Courville, Chris Pal. (2016)  
**Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/9f0687bcd0a7d7fc91b8c5d36c003a38b8853105)  
Influential Citation Count (26), SS-ID (9f0687bcd0a7d7fc91b8c5d36c003a38b8853105)  
**ABSTRACT**  
We propose zoneout, a novel method for regularizing RNNs. At each timestep, zoneout stochastically forces some hidden units to maintain their previous values. Like dropout, zoneout uses random noise to train a pseudo-ensemble, improving generalization. But by preserving instead of dropping hidden units, gradient information and state information are more readily propagated through time, as in feedforward stochastic depth networks. We perform an empirical investigation of various RNN regularizers, and find that zoneout gives significant performance improvements across tasks. We achieve competitive results with relatively simple models in character- and word-level language modelling on the Penn Treebank and Text8 datasets, and combining with recurrent batch normalization yields state-of-the-art results on permuted sequential MNIST.
{{< /ci-details >}}
{{< ci-details summary="Recurrent Batch Normalization (Tim Cooijmans et al., 2016)">}}
Tim Cooijmans, Nicolas Ballas, César Laurent, Aaron C. Courville. (2016)  
**Recurrent Batch Normalization**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/952454718139dba3aafc6b3b67c4f514ac3964af)  
Influential Citation Count (51), SS-ID (952454718139dba3aafc6b3b67c4f514ac3964af)  
**ABSTRACT**  
We propose a reparameterization of LSTM that brings the benefits of batch normalization to recurrent neural networks. Whereas previous works only apply batch normalization to the input-to-hidden transformation of RNNs, we demonstrate that it is both possible and beneficial to batch-normalize the hidden-to-hidden transition, thereby reducing internal covariate shift between time steps. We evaluate our proposal on various sequential problems such as sequence classification, language modeling and question answering. Our empirical results show that our batch-normalized LSTM consistently leads to faster convergence and improved generalization.
{{< /ci-details >}}
{{< ci-details summary="Full-Capacity Unitary Recurrent Neural Networks (Scott Wisdom et al., 2016)">}}
Scott Wisdom, Thomas Powers, J. Hershey, Jonathan Le Roux, L. Atlas. (2016)  
**Full-Capacity Unitary Recurrent Neural Networks**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/79c78c98ea317ba8cdf25a9783ef4b8a7552db75)  
Influential Citation Count (34), SS-ID (79c78c98ea317ba8cdf25a9783ef4b8a7552db75)  
**ABSTRACT**  
Recurrent neural networks are powerful models for processing sequential data, but they are generally plagued by vanishing and exploding gradient problems. Unitary recurrent neural networks (uRNNs), which use unitary recurrence matrices, have recently been proposed as a means to avoid these issues. However, in previous experiments, the recurrence matrices were restricted to be a product of parameterized unitary matrices, and an open question remains: when does such a parameterization fail to represent all unitary matrices, and how does this restricted representational capacity limit what can be learned? To address this question, we propose full-capacity uRNNs that optimize their recurrence matrix over all unitary matrices, leading to significantly improved performance over uRNNs that use a restricted-capacity recurrence matrix. Our contribution consists of two main components. First, we provide a theoretical argument to determine if a unitary parameterization has restricted capacity. Using this argument, we show that a recently proposed unitary parameterization has restricted capacity for hidden state dimension greater than 7. Second, we show how a complete, full-capacity unitary recurrence matrix can be optimized over the differentiable manifold of unitary matrices. The resulting multiplicative gradient step is very simple and does not require gradient clipping or learning rate adaptation. We confirm the utility of our claims by empirically evaluating our new full-capacity uRNNs on both synthetic and natural data, achieving superior performance compared to both LSTMs and the original restricted-capacity uRNNs.
{{< /ci-details >}}
{{< ci-details summary="Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences (Daniel Neil et al., 2016)">}}
Daniel Neil, Michael Pfeiffer, Shih-Chii Liu. (2016)  
**Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/a4b6b32c21609b9b093e0aacc5c0d82e70a9be52)  
Influential Citation Count (50), SS-ID (a4b6b32c21609b9b093e0aacc5c0d82e70a9be52)  
**ABSTRACT**  
Recurrent Neural Networks (RNNs) have become the state-of-the-art choice for extracting patterns from temporal sequences. Current RNN models are ill suited to process irregularly sampled data triggered by events generated in continuous time by sensors or other neurons. Such data can occur, for example, when the input comes from novel event-driven artificial sensors which generate sparse, asynchronous streams of events or from multiple conventional sensors with different update intervals. In this work, we introduce the Phased LSTM model, which extends the LSTM unit by adding a new time gate. This gate is controlled by a parametrized oscillation with a frequency range which require updates of the memory cell only during a small percentage of the cycle. Even with the sparse updates imposed by the oscillation, the Phased LSTM network achieves faster convergence than regular LSTMs on tasks which require learning of long sequences. The model naturally integrates inputs from sensors of arbitrary sampling rates, thereby opening new areas of investigation for processing asynchronous sensory events that carry timing information. It also greatly improves the performance of LSTMs in standard RNN applications, and does so with an order-of-magnitude fewer computes.
{{< /ci-details >}}
{{< ci-details summary="SUPERSEDED - CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (C. Veaux et al., 2016)">}}
C. Veaux, J. Yamagishi, Kirsten MacDonald. (2016)  
**SUPERSEDED - CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit**  
  
[Paper Link](https://www.semanticscholar.org/paper/d4903c15a7aba8e2c2386b2fe95edf0905144d6a)  
Influential Citation Count (75), SS-ID (d4903c15a7aba8e2c2386b2fe95edf0905144d6a)  
{{< /ci-details >}}
{{< ci-details summary="WaveNet: A Generative Model for Raw Audio (Aäron van den Oord et al., 2016)">}}
Aäron van den Oord, S. Dieleman, H. Zen, K. Simonyan, Oriol Vinyals, A. Graves, Nal Kalchbrenner, A. Senior, K. Kavukcuoglu. (2016)  
**WaveNet: A Generative Model for Raw Audio**  
SSW  
[Paper Link](https://www.semanticscholar.org/paper/df0402517a7338ae28bc54acaac400de6b456a46)  
Influential Citation Count (786), SS-ID (df0402517a7338ae28bc54acaac400de6b456a46)  
**ABSTRACT**  
This paper introduces WaveNet, a deep neural network for generating raw audio waveforms. The model is fully probabilistic and autoregressive, with the predictive distribution for each audio sample conditioned on all previous ones; nonetheless we show that it can be efficiently trained on data with tens of thousands of samples per second of audio. When applied to text-to-speech, it yields state-of-the-art performance, with human listeners rating it as significantly more natural sounding than the best parametric and concatenative systems for both English and Mandarin. A single WaveNet can capture the characteristics of many different speakers with equal fidelity, and can switch between them by conditioning on the speaker identity. When trained to model music, we find that it generates novel and often highly realistic musical fragments. We also show that it can be employed as a discriminative model, returning promising results for phoneme recognition.
{{< /ci-details >}}
{{< ci-details summary="Recurrent Dropout without Memory Loss (Stanislau Semeniuta et al., 2016)">}}
Stanislau Semeniuta, Aliaksei Severyn, E. Barth. (2016)  
**Recurrent Dropout without Memory Loss**  
COLING  
[Paper Link](https://www.semanticscholar.org/paper/cf76789618f5db929393c1187514ce6c3502c3cd)  
Influential Citation Count (26), SS-ID (cf76789618f5db929393c1187514ce6c3502c3cd)  
**ABSTRACT**  
This paper presents a novel approach to recurrent neural network (RNN) regularization. Differently from the widely adopted dropout method, which is applied to forward connections of feedforward architectures or RNNs, we propose to drop neurons directly in recurrent connections in a way that does not cause loss of long-term memory. Our approach is as easy to implement and apply as the regular feed-forward dropout and we demonstrate its effectiveness for the most effective modern recurrent network – Long Short-Term Memory network. Our experiments on three NLP benchmarks show consistent improvements even when combined with conventional feed-forward dropout.
{{< /ci-details >}}
{{< ci-details summary="TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems (Martín Abadi et al., 2016)">}}
Martín Abadi, Ashish Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. Corrado, Andy Davis, J. Dean, Matthieu Devin, S. Ghemawat, Ian J. Goodfellow, A. Harp, Geoffrey Irving, M. Isard, Yangqing Jia, R. Józefowicz, Lukasz Kaiser, M. Kudlur, J. Levenberg, Dandelion Mané, R. Monga, Sherry Moore, D. Murray, C. Olah, M. Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, P. Tucker, Vincent Vanhoucke, Vijay Vasudevan, F. Viégas, Oriol Vinyals, P. Warden, M. Wattenberg, M. Wicke, Yuan Yu, Xiaoqiang Zheng. (2016)  
**TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/9c9d7247f8c51ec5a02b0d911d1d7b9e8160495d)  
Influential Citation Count (978), SS-ID (9c9d7247f8c51ec5a02b0d911d1d7b9e8160495d)  
**ABSTRACT**  
TensorFlow is an interface for expressing machine learning algorithms, and an implementation for executing such algorithms. A computation expressed using TensorFlow can be executed with little or no change on a wide variety of heterogeneous systems, ranging from mobile devices such as phones and tablets up to large-scale distributed systems of hundreds of machines and thousands of computational devices such as GPU cards. The system is flexible and can be used to express a wide variety of algorithms, including training and inference algorithms for deep neural network models, and it has been used for conducting research and for deploying machine learning systems into production across more than a dozen areas of computer science and other fields, including speech recognition, computer vision, robotics, information retrieval, natural language processing, geographic information extraction, and computational drug discovery. This paper describes the TensorFlow interface and an implementation of that interface that we have built at Google. The TensorFlow API and a reference implementation were released as an open-source package under the Apache 2.0 license in November, 2015 and are available at www.tensorflow.org.
{{< /ci-details >}}
{{< ci-details summary="Architectural Complexity Measures of Recurrent Neural Networks (Saizheng Zhang et al., 2016)">}}
Saizheng Zhang, Yuhuai Wu, Tong Che, Zhouhan Lin, R. Memisevic, R. Salakhutdinov, Yoshua Bengio. (2016)  
**Architectural Complexity Measures of Recurrent Neural Networks**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/f6fda11d2b31ad66dd008a65f7e708aa64a27703)  
Influential Citation Count (18), SS-ID (f6fda11d2b31ad66dd008a65f7e708aa64a27703)  
**ABSTRACT**  
In this paper, we systematically analyze the connecting architectures of recurrent neural networks (RNNs). Our main contribution is twofold: first, we present a rigorous graph-theoretic framework describing the connecting architectures of RNNs in general. Second, we propose three architecture complexity measures of RNNs: (a) the recurrent depth, which captures the RNN's over-time nonlinear complexity, (b) the feedforward depth, which captures the local input-output nonlinearity (similar to the "depth" in feedforward neural networks (FNNs)), and (c) the recurrent skip coefficient which captures how rapidly the information propagates over time. We rigorously prove each measure's existence and computability. Our experimental results show that RNNs might benefit from larger recurrent depth and feedforward depth. We further demonstrate that increasing recurrent skip coefficient offers performance boosts on long term dependency problems.
{{< /ci-details >}}
{{< ci-details summary="Multi-Scale Context Aggregation by Dilated Convolutions (F. Yu et al., 2015)">}}
F. Yu, V. Koltun. (2015)  
**Multi-Scale Context Aggregation by Dilated Convolutions**  
ICLR  
[Paper Link](https://www.semanticscholar.org/paper/7f5fc84819c0cf94b771fe15141f65b123f7b8ec)  
Influential Citation Count (393), SS-ID (7f5fc84819c0cf94b771fe15141f65b123f7b8ec)  
**ABSTRACT**  
State-of-the-art models for semantic segmentation are based on adaptations of convolutional networks that had originally been designed for image classification. However, dense prediction and image classification are structurally different. In this work, we develop a new convolutional network module that is specifically designed for dense prediction. The presented module uses dilated convolutions to systematically aggregate multi-scale contextual information without losing resolution. The architecture is based on the fact that dilated convolutions support exponential expansion of the receptive field without loss of resolution or coverage. We show that the presented context module increases the accuracy of state-of-the-art semantic segmentation systems. In addition, we examine the adaptation of image classification networks to dense prediction and show that simplifying the adapted network can increase accuracy.
{{< /ci-details >}}
{{< ci-details summary="Unitary Evolution Recurrent Neural Networks (Martín Arjovsky et al., 2015)">}}
Martín Arjovsky, Amar Shah, Yoshua Bengio. (2015)  
**Unitary Evolution Recurrent Neural Networks**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/e9c771197a6564762754e48c1daafb066f449f2e)  
Influential Citation Count (98), SS-ID (e9c771197a6564762754e48c1daafb066f449f2e)  
**ABSTRACT**  
Recurrent neural networks (RNNs) are notoriously difficult to train. When the eigenvalues of the hidden to hidden weight matrix deviate from absolute value 1, optimization becomes difficult due to the well studied issue of vanishing and exploding gradients, especially when trying to learn long-term dependencies. To circumvent this problem, we propose a new architecture that learns a unitary weight matrix, with eigenvalues of absolute value exactly 1. The challenge we address is that of parametrizing unitary matrices in a way that does not require expensive computations (such as eigendecomposition) after each weight update. We construct an expressive unitary weight matrix by composing several structured matrices that act as building blocks with parameters to be learned. Optimization with this parameterization becomes feasible only when considering hidden states in the complex domain. We demonstrate the potential of this architecture by achieving state of the art results in several hard tasks involving very long-term dependencies.
{{< /ci-details >}}
{{< ci-details summary="A Simple Way to Initialize Recurrent Networks of Rectified Linear Units (Quoc V. Le et al., 2015)">}}
Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton. (2015)  
**A Simple Way to Initialize Recurrent Networks of Rectified Linear Units**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/d46b81707786d18499f911b4ab72bb10c65406ba)  
Influential Citation Count (80), SS-ID (d46b81707786d18499f911b4ab72bb10c65406ba)  
**ABSTRACT**  
Learning long term dependencies in recurrent networks is difficult due to vanishing and exploding gradients. To overcome this difficulty, researchers have developed sophisticated optimization techniques and network architectures. In this paper, we propose a simpler solution that use recurrent neural networks composed of rectified linear units. Key to our solution is the use of the identity matrix or its scaled version to initialize the recurrent weight matrix. We find that our solution is comparable to LSTM on our four benchmarks: two toy problems involving long-range temporal structures, a large language modeling problem and a benchmark speech recognition problem.
{{< /ci-details >}}
{{< ci-details summary="Learning the speech front-end with raw waveform CLDNNs (Tara N. Sainath et al., 2015)">}}
Tara N. Sainath, Ron J. Weiss, A. Senior, K. Wilson, Oriol Vinyals. (2015)  
**Learning the speech front-end with raw waveform CLDNNs**  
INTERSPEECH  
[Paper Link](https://www.semanticscholar.org/paper/fd5474f21495989777cbff507ecf1b37b7091475)  
Influential Citation Count (30), SS-ID (fd5474f21495989777cbff507ecf1b37b7091475)  
**ABSTRACT**  
Learning an acoustic model directly from the raw waveform has been an active area of research. However, waveformbased models have not yet matched the performance of logmel trained neural networks. We will show that raw waveform features match the performance of log-mel filterbank energies when used with a state-of-the-art CLDNN acoustic model trained on over 2,000 hours of speech. Specifically, we will show the benefit of the CLDNN, namely the time convolution layer in reducing temporal variations, the frequency convolution layer for preserving locality and reducing frequency variations, as well as the LSTM layers for temporal modeling. In addition, by stacking raw waveform features with log-mel features, we achieve a 3% relative reduction in word error rate.
{{< /ci-details >}}
{{< ci-details summary="Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (Junyoung Chung et al., 2014)">}}
Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, Yoshua Bengio. (2014)  
**Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling**  
ArXiv  
[Paper Link](https://www.semanticscholar.org/paper/adfcf065e15fd3bc9badf6145034c84dfb08f204)  
Influential Citation Count (1359), SS-ID (adfcf065e15fd3bc9badf6145034c84dfb08f204)  
**ABSTRACT**  
In this paper we compare different types of recurrent units in recurrent neural networks (RNNs). Especially, we focus on more sophisticated units that implement a gating mechanism, such as a long short-term memory (LSTM) unit and a recently proposed gated recurrent unit (GRU). We evaluate these recurrent units on the tasks of polyphonic music modeling and speech signal modeling. Our experiments revealed that these advanced recurrent units are indeed better than more traditional recurrent units such as tanh units. Also, we found GRU to be comparable to LSTM.
{{< /ci-details >}}
{{< ci-details summary="A Clockwork RNN (J. Koutník et al., 2014)">}}
J. Koutník, Klaus Greff, Faustino J. Gomez, J. Schmidhuber. (2014)  
**A Clockwork RNN**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/5522764282c85aea422f1c4dc92ff7e0ca6987bc)  
Influential Citation Count (59), SS-ID (5522764282c85aea422f1c4dc92ff7e0ca6987bc)  
**ABSTRACT**  
Sequence prediction and classification are ubiquitous and challenging problems in machine learning that can require identifying complex dependencies between temporally distant inputs. Recurrent Neural Networks (RNNs) have the ability, in theory, to cope with these temporal dependencies by virtue of the short-term memory implemented by their recurrent (feedback) connections. However, in practice they are difficult to train successfully when long-term memory is required. This paper introduces a simple, yet powerful modification to the simple RNN (SRN) architecture, the Clockwork RNN (CW-RNN), in which the hidden layer is partitioned into separate modules, each processing inputs at its own temporal granularity, making computations only at its prescribed clock rate. Rather than making the standard RNN models more complex, CW-RNN reduces the number of SRN parameters, improves the performance significantly in the tasks tested, and speeds up the network evaluation. The network is demonstrated in preliminary experiments involving three tasks: audio signal generation, TIMIT spoken word classification, where it outperforms both SRN and LSTM networks, and online handwriting recognition, where it outperforms SRNs.
{{< /ci-details >}}
{{< ci-details summary="On the difficulty of training recurrent neural networks (Razvan Pascanu et al., 2012)">}}
Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. (2012)  
**On the difficulty of training recurrent neural networks**  
ICML  
[Paper Link](https://www.semanticscholar.org/paper/84069287da0a6b488b8c933f3cb5be759cb6237e)  
Influential Citation Count (305), SS-ID (84069287da0a6b488b8c933f3cb5be759cb6237e)  
**ABSTRACT**  
There are two widely known issues with properly training recurrent neural networks, the vanishing and the exploding gradient problems detailed in Bengio et al. (1994). In this paper we attempt to improve the understanding of the underlying issues by exploring these problems from an analytical, a geometric and a dynamical systems perspective. Our analysis is used to justify a simple yet effective solution. We propose a gradient norm clipping strategy to deal with exploding gradients and a soft constraint for the vanishing gradients problem. We validate empirically our hypothesis and proposed solutions in the experimental section.
{{< /ci-details >}}
{{< ci-details summary="A brief survey on sequence classification (Zhengzheng Xing et al., 2010)">}}
Zhengzheng Xing, J. Pei, Eamonn J. Keogh. (2010)  
**A brief survey on sequence classification**  
SKDD  
[Paper Link](https://www.semanticscholar.org/paper/019475245d325f70fc3c930b9e96c0c48196ca21)  
Influential Citation Count (34), SS-ID (019475245d325f70fc3c930b9e96c0c48196ca21)  
**ABSTRACT**  
Sequence classification has a broad range of applications such as genomic analysis, information retrieval, health informatics, finance, and abnormal detection. Different from the classification task on feature vectors, sequences do not have explicit features. Even with sophisticated feature selection techniques, the dimensionality of potential features may still be very high and the sequential nature of features is difficult to capture. This makes sequence classification a more challenging task than classification on feature vectors. In this paper, we present a brief review of the existing work on sequence classification. We summarize the sequence classification in terms of methodologies and application domains. We also provide a review on several extensions of the sequence classification problem, such as early classification on sequences and semi-supervised learning on sequences.
{{< /ci-details >}}
{{< ci-details summary="Gradient-based learning applied to document recognition (Yann LeCun et al., 1998)">}}
Yann LeCun, L. Bottou, Yoshua Bengio, P. Haffner. (1998)  
**Gradient-based learning applied to document recognition**  
Proc. IEEE  
[Paper Link](https://www.semanticscholar.org/paper/162d958ff885f1462aeda91cd72582323fd6a1f4)  
Influential Citation Count (5728), SS-ID (162d958ff885f1462aeda91cd72582323fd6a1f4)  
**ABSTRACT**  
Multilayer neural networks trained with the back-propagation algorithm constitute the best example of a successful gradient based learning technique. Given an appropriate network architecture, gradient-based learning algorithms can be used to synthesize a complex decision surface that can classify high-dimensional patterns, such as handwritten characters, with minimal preprocessing. This paper reviews various methods applied to handwritten character recognition and compares them on a standard handwritten digit recognition task. Convolutional neural networks, which are specifically designed to deal with the variability of 2D shapes, are shown to outperform all other techniques. Real-life document recognition systems are composed of multiple modules including field extraction, segmentation recognition, and language modeling. A new learning paradigm, called graph transformer networks (GTN), allows such multimodule systems to be trained globally using gradient-based methods so as to minimize an overall performance measure. Two systems for online handwriting recognition are described. Experiments demonstrate the advantage of global training, and the flexibility of graph transformer networks. A graph transformer network for reading a bank cheque is also described. It uses convolutional neural network character recognizers combined with global training techniques to provide record accuracy on business and personal cheques. It is deployed commercially and reads several million cheques per day.
{{< /ci-details >}}
{{< ci-details summary="Long Short-Term Memory (S. Hochreiter et al., 1997)">}}
S. Hochreiter, J. Schmidhuber. (1997)  
**Long Short-Term Memory**  
Neural Computation  
[Paper Link](https://www.semanticscholar.org/paper/44d2abe2175df8153f465f6c39b68b76a0d40ab9)  
Influential Citation Count (9421), SS-ID (44d2abe2175df8153f465f6c39b68b76a0d40ab9)  
**ABSTRACT**  
Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through constant error carousels within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O. 1. Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.
{{< /ci-details >}}
{{< ci-details summary="Hierarchical Recurrent Neural Networks for Long-Term Dependencies (Salah El Hihi et al., 1995)">}}
Salah El Hihi, Yoshua Bengio. (1995)  
**Hierarchical Recurrent Neural Networks for Long-Term Dependencies**  
NIPS  
[Paper Link](https://www.semanticscholar.org/paper/b13813b49f160e1a2010c44bd4fb3d09a28446e3)  
Influential Citation Count (17), SS-ID (b13813b49f160e1a2010c44bd4fb3d09a28446e3)  
**ABSTRACT**  
We have already shown that extracting long-term dependencies from sequential data is difficult, both for determimstic dynamical systems such as recurrent networks, and probabilistic models such as hidden Markov models (HMMs) or input/output hidden Markov models (IOHMMs). In practice, to avoid this problem, researchers have used domain specific a-priori knowledge to give meaning to the hidden or state variables representing past context. In this paper, we propose to use a more general type of a-priori knowledge, namely that the temporal dependencies are structured hierarchically. This implies that long-term dependencies are represented by variables with a long time scale. This principle is applied to a recurrent network which includes delays and multiple time scales. Experiments confirm the advantages of such structures. A similar approach is proposed for HMMs and IOHMMs.
{{< /ci-details >}}
{{< ci-details summary="Building a Large Annotated Corpus of English: The Penn Treebank (M. Marcus et al., 1993)">}}
M. Marcus, Beatrice Santorini, Mary Ann Marcinkiewicz. (1993)  
**Building a Large Annotated Corpus of English: The Penn Treebank**  
CL  
[Paper Link](https://www.semanticscholar.org/paper/0b44fcbeea9415d400c5f5789d6b892b6f98daff)  
Influential Citation Count (1366), SS-ID (0b44fcbeea9415d400c5f5789d6b892b6f98daff)  
**ABSTRACT**  
Abstract : As a result of this grant, the researchers have now published oil CDROM a corpus of over 4 million words of running text annotated with part-of- speech (POS) tags, with over 3 million words of that material assigned skeletal grammatical structure. This material now includes a fully hand-parsed version of the classic Brown corpus. About one half of the papers at the ACL Workshop on Using Large Text Corpora this past summer were based on the materials generated by this grant.
{{< /ci-details >}}
{{< ci-details summary="A SYSTEMIC STUDY OF MONETARY SYSTEMS (E. Caianiello et al., 1982)">}}
E. Caianiello, G. Scarpetta, G. Simoncelli. (1982)  
**A SYSTEMIC STUDY OF MONETARY SYSTEMS**  
  
[Paper Link](https://www.semanticscholar.org/paper/5970b69d573d2717e5f5be48aaff42d70f9107d7)  
Influential Citation Count (1), SS-ID (5970b69d573d2717e5f5be48aaff42d70f9107d7)  
**ABSTRACT**  
Abstract Self-organizing systems are defined as able to change their structure, according to need, within specific equivalence classes. Once hierarchical levels and their value functions are assigned, requirements of invariance under transformations within an equivalence class can be used as a principle to determine the population of each level. This program is carried out in complete detail for the monetary system. It has been possible to deduce the distribution law that specifies how many “coins” for each level must stay in circulation; this law agrees well with empirically observed data from about ten different countries. It is also possible to understand the processes of reorganization of monetary systems induced by inflation, i.e. the disappearance of coins of low value and the simultaneous issue of coins or higher value.
{{< /ci-details >}}
