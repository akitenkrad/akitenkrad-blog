---
draft: false
title: "arXiv @ 2024.03.05"
date: 2024-03-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.03.05"
    identifier: arxiv_20240305
    parent: 202403_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.AR (1)](#csar-1)
- [cs.LG (3)](#cslg-3)
- [cs.SE (1)](#csse-1)
- [eess.AS (1)](#eessas-1)
- [eess.IV (1)](#eessiv-1)

## Keywords

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>keyword</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>

<script>
$(function() {
  $("table").addClass("keyword-table table-bordered border-success");
  $("table thead").addClass("sticky-top");
  $("table tbody td").css("text-align", "");
});
</script>


## cs.LG (3)



### (1/3 | 1/7) Improving Uncertainty Sampling with Bell Curve Weight Function (Zan-Kai Chong et al., 2024)

{{<citation>}}

Zan-Kai Chong, Hiroyuki Ohsaki, Bok-Min Goi. (2024)  
**Improving Uncertainty Sampling with Bell Curve Weight Function**
<br/>
<button class="copy-to-clipboard" title="Improving Uncertainty Sampling with Bell Curve Weight Function" index=1>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-1 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keyword Score: 50  
Keywords: Active Learning, Simulation, Simulator, Supervised Learning, Supervised Learning  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01352v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01352v1.pdf" filename="2403.01352v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Typically, a <b>supervised</b> <b>learning</b> model is trained using passive learning by randomly selecting unlabelled instances to annotate. This approach is effective for learning a model, but can be costly in cases where acquiring labelled instances is expensive. For example, it can be time-consuming to manually identify spam mails (labelled instances) from thousands of emails (unlabelled instances) flooding an inbox during initial data collection. Generally, we answer the above scenario with uncertainty sampling, an <b>active</b> <b>learning</b> method that improves the efficiency of <b>supervised</b> <b>learning</b> by using fewer labelled instances than passive learning. Given an unlabelled data pool, uncertainty sampling queries the labels of instances where the predicted probabilities, p, fall into the uncertainty region, i.e., $p \approx 0.5$. The newly acquired labels are then added to the existing labelled data pool to learn a new model. Nonetheless, the performance of uncertainty sampling is susceptible to the area of unpredictable responses (AUR) and the nature of the dataset. It is difficult to determine whether to use passive learning or uncertainty sampling without prior knowledge of a new dataset. To address this issue, we propose bell curve sampling, which employs a bell curve weight function to acquire new labels. With the bell curve centred at p=0.5, bell curve sampling selects instances whose predicted values are in the uncertainty area most of the time without neglecting the rest. <b>Simulation</b> results show that, most of the time bell curve sampling outperforms uncertainty sampling and passive learning in datasets of different natures and with AUR.

{{</citation>}}


### (2/3 | 2/7) Bandit Profit-maximization for Targeted Marketing (Joon Suk Huh et al., 2024)

{{<citation>}}

Joon Suk Huh, Ellen Vitercik, Kirthevasan Kandasamy. (2024)  
**Bandit Profit-maximization for Targeted Marketing**
<br/>
<button class="copy-to-clipboard" title="Bandit Profit-maximization for Targeted Marketing" index=2>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-2 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-GT, cs-LG, cs.LG, econ-GN, q-fin-EC, q-fin-GN  
Keyword Score: 10  
Keywords: Bandit Algorithm  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01361v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01361v1.pdf" filename="2403.01361v1.pdf">Download PDF</button>

---


**ABSTRACT**  
We study a sequential profit-maximization problem, optimizing for both price and ancillary variables like marketing expenditures. Specifically, we aim to maximize profit over an arbitrary sequence of multiple demand curves, each dependent on a distinct ancillary variable, but sharing the same price. A prototypical example is targeted marketing, where a firm (seller) wishes to sell a product over multiple markets. The firm may invest different marketing expenditures for different markets to optimize customer acquisition, but must maintain the same price across all markets. Moreover, markets may have heterogeneous demand curves, each responding to prices and marketing expenditures differently. The firm's objective is to maximize its gross profit, the total revenue minus marketing costs. Our results are near-optimal algorithms for this class of problems in an adversarial <b>bandit</b> setting, where demand curves are arbitrary non-adaptive sequences, and the firm observes only noisy evaluations of chosen points on the demand curves. We prove a regret upper bound of $\widetilde{\mathcal{O}}\big(nT^{3/4}\big)$ and a lower bound of $\Omega\big((nT)^{3/4}\big)$ for monotonic demand curves, and a regret bound of $\widetilde{\Theta}\big(nT^{2/3}\big)$ for demands curves that are monotonic in price and concave in the ancillary variables.

{{</citation>}}


### (3/3 | 3/7) SANGRIA: Stacked Autoencoder Neural Networks with Gradient Boosting for Indoor Localization (Danish Gufran et al., 2024)

{{<citation>}}

Danish Gufran, Saideep Tiku, Sudeep Pasricha. (2024)  
**SANGRIA: Stacked Autoencoder Neural Networks with Gradient Boosting for Indoor Localization**
<br/>
<button class="copy-to-clipboard" title="SANGRIA: Stacked Autoencoder Neural Networks with Gradient Boosting for Indoor Localization" index=3>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-3 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keyword Score: 10  
Keywords: Autoencoder  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01348v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01348v1.pdf" filename="2403.01348v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Indoor localization is a critical task in many embedded applications, such as asset tracking, emergency response, and realtime navigation. In this article, we propose a novel fingerprintingbased framework for indoor localization called SANGRIA that uses stacked <b>autoencoder</b> neural networks with gradient boosted trees. Our approach is designed to overcome the device heterogeneity challenge that can create uncertainty in wireless signal measurements across embedded devices used for localization. We compare SANGRIA to several state-of-the-art frameworks and demonstrate 42.96% lower average localization error across diverse indoor locales and heterogeneous devices.

{{</citation>}}


## eess.IV (1)



### (1/1 | 4/7) Enhancing Retinal Vascular Structure Segmentation in Images With a Novel Design Two-Path Interactive Fusion Module Model (Rui Yang et al., 2024)

{{<citation>}}

Rui Yang, Shunpu Zhang. (2024)  
**Enhancing Retinal Vascular Structure Segmentation in Images With a Novel Design Two-Path Interactive Fusion Module Model**
<br/>
<button class="copy-to-clipboard" title="Enhancing Retinal Vascular Structure Segmentation in Images With a Novel Design Two-Path Interactive Fusion Module Model" index=4>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-4 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keyword Score: 20  
Keywords: Convolution, Transformer  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01362v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01362v1.pdf" filename="2403.01362v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Precision in identifying and differentiating micro and macro blood vessels in the retina is crucial for the diagnosis of retinal diseases, although it poses a significant challenge. Current autoencoding-based segmentation approaches encounter limitations as they are constrained by the encoder and undergo a reduction in resolution during the encoding stage. The inability to recover lost information in the decoding phase further impedes these approaches. Consequently, their capacity to extract the retinal microvascular structure is restricted. To address this issue, we introduce Swin-Res-Net, a specialized module designed to enhance the precision of retinal vessel segmentation. Swin-Res-Net utilizes the Swin <b>transformer</b> which uses shifted windows with displacement for partitioning, to reduce network complexity and accelerate model convergence. Additionally, the model incorporates interactive fusion with a functional module in the Res2Net architecture. The Res2Net leverages multi-scale techniques to enlarge the receptive field of the <b>convolutional</b> kernel, enabling the extraction of additional semantic information from the image. This combination creates a new module that enhances the localization and separation of micro vessels in the retina. To improve the efficiency of processing vascular information, we've added a module to eliminate redundant information between the encoding and decoding steps. Our proposed architecture produces outstanding results, either meeting or surpassing those of other published models. The AUC reflects significant enhancements, achieving values of 0.9956, 0.9931, and 0.9946 in pixel-wise segmentation of retinal vessels across three widely utilized datasets: CHASE-DB1, DRIVE, and STARE, respectively. Moreover, Swin-Res-Net outperforms alternative architectures, demonstrating superior performance in both IOU and F1 measure metrics.

{{</citation>}}


## cs.SE (1)



### (1/1 | 5/7) ModelWriter: Text & Model-Synchronized Document Engineering Platform (Ferhat Erata et al., 2024)

{{<citation>}}

Ferhat Erata, Claire Gardent, Bikash Gyawali, Anastasia Shimorina, Yvan Lussaud, Bedir Tekinerdogan, Geylani Kardas, Anne Monceaux. (2024)  
**ModelWriter: Text & Model-Synchronized Document Engineering Platform**
<br/>
<button class="copy-to-clipboard" title="ModelWriter: Text & Model-Synchronized Document Engineering Platform" index=5>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-5 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keyword Score: 20  
Keywords: Reasoning, Semantic Parsing  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01359v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01359v1.pdf" filename="2403.01359v1.pdf">Download PDF</button>

---


**ABSTRACT**  
The ModelWriter platform provides a generic framework for automated traceability analysis. In this paper, we demonstrate how this framework can be used to trace the consistency and completeness of technical documents that consist of a set of System Installation Design Principles used by Airbus to ensure the correctness of aircraft system installation. We show in particular, how the platform allows the integration of two types of <b>reasoning:</b> <b>reasoning</b> about the meaning of text using <b>semantic</b> <b>parsing</b> and description logic theorem proving; and <b>reasoning</b> about document structure using first-order relational logic and finite model finding for traceability analysis.

{{</citation>}}


## cs.AR (1)



### (1/1 | 6/7) Efficient FIR filtering with Bit Layer Multiply Accumulator (Vincenzo Liguori, 2024)

{{<citation>}}

Vincenzo Liguori. (2024)  
**Efficient FIR filtering with Bit Layer Multiply Accumulator**
<br/>
<button class="copy-to-clipboard" title="Efficient FIR filtering with Bit Layer Multiply Accumulator" index=6>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-6 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: cs.AR  
Categories: cs-AR, cs-ET, cs.AR  
Keyword Score: 10  
Keywords: Quantization  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01351v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01351v1.pdf" filename="2403.01351v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Bit Layer Multiplier Accumulator (BLMAC) is an efficient method to perform dot products without multiplications that exploits the bit level sparsity of the weights. A total of 1,980,000 low, high, band pass and band stop type I FIR filters were generated by systematically sweeping through the cut off frequencies and by varying the number of taps from 55 to 255. After their coefficients were <b>quantized</b> to 16 bits, applying the filter using a BLMAC required, on average, from ~123.3 to ~513.6 additions, depending on the number of taps. A BLMAC dot product machine, specialised for 127 taps FIR filters, was designed for AMD FPGAs. The design footprint is ~110 LUTs, including coefficient and sample storage and is able to apply the filter in ~232 clock cycles on average. This implies a filtering rate of 1.4-3.4 Msamples/s, depending on the FPGA family.

{{</citation>}}


## eess.AS (1)



### (1/1 | 7/7) a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification (Hye-jin Shim et al., 2024)

{{<citation>}}

Hye-jin Shim, Jee-weon Jung, Tomi Kinnunen, Nicholas Evans, Jean-Francois Bonastre, Itshak Lapidot. (2024)  
**a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification**
<br/>
<button class="copy-to-clipboard" title="a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification" index=7>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-7 align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: eess.AS  
Categories: cs-LG, eess-AS, eess.AS  
Keyword Score: 6  
Keywords: Benchmarking, Benchmarking  
<a type="button" class="btn btn-outline-primary" href="http://arxiv.org/abs/2403.01355v1" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="https://arxiv.org/pdf/2403.01355v1.pdf" filename="2403.01355v1.pdf">Download PDF</button>

---


**ABSTRACT**  
Spoofing detection is today a mainstream research topic. Standard metrics can be applied to evaluate the performance of isolated spoofing detection solutions and others have been proposed to support their evaluation when they are combined with speaker detection. These either have well-known deficiencies or restrict the architectural approach to combine speaker and spoof detectors. In this paper, we propose an architecture-agnostic detection cost function (a-DCF). A generalisation of the original DCF used widely for the assessment of automatic speaker verification (ASV), the a-DCF is designed for the evaluation of spoofing-robust ASV. Like the DCF, the a-DCF reflects the cost of decisions in a Bayes risk sense, with explicitly defined class priors and detection cost model. We demonstrate the merit of the a-DCF through the <b>benchmarking</b> evaluation of architecturally-heterogeneous spoofing-robust ASV solutions.

{{</citation>}}
