---
draft: false
title: "arXiv @ 2024.01.26"
date: 2024-01-26
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.26"
    identifier: arxiv_20240126
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.LG (25)](#cslg-25)
- [cs.CV (26)](#cscv-26)
- [cs.SD (1)](#cssd-1)
- [cs.CY (5)](#cscy-5)
- [cs.CL (27)](#cscl-27)
- [cs.IR (9)](#csir-9)
- [cs.SI (2)](#cssi-2)
- [cs.HC (8)](#cshc-8)
- [cs.SE (3)](#csse-3)
- [cs.DL (2)](#csdl-2)
- [math.NA (1)](#mathna-1)
- [cs.DC (2)](#csdc-2)
- [cs.CR (2)](#cscr-2)
- [cs.CE (1)](#csce-1)
- [hep-ph (1)](#hep-ph-1)
- [q-fin.RM (1)](#q-finrm-1)
- [cs.GT (1)](#csgt-1)
- [cs.RO (1)](#csro-1)
- [cs.OS (1)](#csos-1)
- [cs.NI (1)](#csni-1)
- [cs.AR (1)](#csar-1)
- [eess.IV (1)](#eessiv-1)
- [q-bio.GN (1)](#q-biogn-1)

## cs.LG (25)



### (1/123) Inverse Molecular Design with Multi-Conditional Diffusion Guidance (Gang Liu et al., 2024)

{{<citation>}}

Gang Liu, Jiaxin Xu, Tengfei Luo, Meng Jiang. (2024)  
**Inverse Molecular Design with Multi-Conditional Diffusion Guidance**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13858v1)  

---


**ABSTRACT**  
Inverse molecular design with diffusion models holds great potential for advancements in material and drug discovery. Despite success in unconditional molecule generation, integrating multiple properties such as synthetic score and gas permeability as condition constraints into diffusion models remains unexplored. We introduce multi-conditional diffusion guidance. The proposed Transformer-based denoising model has a condition encoder that learns the representations of numerical and categorical conditions. The denoising model, consisting of a structure encoder-decoder, is trained for denoising under the representation of conditions. The diffusion process becomes graph-dependent to accurately estimate graph-related noise in molecules, unlike the previous models that focus solely on the marginal distributions of atoms or bonds. We extensively validate our model for multi-conditional polymer and small molecule generation. Results demonstrate our superiority across metrics from distribution learning to condition control for molecular properties. An inverse polymer design task for gas separation with feedback from domain experts further demonstrates its practical utility.

{{</citation>}}


### (2/123) Embedding Attack Project (Work Report) (Jiameng Pu et al., 2024)

{{<citation>}}

Jiameng Pu, Zafar Takhirov. (2024)  
**Embedding Attack Project (Work Report)**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI, Computer Vision, Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13854v1)  

---


**ABSTRACT**  
This report summarizes all the MIA experiments (Membership Inference Attacks) of the Embedding Attack Project, including threat models, experimental setup, experimental results, findings and discussion. Current results cover the evaluation of two main MIA strategies (loss-based and embedding-based MIAs) on 6 AI models ranging from Computer Vision to Language Modelling. There are two ongoing experiments on MIA defense and neighborhood-comparison embedding attacks. These are ongoing projects.   The current work on MIA and PIA can be summarized into six conclusions: (1) Amount of overfitting is directly proportional to model's vulnerability; (2) early embedding layers in the model are less susceptible to privacy leaks; (3) Deeper model layers contain more membership information; (4) Models are more vulnerable to MIA if both embeddings and corresponding training labels are compromised; (5) it is possible to use pseudo-labels to increase the MIA success; and (6) although MIA and PIA success rates are proportional, reducing the MIA does not necessarily reduce the PIA.

{{</citation>}}


### (3/123) The Calibration Gap between Model and Human Confidence in Large Language Models (Mark Steyvers et al., 2024)

{{<citation>}}

Mark Steyvers, Heliodoro Tejeda, Aakriti Kumar, Catarina Belem, Sheer Karny, Xinyue Hu, Lukas Mayer, Padhraic Smyth. (2024)  
**The Calibration Gap between Model and Human Confidence in Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-HC, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13835v1)  

---


**ABSTRACT**  
For large language models (LLMs) to be trusted by humans they need to be well-calibrated in the sense that they can accurately assess and communicate how likely it is that their predictions are correct. Recent work has focused on the quality of internal LLM confidence assessments, but the question remains of how well LLMs can communicate this internal model confidence to human users. This paper explores the disparity between external human confidence in an LLM's responses and the internal confidence of the model. Through experiments involving multiple-choice questions, we systematically examine human users' ability to discern the reliability of LLM outputs. Our study focuses on two key areas: (1) assessing users' perception of true LLM confidence and (2) investigating the impact of tailored explanations on this perception. The research highlights that default explanations from LLMs often lead to user overestimation of both the model's confidence and its' accuracy. By modifying the explanations to more accurately reflect the LLM's internal confidence, we observe a significant shift in user perception, aligning it more closely with the model's actual confidence levels. This adjustment in explanatory approach demonstrates potential for enhancing user trust and accuracy in assessing LLM outputs. The findings underscore the importance of transparent communication of confidence levels in LLMs, particularly in high-stakes applications where understanding the reliability of AI-generated information is essential.

{{</citation>}}


### (4/123) Traffic Learning and Proactive UAV Trajectory Planning for Data Uplink in Markovian IoT Models (Eslam Eldeeb et al., 2024)

{{<citation>}}

Eslam Eldeeb, Mohammad Shehab, Hirley Alves. (2024)  
**Traffic Learning and Proactive UAV Trajectory Planning for Data Uplink in Markovian IoT Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NI, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.13827v1)  

---


**ABSTRACT**  
The age of information (AoI) is used to measure the freshness of the data. In IoT networks, the traditional resource management schemes rely on a message exchange between the devices and the base station (BS) before communication which causes high AoI, high energy consumption, and low reliability. Unmanned aerial vehicles (UAVs) as flying BSs have many advantages in minimizing the AoI, energy-saving, and throughput improvement. In this paper, we present a novel learning-based framework that estimates the traffic arrival of IoT devices based on Markovian events. The learning proceeds to optimize the trajectory of multiple UAVs and their scheduling policy. First, the BS predicts the future traffic of the devices. We compare two traffic predictors: the forward algorithm (FA) and the long short-term memory (LSTM). Afterward, we propose a deep reinforcement learning (DRL) approach to optimize the optimal policy of each UAV. Finally, we manipulate the optimum reward function for the proposed DRL approach. Simulation results show that the proposed algorithm outperforms the random-walk (RW) baseline model regarding the AoI, scheduling accuracy, and transmission power.

{{</citation>}}


### (5/123) Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on Hugging Face (Xinyu Yang et al., 2024)

{{<citation>}}

Xinyu Yang, Weixin Liang, James Zou. (2024)  
**Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on Hugging Face**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13822v1)  

---


**ABSTRACT**  
Advances in machine learning are closely tied to the creation of datasets. While data documentation is widely recognized as essential to the reliability, reproducibility, and transparency of ML, we lack a systematic empirical understanding of current dataset documentation practices. To shed light on this question, here we take Hugging Face -- one of the largest platforms for sharing and collaborating on ML models and datasets -- as a prominent case study. By analyzing all 7,433 dataset documentation on Hugging Face, our investigation provides an overview of the Hugging Face dataset ecosystem and insights into dataset documentation practices, yielding 5 main findings: (1) The dataset card completion rate shows marked heterogeneity correlated with dataset popularity. (2) A granular examination of each section within the dataset card reveals that the practitioners seem to prioritize Dataset Description and Dataset Structure sections, while the Considerations for Using the Data section receives the lowest proportion of content. (3) By analyzing the subsections within each section and utilizing topic modeling to identify key topics, we uncover what is discussed in each section, and underscore significant themes encompassing both technical and social impacts, as well as limitations within the Considerations for Using the Data section. (4) Our findings also highlight the need for improved accessibility and reproducibility of datasets in the Usage sections. (5) In addition, our human annotation evaluation emphasizes the pivotal role of comprehensive dataset content in shaping individuals' perceptions of a dataset card's overall quality. Overall, our study offers a unique perspective on analyzing dataset documentation through large-scale data science analysis and underlines the need for more thorough dataset documentation in machine learning research.

{{</citation>}}


### (6/123) NLICE: Synthetic Medical Record Generation for Effective Primary Healthcare Differential Diagnosis (Zaid Al-Ars et al., 2024)

{{<citation>}}

Zaid Al-Ars, Obinna Agba, Zhuoran Guo, Christiaan Boerkamp, Ziyaad Jaber, Tareq Jaber. (2024)  
**NLICE: Synthetic Medical Record Generation for Effective Primary Healthcare Differential Diagnosis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2401.13756v1)  

---


**ABSTRACT**  
This paper offers a systematic method for creating medical knowledge-grounded patient records for use in activities involving differential diagnosis. Additionally, an assessment of machine learning models that can differentiate between various conditions based on given symptoms is also provided. We use a public disease-symptom data source called SymCat in combination with Synthea to construct the patients records. In order to increase the expressive nature of the synthetic data, we use a medically-standardized symptom modeling method called NLICE to augment the synthetic data with additional contextual information for each condition. In addition, Naive Bayes and Random Forest models are evaluated and compared on the synthetic data. The paper shows how to successfully construct SymCat-based and NLICE-based datasets. We also show results for the effectiveness of using the datasets to train predictive disease models. The SymCat-based dataset is able to train a Naive Bayes and Random Forest model yielding a 58.8% and 57.1% Top-1 accuracy score, respectively. In contrast, the NLICE-based dataset improves the results, with a Top-1 accuracy of 82.0% and Top-5 accuracy values of more than 90% for both models. Our proposed data generation approach solves a major barrier to the application of artificial intelligence methods in the healthcare domain. Our novel NLICE symptom modeling approach addresses the incomplete and insufficient information problem in the current binary symptom representation approach. The NLICE code is open sourced at https://github.com/guozhuoran918/NLICE.

{{</citation>}}


### (7/123) Conformal Prediction Sets Improve Human Decision Making (Jesse C. Cresswell et al., 2024)

{{<citation>}}

Jesse C. Cresswell, Yi Sui, Bhargava Kumar, Noël Vouitsis. (2024)  
**Conformal Prediction Sets Improve Human Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13744v1)  

---


**ABSTRACT**  
In response to everyday queries, humans explicitly signal uncertainty and offer alternative answers when they are unsure. Machine learning models that output calibrated prediction sets through conformal prediction mimic this human behaviour; larger sets signal greater uncertainty while providing alternatives. In this work, we study the usefulness of conformal prediction sets as an aid for human decision making by conducting a pre-registered randomized controlled trial with conformal prediction sets provided to human subjects. With statistical significance, we find that when humans are given conformal prediction sets their accuracy on tasks improves compared to fixed-size prediction sets with the same coverage guarantee. The results show that quantifying model uncertainty with conformal prediction is helpful for human-in-the-loop decision making and human-AI teams.

{{</citation>}}


### (8/123) The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations (Matthias Lehmann, 2024)

{{<citation>}}

Matthias Lehmann. (2024)  
**The Definitive Guide to Policy Gradients in Deep Reinforcement Learning: Theory, Algorithms and Implementations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13662v1)  

---


**ABSTRACT**  
In recent years, various powerful policy gradient algorithms have been proposed in deep reinforcement learning. While all these algorithms build on the Policy Gradient Theorem, the specific design choices differ significantly across algorithms. We provide a holistic overview of on-policy policy gradient algorithms to facilitate the understanding of both their theoretical foundations and their practical implementations. In this overview, we include a detailed proof of the continuous version of the Policy Gradient Theorem, convergence results and a comprehensive discussion of practical algorithms. We compare the most prominent algorithms on continuous control environments and provide insights on the benefits of regularization. All code is available at https://github.com/Matt00n/PolicyGradientsJax.

{{</citation>}}


### (9/123) Inadequacy of common stochastic neural networks for reliable clinical decision support (Adrian Lindenmeyer et al., 2024)

{{<citation>}}

Adrian Lindenmeyer, Malte Blattmann, Stefan Franke, Thomas Neumuth, Daniel Schneider. (2024)  
**Inadequacy of common stochastic neural networks for reliable clinical decision support**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2401.13657v2)  

---


**ABSTRACT**  
Widespread adoption of AI for medical decision making is still hindered due to ethical and safety-related concerns. For AI-based decision support systems in healthcare settings it is paramount to be reliable and trustworthy. Common deep learning approaches, however, have the tendency towards overconfidence under data shift. Such inappropriate extrapolation beyond evidence-based scenarios may have dire consequences. This highlights the importance of reliable estimation of local uncertainty and its communication to the end user. While stochastic neural networks have been heralded as a potential solution to these issues, this study investigates their actual reliability in clinical applications. We centered our analysis on the exemplary use case of mortality prediction for ICU hospitalizations using EHR from MIMIC3 study. For predictions on the EHR time series, Encoder-Only Transformer models were employed. Stochasticity of model functions was achieved by incorporating common methods such as Bayesian neural network layers and model ensembles. Our models achieve state of the art performance in terms of discrimination performance (AUC ROC: 0.868+-0.011, AUC PR: 0.554+-0.034) and calibration on the mortality prediction benchmark. However, epistemic uncertainty is critically underestimated by the selected stochastic deep learning methods. A heuristic proof for the responsible collapse of the posterior distribution is provided. Our findings reveal the inadequacy of commonly used stochastic deep learning approaches to reliably recognize OoD samples. In both methods, unsubstantiated model confidence is not prevented due to strongly biased functional posteriors, rendering them inappropriate for reliable clinical decision support. This highlights the need for approaches with more strictly enforced or inherent distance-awareness to known data points, e.g., using kernel-based techniques.

{{</citation>}}


### (10/123) Prompt Weight Experiments for LLM Instruction Fine-Tuning (Mathew Huerta-Enochian, 2024)

{{<citation>}}

Mathew Huerta-Enochian. (2024)  
**Prompt Weight Experiments for LLM Instruction Fine-Tuning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2401.13586v1)  

---


**ABSTRACT**  
We present a small study analyzing how prompt token classification loss weighting (PLW) affects the performance of 7B-size LLaMA models fine-tuned on instruction tasks. We recreated Stanford's Alpaca experiment with both LLaMA 1 and LLaMA 2 using multiple instruction datasets. We found that models fine-tuned on our short-completion dataset have a negative quadratic relationship with PLW while models fine-tuned on long-completion datasets were unaffected by PLW.

{{</citation>}}


### (11/123) Multi-Agent Diagnostics for Robustness via Illuminated Diversity (Mikayel Samvelyan et al., 2024)

{{<citation>}}

Mikayel Samvelyan, Davide Paglieri, Minqi Jiang, Jack Parker-Holder, Tim Rocktäschel. (2024)  
**Multi-Agent Diagnostics for Robustness via Illuminated Diversity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2401.13460v1)  

---


**ABSTRACT**  
In the rapidly advancing field of multi-agent systems, ensuring robustness in unfamiliar and adversarial settings is crucial. Notwithstanding their outstanding performance in familiar environments, these systems often falter in new situations due to overfitting during the training phase. This is especially pronounced in settings where both cooperative and competitive behaviours are present, encapsulating a dual nature of overfitting and generalisation challenges. To address this issue, we present Multi-Agent Diagnostics for Robustness via Illuminated Diversity (MADRID), a novel approach for generating diverse adversarial scenarios that expose strategic vulnerabilities in pre-trained multi-agent policies. Leveraging the concepts from open-ended learning, MADRID navigates the vast space of adversarial settings, employing a target policy's regret to gauge the vulnerabilities of these settings. We evaluate the effectiveness of MADRID on the 11vs11 version of Google Research Football, one of the most complex environments for multi-agent reinforcement learning. Specifically, we employ MADRID for generating a diverse array of adversarial settings for TiZero, the state-of-the-art approach which "masters" the game through 45 days of training on a large-scale distributed infrastructure. We expose key shortcomings in TiZero's tactical decision-making, underlining the crucial importance of rigorous evaluation in multi-agent systems.

{{</citation>}}


### (12/123) Symbolic Equation Solving via Reinforcement Learning (Lennart Dabelow et al., 2024)

{{<citation>}}

Lennart Dabelow, Masahito Ueda. (2024)  
**Symbolic Equation Solving via Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SC, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13447v1)  

---


**ABSTRACT**  
Machine-learning methods are gradually being adopted in a great variety of social, economic, and scientific contexts, yet they are notorious for struggling with exact mathematics. A typical example is computer algebra, which includes tasks like simplifying mathematical terms, calculating formal derivatives, or finding exact solutions of algebraic equations. Traditional software packages for these purposes are commonly based on a huge database of rules for how a specific operation (e.g., differentiation) transforms a certain term (e.g., sine function) into another one (e.g., cosine function). Thus far, these rules have usually needed to be discovered and subsequently programmed by humans. Focusing on the paradigmatic example of solving linear equations in symbolic form, we demonstrate how the process of finding elementary transformation rules and step-by-step solutions can be automated using reinforcement learning with deep neural networks.

{{</citation>}}


### (13/123) Beyond Accuracy-Fairness: Stop evaluating bias mitigation methods solely on between-group metrics (Sofie Goethals et al., 2024)

{{<citation>}}

Sofie Goethals, Toon Calders, David Martens. (2024)  
**Beyond Accuracy-Fairness: Stop evaluating bias mitigation methods solely on between-group metrics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.13391v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) finds widespread applications across various domains, sparking concerns about fairness in its deployment. While fairness in AI remains a central concern, the prevailing discourse often emphasizes outcome-based metrics without a nuanced consideration of the differential impacts within subgroups. Bias mitigation techniques do not only affect the ranking of pairs of instances across sensitive groups, but often also significantly affect the ranking of instances within these groups. Such changes are hard to explain and raise concerns regarding the validity of the intervention. Unfortunately, these effects largely remain under the radar in the accuracy-fairness evaluation framework that is usually applied. This paper challenges the prevailing metrics for assessing bias mitigation techniques, arguing that they do not take into account the changes within-groups and that the resulting prediction labels fall short of reflecting real-world scenarios. We propose a paradigm shift: initially, we should focus on generating the most precise ranking for each subgroup. Following this, individuals should be chosen from these rankings to meet both fairness standards and practical considerations.

{{</citation>}}


### (14/123) Mitigating System Bias in Resource Constrained Asynchronous Federated Learning Systems (Jikun Gao et al., 2024)

{{<citation>}}

Jikun Gao, Ioannis Mavromatis, Peizheng Li, Pietro Carnelli, Aftab Khan. (2024)  
**Mitigating System Bias in Resource Constrained Asynchronous Federated Learning Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.13366v1)  

---


**ABSTRACT**  
Federated learning (FL) systems face performance challenges in dealing with heterogeneous devices and non-identically distributed data across clients. We propose a dynamic global model aggregation method within Asynchronous Federated Learning (AFL) deployments to address these issues. Our aggregation method scores and adjusts the weighting of client model updates based on their upload frequency to accommodate differences in device capabilities. Additionally, we also immediately provide an updated global model to clients after they upload their local models to reduce idle time and improve training efficiency. We evaluate our approach within an AFL deployment consisting of 10 simulated clients with heterogeneous compute constraints and non-IID data. The simulation results, using the FashionMNIST dataset, demonstrate over 10% and 19% improvement in global model accuracy compared to state-of-the-art methods PAPAYA and FedAsync, respectively. Our dynamic aggregation method allows reliable global model training despite limiting client resources and statistical data heterogeneity. This improves robustness and scalability for real-world FL deployments.

{{</citation>}}


### (15/123) Explainable Bayesian Optimization (Tanmay Chakraborty et al., 2024)

{{<citation>}}

Tanmay Chakraborty, Christin Seifert, Christian Wirth. (2024)  
**Explainable Bayesian Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13334v1)  

---


**ABSTRACT**  
In industry, Bayesian optimization (BO) is widely applied in the human-AI collaborative parameter tuning of cyber-physical systems. However, BO's solutions may deviate from human experts' actual goal due to approximation errors and simplified objectives, requiring subsequent tuning. The black-box nature of BO limits the collaborative tuning process because the expert does not trust the BO recommendations. Current explainable AI (XAI) methods are not tailored for optimization and thus fall short of addressing this gap. To bridge this gap, we propose TNTRules (TUNE-NOTUNE Rules), a post-hoc, rule-based explainability method that produces high quality explanations through multiobjective optimization. Our evaluation of benchmark optimization problems and real-world hyperparameter optimization tasks demonstrates TNTRules' superiority over state-of-the-art XAI methods in generating high quality explanations. This work contributes to the intersection of BO and XAI, providing interpretable optimization techniques for real-world applications.

{{</citation>}}


### (16/123) Classification of Radiologically Isolated Syndrome and Clinically Isolated Syndrome with Machine-Learning Techniques (V Mato-Abad et al., 2024)

{{<citation>}}

V Mato-Abad, A Labiano-Fontcuberta, S Rodriguez-Yanez, R Garcia-Vazquez, CR Munteanu, J Andrade-Garda, A Domingo-Santos, V Galan Sanchez-Seco, Y Aladro, ML Martinez-Gines, L Ayuso, J Benito-Leon. (2024)  
**Classification of Radiologically Isolated Syndrome and Clinically Isolated Syndrome with Machine-Learning Techniques**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2401.13301v1)  

---


**ABSTRACT**  
Background and purpose: The unanticipated detection by magnetic resonance imaging (MRI) in the brain of asymptomatic subjects of white matter lesions suggestive of multiple sclerosis (MS) has been named radiologically isolated syndrome (RIS). As the difference between early MS [i.e. clinically isolated syndrome (CIS)] and RIS is the occurrence of a clinical event, it is logical to improve detection of the subclinical form without interfering with MRI as there are radiological diagnostic criteria for that. Our objective was to use machine-learning classification methods to identify morphometric measures that help to discriminate patients with RIS from those with CIS.   Methods: We used a multimodal 3-T MRI approach by combining MRI biomarkers (cortical thickness, cortical and subcortical grey matter volume, and white matter integrity) of a cohort of 17 patients with RIS and 17 patients with CIS for single-subject level classification.   Results: The best proposed models to predict the diagnosis of CIS and RIS were based on the Naive Bayes, Bagging and Multilayer Perceptron classifiers using only three features: the left rostral middle frontal gyrus volume and the fractional anisotropy values in the right amygdala and right lingual gyrus. The Naive Bayes obtained the highest accuracy [overall classification, 0.765; area under the receiver operating characteristic (AUROC), 0.782].   Conclusions: A machine-learning approach applied to multimodal MRI data may differentiate between the earliest clinical expressions of MS (CIS and RIS) with an accuracy of 78%.   Keywords: Bagging; Multilayer Perceptron; Naive Bayes classifier; clinically isolated syndrome; diffusion tensor imaging; machine-learning; magnetic resonance imaging; multiple sclerosis; radiologically isolated syndrome.

{{</citation>}}


### (17/123) Can I trust my fake data -- A comprehensive quality assessment framework for synthetic tabular data in healthcare (Vibeke Binz Vallevik et al., 2024)

{{<citation>}}

Vibeke Binz Vallevik, Aleksandar Babic, Serena Elizabeth Marshall, Severin Elvatun, Helga Brøgger, Sharmini Alagaratnam, Bjørn Edwin, Narasimha Raghavan Veeraragavan, Anne Kjersti Befring, Jan Franz Nygård. (2024)  
**Can I trust my fake data -- A comprehensive quality assessment framework for synthetic tabular data in healthcare**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13716v1)  

---


**ABSTRACT**  
Ensuring safe adoption of AI tools in healthcare hinges on access to sufficient data for training, testing and validation. In response to privacy concerns and regulatory requirements, using synthetic data has been suggested. Synthetic data is created by training a generator on real data to produce a dataset with similar statistical properties. Competing metrics with differing taxonomies for quality evaluation have been suggested, resulting in a complex landscape. Optimising quality entails balancing considerations that make the data fit for use, yet relevant dimensions are left out of existing frameworks. We performed a comprehensive literature review on the use of quality evaluation metrics on SD within the scope of tabular healthcare data and SD made using deep generative methods. Based on this and the collective team experiences, we developed a conceptual framework for quality assurance. The applicability was benchmarked against a practical case from the Dutch National Cancer Registry. We present a conceptual framework for quality assurance of SD for AI applications in healthcare that aligns diverging taxonomies, expands on common quality dimensions to include the dimensions of Fairness and Carbon footprint, and proposes stages necessary to support real-life applications. Building trust in synthetic data by increasing transparency and reducing the safety risk will accelerate the development and uptake of trustworthy AI tools for the benefit of patients. Despite the growing emphasis on algorithmic fairness and carbon footprint, these metrics were scarce in the literature review. The overwhelming focus was on statistical similarity using distance metrics while sequential logic detection was scarce. A consensus-backed framework that includes all relevant quality dimensions can provide assurance for safe and responsible real-life applications of SD.

{{</citation>}}


### (18/123) Discovering Mathematical Formulas from Data via GPT-guided Monte Carlo Tree Search (Yanjie Li et al., 2024)

{{<citation>}}

Yanjie Li, Weijun Li, Lina Yu, Min Wu, Jingyi Liu, Wenqiang Li, Meilan Hao, Shu Wei, Yusong Deng. (2024)  
**Discovering Mathematical Formulas from Data via GPT-guided Monte Carlo Tree Search**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.14424v3)  

---


**ABSTRACT**  
Finding a concise and interpretable mathematical formula that accurately describes the relationship between each variable and the predicted value in the data is a crucial task in scientific research, as well as a significant challenge in artificial intelligence. This problem is referred to as symbolic regression, which is an NP-hard problem. In the previous year, a novel symbolic regression methodology utilizing Monte Carlo Tree Search (MCTS) was advanced, achieving state-of-the-art results on a diverse range of datasets. although this algorithm has shown considerable improvement in recovering target expressions compared to previous methods, the lack of guidance during the MCTS process severely hampers its search efficiency. Recently, some algorithms have added a pre-trained policy network to guide the search of MCTS, but the pre-trained policy network generalizes poorly. To optimize the trade-off between efficiency and versatility, we introduce SR-GPT, a novel algorithm for symbolic regression that integrates Monte Carlo Tree Search (MCTS) with a Generative Pre-Trained Transformer (GPT). By using GPT to guide the MCTS, the search efficiency of MCTS is significantly improved. Next, we utilize the MCTS results to further refine the GPT, enhancing its capabilities and providing more accurate guidance for the MCTS. MCTS and GPT are coupled together and optimize each other until the target expression is successfully determined. We conducted extensive evaluations of SR-GPT using 222 expressions sourced from over 10 different symbolic regression datasets. The experimental results demonstrate that SR-GPT outperforms existing state-of-the-art algorithms in accurately recovering symbolic expressions both with and without added noise.

{{</citation>}}


### (19/123) Adaptive Crowdsourcing Via Self-Supervised Learning (Anmol Kagrecha et al., 2024)

{{<citation>}}

Anmol Kagrecha, Henrik Marklund, Benjamin Van Roy, Hong Jun Jeon, Richard Zeckhauser. (2024)  
**Adaptive Crowdsourcing Via Self-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-HC, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.13239v1)  

---


**ABSTRACT**  
Common crowdsourcing systems average estimates of a latent quantity of interest provided by many crowdworkers to produce a group estimate. We develop a new approach -- just-predict-others -- that leverages self-supervised learning and a novel aggregation scheme. This approach adapts weights assigned to crowdworkers based on estimates they provided for previous quantities. When skills vary across crowdworkers or their estimates correlate, the weighted sum offers a more accurate group estimate than the average. Existing algorithms such as expectation maximization can, at least in principle, produce similarly accurate group estimates. However, their computational requirements become onerous when complex models, such as neural networks, are required to express relationships among crowdworkers. Just-predict-others accommodates such complexity as well as many other practical challenges. We analyze the efficacy of just-predict-others through theoretical and computational studies. Among other things, we establish asymptotic optimality as the number of engagements per crowdworker grows.

{{</citation>}}


### (20/123) On Principled Local Optimization Methods for Federated Learning (Honglin Yuan, 2024)

{{<citation>}}

Honglin Yuan. (2024)  
**On Principled Local Optimization Methods for Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG, math-OC, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13216v1)  

---


**ABSTRACT**  
Federated Learning (FL), a distributed learning paradigm that scales on-device learning collaboratively, has emerged as a promising approach for decentralized AI applications. Local optimization methods such as Federated Averaging (FedAvg) are the most prominent methods for FL applications. Despite their simplicity and popularity, the theoretical understanding of local optimization methods is far from clear. This dissertation aims to advance the theoretical foundation of local methods in the following three directions.   First, we establish sharp bounds for FedAvg, the most popular algorithm in Federated Learning. We demonstrate how FedAvg may suffer from a notion we call iterate bias, and how an additional third-order smoothness assumption may mitigate this effect and lead to better convergence rates. We explain this phenomenon from a Stochastic Differential Equation (SDE) perspective.   Second, we propose Federated Accelerated Stochastic Gradient Descent (FedAc), the first principled acceleration of FedAvg, which provably improves the convergence rate and communication efficiency. Our technique uses on a potential-based perturbed iterate analysis, a novel stability analysis of generalized accelerated SGD, and a strategic tradeoff between acceleration and stability.   Third, we study the Federated Composite Optimization problem, which extends the classic smooth setting by incorporating a shared non-smooth regularizer. We show that direct extensions of FedAvg may suffer from the "curse of primal averaging," resulting in slow convergence. As a solution, we propose a new primal-dual algorithm, Federated Dual Averaging, which overcomes the curse of primal averaging by employing a novel inter-client dual averaging procedure.

{{</citation>}}


### (21/123) Multitask Active Learning for Graph Anomaly Detection (Wenjing Chang et al., 2024)

{{<citation>}}

Wenjing Chang, Kay Liu, Kaize Ding, Philip S. Yu, Jianjun Yu. (2024)  
**Multitask Active Learning for Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Active Learning, Anomaly Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.13210v1)  

---


**ABSTRACT**  
In the web era, graph machine learning has been widely used on ubiquitous graph-structured data. As a pivotal component for bolstering web security and enhancing the robustness of graph-based applications, the significance of graph anomaly detection is continually increasing. While Graph Neural Networks (GNNs) have demonstrated efficacy in supervised and semi-supervised graph anomaly detection, their performance is contingent upon the availability of sufficient ground truth labels. The labor-intensive nature of identifying anomalies from complex graph structures poses a significant challenge in real-world applications. Despite that, the indirect supervision signals from other tasks (e.g., node classification) are relatively abundant. In this paper, we propose a novel MultItask acTIve Graph Anomaly deTEction framework, namely MITIGATE. Firstly, by coupling node classification tasks, MITIGATE obtains the capability to detect out-of-distribution nodes without known anomalies. Secondly, MITIGATE quantifies the informativeness of nodes by the confidence difference across tasks, allowing samples with conflicting predictions to provide informative yet not excessively challenging information for subsequent training. Finally, to enhance the likelihood of selecting representative nodes that are distant from known patterns, MITIGATE adopts a masked aggregation mechanism for distance measurement, considering both inherent features of nodes and current labeled status. Empirical studies on four datasets demonstrate that MITIGATE significantly outperforms the state-of-the-art methods for anomaly detection. Our code is publicly available at: https://github.com/AhaChang/MITIGATE.

{{</citation>}}


### (22/123) Topology-aware Embedding Memory for Learning on Expanding Graphs (Xikun Zhang et al., 2024)

{{<citation>}}

Xikun Zhang, Dongjin Song, Yixin Chen, Dacheng Tao. (2024)  
**Topology-aware Embedding Memory for Learning on Expanding Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.13200v1)  

---


**ABSTRACT**  
Memory replay based techniques have shown great success for continual learning with incrementally accumulated Euclidean data. Directly applying them to continually expanding graphs, however, leads to the potential memory explosion problem due to the need to buffer representative nodes and their associated topological neighborhood structures. To this end, we systematically analyze the key challenges in the memory explosion problem, and present a general framework, i.e., Parameter Decoupled Graph Neural Networks (PDGNNs) with Topology-aware Embedding Memory (TEM), to tackle this issue. The proposed framework not only reduces the memory space complexity from $\mathcal{O}(nd^L)$ to $\mathcal{O}(n)$~\footnote{$n$: memory budget, $d$: average node degree, $L$: the radius of the GNN receptive field}, but also fully utilizes the topological information for memory replay. Specifically, PDGNNs decouple trainable parameters from the computation ego-subgraph via \textit{Topology-aware Embeddings} (TEs), which compress ego-subgraphs into compact vectors (i.e., TEs) to reduce the memory consumption. Based on this framework, we discover a unique \textit{pseudo-training effect} in continual learning on expanding graphs and this effect motivates us to develop a novel \textit{coverage maximization sampling} strategy that can enhance the performance with a tight memory budget. Thorough empirical studies demonstrate that, by tackling the memory explosion problem and incorporating topological information into memory replay, PDGNNs with TEM significantly outperform state-of-the-art techniques, especially in the challenging class-incremental setting.

{{</citation>}}


### (23/123) Compositional Generative Inverse Design (Tailin Wu et al., 2024)

{{<citation>}}

Tailin Wu, Takashi Maruyama, Long Wei, Tao Zhang, Yilun Du, Gianluca Iaccarino, Jure Leskovec. (2024)  
**Compositional Generative Inverse Design**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13171v1)  

---


**ABSTRACT**  
Inverse design, where we seek to design input variables in order to optimize an underlying objective function, is an important problem that arises across fields such as mechanical engineering to aerospace engineering. Inverse design is typically formulated as an optimization problem, with recent works leveraging optimization across learned dynamics models. However, as models are optimized they tend to fall into adversarial modes, preventing effective sampling. We illustrate that by instead optimizing over the learned energy function captured by the diffusion model, we can avoid such adversarial examples and significantly improve design performance. We further illustrate how such a design system is compositional, enabling us to combine multiple different diffusion models representing subcomponents of our desired system to design systems with every specified component. In an N-body interaction task and a challenging 2D multi-airfoil design task, we demonstrate that by composing the learned diffusion model at test time, our method allows us to design initial states and boundary shapes that are more complex than those in the training data. Our method outperforms state-of-the-art neural inverse design method by an average of 41.5% in prediction MAE and 14.3% in design objective for the N-body dataset and discovers formation flying to minimize drag in the multi-airfoil design task. Project website and code can be found at https://github.com/AI4Science-WestlakeU/cindm.

{{</citation>}}


### (24/123) EMP: Effective Multidimensional Persistence for Graph Representation Learning (Ignacio Segovia-Dominguez et al., 2024)

{{<citation>}}

Ignacio Segovia-Dominguez, Yuzhou Chen, Cuneyt G. Akcora, Zhiwei Zhen, Murat Kantarcioglu, Yulia R. Gel, Baris Coskunuzer. (2024)  
**EMP: Effective Multidimensional Persistence for Graph Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CG, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.13713v1)  

---


**ABSTRACT**  
Topological data analysis (TDA) is gaining prominence across a wide spectrum of machine learning tasks that spans from manifold learning to graph classification. A pivotal technique within TDA is persistent homology (PH), which furnishes an exclusive topological imprint of data by tracing the evolution of latent structures as a scale parameter changes. Present PH tools are confined to analyzing data through a single filter parameter. However, many scenarios necessitate the consideration of multiple relevant parameters to attain finer insights into the data. We address this issue by introducing the Effective Multidimensional Persistence (EMP) framework. This framework empowers the exploration of data by simultaneously varying multiple scale parameters. The framework integrates descriptor functions into the analysis process, yielding a highly expressive data summary. It seamlessly integrates established single PH summaries into multidimensional counterparts like EMP Landscapes, Silhouettes, Images, and Surfaces. These summaries represent data's multidimensional aspects as matrices and arrays, aligning effectively with diverse ML models. We provide theoretical guarantees and stability proofs for EMP summaries. We demonstrate EMP's utility in graph classification tasks, showing its effectiveness. Results reveal that EMP enhances various single PH descriptors, outperforming cutting-edge methods on multiple benchmark datasets.

{{</citation>}}


### (25/123) SpacTor-T5: Pre-training T5 Models with Span Corruption and Replaced Token Detection (Ke Ye et al., 2024)

{{<citation>}}

Ke Ye, Heinrich Jiang, Afshin Rostamizadeh, Ayan Chakrabarti, Giulia DeSalvo, Jean-François Kagy, Lazaros Karydas, Gui Citovsky, Sanjiv Kumar. (2024)  
**SpacTor-T5: Pre-training T5 Models with Span Corruption and Replaced Token Detection**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: NLP, T5  
[Paper Link](http://arxiv.org/abs/2401.13160v1)  

---


**ABSTRACT**  
Pre-training large language models is known to be extremely resource intensive and often times inefficient, under-utilizing the information encapsulated in the training text sequences. In this paper, we present SpacTor, a new training procedure consisting of (1) a hybrid objective combining span corruption (SC) and token replacement detection (RTD), and (2) a two-stage curriculum that optimizes the hybrid objective over the initial $\tau$ iterations, then transitions to standard SC loss. We show empirically that the effectiveness of the hybrid objective is tied to the two-stage pre-training schedule, and provide extensive analysis on why this is the case. In our experiments with encoder-decoder architectures (T5) on a variety of NLP tasks, SpacTor-T5 yields the same downstream performance as standard SC pre-training, while enabling a 50% reduction in pre-training iterations and 40% reduction in total FLOPs. Alternatively, given the same amount of computing budget, we find that SpacTor results in significantly improved downstream benchmark performance.

{{</citation>}}


## cs.CV (26)



### (26/123) LAA-Net: Localized Artifact Attention Network for High-Quality Deepfakes Detection (Dat Nguyen et al., 2024)

{{<citation>}}

Dat Nguyen, Nesryne Mejri, Inder Pal Singh, Polina Kuleshova, Marcella Astrid, Anis Kacem, Enjie Ghorbel, Djamila Aouada. (2024)  
**LAA-Net: Localized Artifact Attention Network for High-Quality Deepfakes Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.13856v1)  

---


**ABSTRACT**  
This paper introduces a novel approach for high-quality deepfake detection called Localized Artifact Attention Network (LAA-Net). Existing methods for high-quality deepfake detection are mainly based on a supervised binary classifier coupled with an implicit attention mechanism. As a result, they do not generalize well to unseen manipulations. To handle this issue, two main contributions are made. First, an explicit attention mechanism within a multi-task learning framework is proposed. By combining heatmap-based and self-consistency attention strategies, LAA-Net is forced to focus on a few small artifact-prone vulnerable regions. Second, an Enhanced Feature Pyramid Network (E-FPN) is proposed as a simple and effective mechanism for spreading discriminative low-level features into the final feature output, with the advantage of limiting redundancy. Experiments performed on several benchmarks show the superiority of our approach in terms of Area Under the Curve (AUC) and Average Precision (AP). The code will be released soon.

{{</citation>}}


### (27/123) Democratizing Fine-grained Visual Recognition with Large Language Models (Mingxuan Liu et al., 2024)

{{<citation>}}

Mingxuan Liu, Subhankar Roy, Wenjing Li, Zhun Zhong, Nicu Sebe, Elisa Ricci. (2024)  
**Democratizing Fine-grained Visual Recognition with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.13837v1)  

---


**ABSTRACT**  
Identifying subordinate-level categories from images is a longstanding task in computer vision and is referred to as fine-grained visual recognition (FGVR). It has tremendous significance in real-world applications since an average layperson does not excel at differentiating species of birds or mushrooms due to subtle differences among the species. A major bottleneck in developing FGVR systems is caused by the need of high-quality paired expert annotations. To circumvent the need of expert knowledge we propose Fine-grained Semantic Category Reasoning (FineR) that internally leverages the world knowledge of large language models (LLMs) as a proxy in order to reason about fine-grained category names. In detail, to bridge the modality gap between images and LLM, we extract part-level visual attributes from images as text and feed that information to a LLM. Based on the visual attributes and its internal world knowledge the LLM reasons about the subordinate-level category names. Our training-free FineR outperforms several state-of-the-art FGVR and language and vision assistant models and shows promise in working in the wild and in new domains where gathering expert annotation is arduous.

{{</citation>}}


### (28/123) S2TPVFormer: Spatio-Temporal Tri-Perspective View for temporally coherent 3D Semantic Occupancy Prediction (Sathira Silva et al., 2024)

{{<citation>}}

Sathira Silva, Savindu Bhashitha Wannigama, Roshan Ragel, Gihan Jayatilaka. (2024)  
**S2TPVFormer: Spatio-Temporal Tri-Perspective View for temporally coherent 3D Semantic Occupancy Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.13785v1)  

---


**ABSTRACT**  
Holistic understanding and reasoning in 3D scenes play a vital role in the success of autonomous driving systems. The evolution of 3D semantic occupancy prediction as a pretraining task for autonomous driving and robotic downstream tasks captures finer 3D details compared to methods like 3D detection. Existing approaches predominantly focus on spatial cues, often overlooking temporal cues. Query-based methods tend to converge on computationally intensive Voxel representation for encoding 3D scene information. This study introduces S2TPVFormer, an extension of TPVFormer, utilizing a spatiotemporal transformer architecture for coherent 3D semantic occupancy prediction. Emphasizing the importance of spatiotemporal cues in 3D scene perception, particularly in 3D semantic occupancy prediction, our work explores the less-explored realm of temporal cues. Leveraging Tri-Perspective View (TPV) representation, our spatiotemporal encoder generates temporally rich embeddings, improving prediction coherence while maintaining computational efficiency. To achieve this, we propose a novel Temporal Cross-View Hybrid Attention (TCVHA) mechanism, facilitating effective spatiotemporal information exchange across TPV views. Experimental evaluations on the nuScenes dataset demonstrate a substantial 3.1% improvement in mean Intersection over Union (mIoU) for 3D Semantic Occupancy compared to TPVFormer, confirming the effectiveness of the proposed S2TPVFormer in enhancing 3D scene perception.

{{</citation>}}


### (29/123) How Good is ChatGPT at Face Biometrics? A First Look into Recognition, Soft Biometrics, and Explainability (Ivan DeAndres-Tame et al., 2024)

{{<citation>}}

Ivan DeAndres-Tame, Ruben Tolosana, Ruben Vera-Rodriguez, Aythami Morales, Julian Fierrez, Javier Ortega-Garcia. (2024)  
**How Good is ChatGPT at Face Biometrics? A First Look into Recognition, Soft Biometrics, and Explainability**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.CV  
Keywords: AI, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13641v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) such as GPT developed by OpenAI, have already shown astonishing results, introducing quick changes in our society. This has been intensified by the release of ChatGPT which allows anyone to interact in a simple conversational way with LLMs, without any experience in the field needed. As a result, ChatGPT has been rapidly applied to many different tasks such as code- and song-writer, education, virtual assistants, etc., showing impressive results for tasks for which it was not trained (zero-shot learning).   The present study aims to explore the ability of ChatGPT, based on the recent GPT-4 multimodal LLM, for the task of face biometrics. In particular, we analyze the ability of ChatGPT to perform tasks such as face verification, soft-biometrics estimation, and explainability of the results. ChatGPT could be very valuable to further increase the explainability and transparency of the automatic decisions in human scenarios. Experiments are carried out in order to evaluate the performance and robustness of ChatGPT, using popular public benchmarks and comparing the results with state-of-the-art methods in the field. The results achieved in this study show the potential of LLMs such as ChatGPT for face biometrics, especially to enhance explainability. For reproducibility reasons, we release all the code in GitHub.

{{</citation>}}


### (30/123) SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation (Zhaohu Xing et al., 2024)

{{<citation>}}

Zhaohu Xing, Tian Ye, Yijun Yang, Guang Liu, Lei Zhu. (2024)  
**SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13560v2)  

---


**ABSTRACT**  
The Transformer architecture has shown a remarkable ability in modeling global relationships. However, it poses a significant computational challenge when processing high-dimensional medical images. This hinders its development and widespread adoption in this task. Mamba, as a State Space Model (SSM), recently emerged as a notable manner for long-range dependencies in sequential modeling, excelling in natural language processing filed with its remarkable memory efficiency and computational speed. Inspired by its success, we introduce SegMamba, a novel 3D medical image \textbf{Seg}mentation \textbf{Mamba} model, designed to effectively capture long-range dependencies within whole volume features at every scale. Our SegMamba, in contrast to Transformer-based methods, excels in whole volume feature modeling from a state space model standpoint, maintaining superior processing speed, even with volume features at a resolution of {$64\times 64\times 64$}. Comprehensive experiments on the BraTS2023 dataset demonstrate the effectiveness and efficiency of our SegMamba. The code for SegMamba is available at: https://github.com/ge-xing/SegMamba

{{</citation>}}


### (31/123) PanAf20K: A Large Video Dataset for Wild Ape Detection and Behaviour Recognition (Otto Brookes et al., 2024)

{{<citation>}}

Otto Brookes, Majid Mirmehdi, Colleen Stephens, Samuel Angedakin, Katherine Corogenes, Dervla Dowd, Paula Dieguez, Thurston C. Hicks, Sorrel Jones, Kevin Lee, Vera Leinert, Juan Lapuente, Maureen S. McCarthy, Amelia Meier, Mizuki Murai, Emmanuelle Normand, Virginie Vergnes, Erin G. Wessling, Roman M. Wittig, Kevin Langergraber, Nuria Maldonado, Xinyu Yang, Klaus Zuberbuhler, Christophe Boesch, Mimi Arandjelovic, Hjalmar Kuhl, Tilo Burghardt. (2024)  
**PanAf20K: A Large Video Dataset for Wild Ape Detection and Behaviour Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13554v1)  

---


**ABSTRACT**  
We present the PanAf20K dataset, the largest and most diverse open-access annotated video dataset of great apes in their natural environment. It comprises more than 7 million frames across ~20,000 camera trap videos of chimpanzees and gorillas collected at 18 field sites in tropical Africa as part of the Pan African Programme: The Cultured Chimpanzee. The footage is accompanied by a rich set of annotations and benchmarks making it suitable for training and testing a variety of challenging and ecologically important computer vision tasks including ape detection and behaviour recognition. Furthering AI analysis of camera trap information is critical given the International Union for Conservation of Nature now lists all species in the great ape family as either Endangered or Critically Endangered. We hope the dataset can form a solid basis for engagement of the AI community to improve performance, efficiency, and result interpretation in order to support assessments of great ape presence, abundance, distribution, and behaviour and thereby aid conservation efforts.

{{</citation>}}


### (32/123) Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection (Yongwei Nie et al., 2024)

{{<citation>}}

Yongwei Nie, Hao Huang, Chengjiang Long, Qing Zhang, Pradipta Maji, Hongmin Cai. (2024)  
**Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2401.13551v1)  

---


**ABSTRACT**  
Without human annotations, a typical Unsupervised Video Anomaly Detection (UVAD) method needs to train two models that generate pseudo labels for each other. In previous work, the two models are closely entangled with each other, and it is not known how to upgrade their method without modifying their training framework significantly. Second, previous work usually adopts fixed thresholding to obtain pseudo labels, however the user-specified threshold is not reliable which inevitably introduces errors into the training process. To alleviate these two problems, we propose a novel interleaved framework that alternately trains a One-Class Classification (OCC) model and a Weakly-Supervised (WS) model for UVAD. The OCC or WS models in our method can be easily replaced with other OCC or WS models, which facilitates our method to upgrade with the most recent developments in both fields. For handling the fixed thresholding problem, we break through the conventional cognitive boundary and propose a weighted OCC model that can be trained on both normal and abnormal data. We also propose an adaptive mechanism for automatically finding the optimal threshold for the WS model in a loose to strict manner. Experiments demonstrate that the proposed UVAD method outperforms previous approaches.

{{</citation>}}


### (33/123) QAGait: Revisit Gait Recognition from a Quality Perspective (Zengbin Wang et al., 2024)

{{<citation>}}

Zengbin Wang, Saihui Hou, Man Zhang, Xu Liu, Chunshui Cao, Yongzhen Huang, Peipei Li, Shibiao Xu. (2024)  
**QAGait: Revisit Gait Recognition from a Quality Perspective**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.13531v1)  

---


**ABSTRACT**  
Gait recognition is a promising biometric method that aims to identify pedestrians from their unique walking patterns. Silhouette modality, renowned for its easy acquisition, simple structure, sparse representation, and convenient modeling, has been widely employed in controlled in-the-lab research. However, as gait recognition rapidly advances from in-the-lab to in-the-wild scenarios, various conditions raise significant challenges for silhouette modality, including 1) unidentifiable low-quality silhouettes (abnormal segmentation, severe occlusion, or even non-human shape), and 2) identifiable but challenging silhouettes (background noise, non-standard posture, slight occlusion). To address these challenges, we revisit gait recognition pipeline and approach gait recognition from a quality perspective, namely QAGait. Specifically, we propose a series of cost-effective quality assessment strategies, including Maxmial Connect Area and Template Match to eliminate background noises and unidentifiable silhouettes, Alignment strategy to handle non-standard postures. We also propose two quality-aware loss functions to integrate silhouette quality into optimization within the embedding space. Extensive experiments demonstrate our QAGait can guarantee both gait reliability and performance enhancement. Furthermore, our quality assessment strategies can seamlessly integrate with existing gait datasets, showcasing our superiority. Code is available at https://github.com/wzb-bupt/QAGait.

{{</citation>}}


### (34/123) Research about the Ability of LLM in the Tamper-Detection Area (Xinyu Yang et al., 2024)

{{<citation>}}

Xinyu Yang, Jizhe Zhou. (2024)  
**Research about the Ability of LLM in the Tamper-Detection Area**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13504v1)  

---


**ABSTRACT**  
In recent years, particularly since the early 2020s, Large Language Models (LLMs) have emerged as the most powerful AI tools in addressing a diverse range of challenges, from natural language processing to complex problem-solving in various domains. In the field of tamper detection, LLMs are capable of identifying basic tampering activities.To assess the capabilities of LLMs in more specialized domains, we have collected five different LLMs developed by various companies: GPT-4, LLaMA, Bard, ERNIE Bot 4.0, and Tongyi Qianwen. This diverse range of models allows for a comprehensive evaluation of their performance in detecting sophisticated tampering instances.We devised two domains of detection: AI-Generated Content (AIGC) detection and manipulation detection. AIGC detection aims to test the ability to distinguish whether an image is real or AI-generated. Manipulation detection, on the other hand, focuses on identifying tampered images. According to our experiments, most LLMs can identify composite pictures that are inconsistent with logic, and only more powerful LLMs can distinguish logical, but visible signs of tampering to the human eye. All of the LLMs can't identify carefully forged images and very realistic images generated by AI. In the area of tamper detection, LLMs still have a long way to go, particularly in reliably identifying highly sophisticated forgeries and AI-generated images that closely mimic reality.

{{</citation>}}


### (35/123) Learning Representations for Clustering via Partial Information Discrimination and Cross-Level Interaction (Hai-Xin Zhang et al., 2024)

{{<citation>}}

Hai-Xin Zhang, Dong Huang, Hua-Bao Ling, Guang-Yu Zhang, Wei-jun Sun, Zi-hao Wen. (2024)  
**Learning Representations for Clustering via Partial Information Discrimination and Cross-Level Interaction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13503v1)  

---


**ABSTRACT**  
In this paper, we present a novel deep image clustering approach termed PICI, which enforces the partial information discrimination and the cross-level interaction in a joint learning framework. In particular, we leverage a Transformer encoder as the backbone, through which the masked image modeling with two paralleled augmented views is formulated. After deriving the class tokens from the masked images by the Transformer encoder, three partial information learning modules are further incorporated, including the PISD module for training the auto-encoder via masked image reconstruction, the PICD module for employing two levels of contrastive learning, and the CLI module for mutual interaction between the instance-level and cluster-level subspaces. Extensive experiments have been conducted on six real-world image datasets, which demononstrate the superior clustering performance of the proposed PICI approach over the state-of-the-art deep clustering approaches. The source code is available at https://github.com/Regan-Zhang/PICI.

{{</citation>}}


### (36/123) LDCA: Local Descriptors with Contextual Augmentation for Few-Shot Learning (Maofa Wang et al., 2024)

{{<citation>}}

Maofa Wang, Bingchen Yan. (2024)  
**LDCA: Local Descriptors with Contextual Augmentation for Few-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.13499v1)  

---


**ABSTRACT**  
Few-shot image classification has emerged as a key challenge in the field of computer vision, highlighting the capability to rapidly adapt to new tasks with minimal labeled data. Existing methods predominantly rely on image-level features or local descriptors, often overlooking the holistic context surrounding these descriptors. In this work, we introduce a novel approach termed "Local Descriptor with Contextual Augmentation (LDCA)". Specifically, this method bridges the gap between local and global understanding uniquely by leveraging an adaptive global contextual enhancement module. This module incorporates a visual transformer, endowing local descriptors with contextual awareness capabilities, ranging from broad global perspectives to intricate surrounding nuances. By doing so, LDCA transcends traditional descriptor-based approaches, ensuring each local feature is interpreted within its larger visual narrative. Extensive experiments underscore the efficacy of our method, showing a maximal absolute improvement of 20\% over the next-best on fine-grained classification datasets, thus demonstrating significant advancements in few-shot classification tasks.

{{</citation>}}


### (37/123) Semi-Supervised Coupled Thin-Plate Spline Model for Rotation Correction and Beyond (Lang Nie et al., 2024)

{{<citation>}}

Lang Nie, Chunyu Lin, Kang Liao, Shuaicheng Liu, Yao Zhao. (2024)  
**Semi-Supervised Coupled Thin-Plate Spline Model for Rotation Correction and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.13432v1)  

---


**ABSTRACT**  
Thin-plate spline (TPS) is a principal warp that allows for representing elastic, nonlinear transformation with control point motions. With the increase of control points, the warp becomes increasingly flexible but usually encounters a bottleneck caused by undesired issues, e.g., content distortion. In this paper, we explore generic applications of TPS in single-image-based warping tasks, such as rotation correction, rectangling, and portrait correction. To break this bottleneck, we propose the coupled thin-plate spline model (CoupledTPS), which iteratively couples multiple TPS with limited control points into a more flexible and powerful transformation. Concretely, we first design an iterative search to predict new control points according to the current latent condition. Then, we present the warping flow as a bridge for the coupling of different TPS transformations, effectively eliminating interpolation errors caused by multiple warps. Besides, in light of the laborious annotation cost, we develop a semi-supervised learning scheme to improve warping quality by exploiting unlabeled data. It is formulated through dual transformation between the searched control points of unlabeled data and its graphic augmentation, yielding an implicit correction consistency constraint. Finally, we collect massive unlabeled data to exhibit the benefit of our semi-supervised scheme in rotation correction. Extensive experiments demonstrate the superiority and universality of CoupledTPS over the existing state-of-the-art (SoTA) solutions for rotation correction and beyond. The code and data will be available at https://github.com/nie-lang/CoupledTPS.

{{</citation>}}


### (38/123) UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion (Wei Li et al., 2024)

{{<citation>}}

Wei Li, Xue Xu, Jiachen Liu, Xinyan Xiao. (2024)  
**UNIMO-G: Unified Image Generation through Multimodal Conditional Diffusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13388v2)  

---


**ABSTRACT**  
Existing text-to-image diffusion models primarily generate images from text prompts. However, the inherent conciseness of textual descriptions poses challenges in faithfully synthesizing images with intricate details, such as specific entities or scenes. This paper presents UNIMO-G, a simple multimodal conditional diffusion framework that operates on multimodal prompts with interleaved textual and visual inputs, which demonstrates a unified ability for both text-driven and subject-driven image generation. UNIMO-G comprises two core components: a Multimodal Large Language Model (MLLM) for encoding multimodal prompts, and a conditional denoising diffusion network for generating images based on the encoded multimodal input. We leverage a two-stage training strategy to effectively train the framework: firstly pre-training on large-scale text-image pairs to develop conditional image generation capabilities, and then instruction tuning with multimodal prompts to achieve unified image generation proficiency. A well-designed data processing pipeline involving language grounding and image segmentation is employed to construct multi-modal prompts. UNIMO-G excels in both text-to-image generation and zero-shot subject-driven synthesis, and is notably effective in generating high-fidelity images from complex multimodal prompts involving multiple image entities.

{{</citation>}}


### (39/123) Do You Guys Want to Dance: Zero-Shot Compositional Human Dance Generation with Multiple Persons (Zhe Xu et al., 2024)

{{<citation>}}

Zhe Xu, Kun Wei, Xu Yang, Cheng Deng. (2024)  
**Do You Guys Want to Dance: Zero-Shot Compositional Human Dance Generation with Multiple Persons**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.13363v1)  

---


**ABSTRACT**  
Human dance generation (HDG) aims to synthesize realistic videos from images and sequences of driving poses. Despite great success, existing methods are limited to generating videos of a single person with specific backgrounds, while the generalizability for real-world scenarios with multiple persons and complex backgrounds remains unclear. To systematically measure the generalizability of HDG models, we introduce a new task, dataset, and evaluation protocol of compositional human dance generation (cHDG). Evaluating the state-of-the-art methods on cHDG, we empirically find that they fail to generalize to real-world scenarios. To tackle the issue, we propose a novel zero-shot framework, dubbed MultiDance-Zero, that can synthesize videos consistent with arbitrary multiple persons and background while precisely following the driving poses. Specifically, in contrast to straightforward DDIM or null-text inversion, we first present a pose-aware inversion method to obtain the noisy latent code and initialization text embeddings, which can accurately reconstruct the composed reference image. Since directly generating videos from them will lead to severe appearance inconsistency, we propose a compositional augmentation strategy to generate augmented images and utilize them to optimize a set of generalizable text embeddings. In addition, consistency-guided sampling is elaborated to encourage the background and keypoints of the estimated clean image at each reverse step to be close to those of the reference image, further improving the temporal consistency of generated videos. Extensive qualitative and quantitative results demonstrate the effectiveness and superiority of our approach.

{{</citation>}}


### (40/123) InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions (Ryota Tanaka et al., 2024)

{{<citation>}}

Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko Saito, Jun Suzuki. (2024)  
**InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.13313v1)  

---


**ABSTRACT**  
We study the problem of completing various visual document understanding (VDU) tasks, e.g., question answering and information extraction, on real-world documents through human-written instructions. To this end, we propose InstructDoc, the first large-scale collection of 30 publicly available VDU datasets, each with diverse instructions in a unified format, which covers a wide range of 12 tasks and includes open document types/formats. Furthermore, to enhance the generalization performance on VDU tasks, we design a new instruction-based document reading and understanding model, InstructDr, that connects document images, image encoders, and large language models (LLMs) through a trainable bridging module. Experiments demonstrate that InstructDr can effectively adapt to new VDU datasets, tasks, and domains via given instructions and outperforms existing multimodal LLMs and ChatGPT without specific training.

{{</citation>}}


### (41/123) ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models (Rohan Wadhawan et al., 2024)

{{<citation>}}

Rohan Wadhawan, Hritik Bansal, Kai-Wei Chang, Nanyun Peng. (2024)  
**ConTextual: Evaluating Context-Sensitive Text-Rich Visual Reasoning in Large Multimodal Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, GPT, GPT-4, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.13311v1)  

---


**ABSTRACT**  
Recent advancements in AI have led to the development of large multimodal models (LMMs) capable of processing complex tasks involving joint reasoning over text and visual content in the image (e.g., navigating maps in public places). This paper introduces ConTextual, a novel benchmark comprising instructions designed explicitly to evaluate LMMs' ability to perform context-sensitive text-rich visual reasoning. ConTextual emphasizes diverse real-world scenarios (e.g., time-reading, navigation, shopping and more) demanding a deeper understanding of the interactions between textual and visual elements. Our findings reveal a significant performance gap of 30.8% between the best-performing LMM, GPT-4V(ision), and human capabilities using human evaluation indicating substantial room for improvement in context-sensitive text-rich visual reasoning. Notably, while GPT-4V excelled in abstract categories like meme and quote interpretation, its overall performance still lagged behind humans. In addition to human evaluations, we also employed automatic evaluation metrics using GPT-4, uncovering similar trends in performance disparities. We also perform a fine-grained evaluation across diverse visual contexts and provide qualitative analysis which provides a robust framework for future advancements in the LMM design. https://con-textual.github.io/

{{</citation>}}


### (42/123) Visual Objectification in Films: Towards a New AI Task for Video Interpretation (Julie Tores et al., 2024)

{{<citation>}}

Julie Tores, Lucile Sassatelli, Hui-Yin Wu, Clement Bergman, Lea Andolfi, Victor Ecrement, Frederic Precioso, Thierry Devars, Magali Guaresi, Virginie Julliard, Sarah Lecossais. (2024)  
**Visual Objectification in Films: Towards a New AI Task for Video Interpretation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13296v1)  

---


**ABSTRACT**  
In film gender studies, the concept of 'male gaze' refers to the way the characters are portrayed on-screen as objects of desire rather than subjects. In this article, we introduce a novel video-interpretation task, to detect character objectification in films. The purpose is to reveal and quantify the usage of complex temporal patterns operated in cinema to produce the cognitive perception of objectification. We introduce the ObyGaze12 dataset, made of 1914 movie clips densely annotated by experts for objectification concepts identified in film studies and psychology. We evaluate recent vision models, show the feasibility of the task and where the challenges remain with concept bottleneck models. Our new dataset and code are made available to the community.

{{</citation>}}


### (43/123) DDI-CoCo: A Dataset For Understanding The Effect Of Color Contrast In Machine-Assisted Skin Disease Detection (Ming-Chang Chiu et al., 2024)

{{<citation>}}

Ming-Chang Chiu, Yingfei Wang, Yen-Ju Kuo, Pin-Yu Chen. (2024)  
**DDI-CoCo: A Dataset For Understanding The Effect Of Color Contrast In Machine-Assisted Skin Disease Detection**  

---
Primary Category: cs.CV  
Categories: cs-CE, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13280v1)  

---


**ABSTRACT**  
Skin tone as a demographic bias and inconsistent human labeling poses challenges in dermatology AI. We take another angle to investigate color contrast's impact, beyond skin tones, on malignancy detection in skin disease datasets: We hypothesize that in addition to skin tones, the color difference between the lesion area and skin also plays a role in malignancy detection performance of dermatology AI models. To study this, we first propose a robust labeling method to quantify color contrast scores of each image and validate our method by showing small labeling variations. More importantly, applying our method to \textit{the only} diverse-skin tone and pathologically-confirmed skin disease dataset DDI, yields \textbf{DDI-CoCo Dataset}, and we observe a performance gap between the high and low color difference groups. This disparity remains consistent across various state-of-the-art (SoTA) image classification models, which supports our hypothesis. Furthermore, we study the interaction between skin tone and color difference effects and suggest that color difference can be an additional reason behind model performance bias between skin tones. Our work provides a complementary angle to dermatology AI for improving skin disease detection.

{{</citation>}}


### (44/123) Audio-Infused Automatic Image Colorization by Exploiting Audio Scene Semantics (Pengcheng Zhao et al., 2024)

{{<citation>}}

Pengcheng Zhao, Yanxiang Chen, Yang Zhao, Wei Jia, Zhao Zhang, Ronggang Wang, Richang Hong. (2024)  
**Audio-Infused Automatic Image Colorization by Exploiting Audio Scene Semantics**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13270v1)  

---


**ABSTRACT**  
Automatic image colorization is inherently an ill-posed problem with uncertainty, which requires an accurate semantic understanding of scenes to estimate reasonable colors for grayscale images. Although recent interaction-based methods have achieved impressive performance, it is still a very difficult task to infer realistic and accurate colors for automatic colorization. To reduce the difficulty of semantic understanding of grayscale scenes, this paper tries to utilize corresponding audio, which naturally contains extra semantic information about the same scene. Specifically, a novel audio-infused automatic image colorization (AIAIC) network is proposed, which consists of three stages. First, we take color image semantics as a bridge and pretrain a colorization network guided by color image semantics. Second, the natural co-occurrence of audio and video is utilized to learn the color semantic correlations between audio and visual scenes. Third, the implicit audio semantic representation is fed into the pretrained network to finally realize the audio-guided colorization. The whole process is trained in a self-supervised manner without human annotation. In addition, an audiovisual colorization dataset is established for training and testing. Experiments demonstrate that audio guidance can effectively improve the performance of automatic colorization, especially for some scenes that are difficult to understand only from visual modality.

{{</citation>}}


### (45/123) Value-Driven Mixed-Precision Quantization for Patch-Based Inference on Microcontrollers (Wei Tao et al., 2024)

{{<citation>}}

Wei Tao, Shenglin He, Kai Lu, Xiaoyang Qu, Guokuan Li, Jiguang Wan, Jianzong Wang, Jing Xiao. (2024)  
**Value-Driven Mixed-Precision Quantization for Patch-Based Inference on Microcontrollers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2401.13714v1)  

---


**ABSTRACT**  
Deploying neural networks on microcontroller units (MCUs) presents substantial challenges due to their constrained computation and memory resources. Previous researches have explored patch-based inference as a strategy to conserve memory without sacrificing model accuracy. However, this technique suffers from severe redundant computation overhead, leading to a substantial increase in execution latency. A feasible solution to address this issue is mixed-precision quantization, but it faces the challenges of accuracy degradation and a time-consuming search time. In this paper, we propose QuantMCU, a novel patch-based inference method that utilizes value-driven mixed-precision quantization to reduce redundant computation. We first utilize value-driven patch classification (VDPC) to maintain the model accuracy. VDPC classifies patches into two classes based on whether they contain outlier values. For patches containing outlier values, we apply 8-bit quantization to the feature maps on the dataflow branches that follow. In addition, for patches without outlier values, we utilize value-driven quantization search (VDQS) on the feature maps of their following dataflow branches to reduce search time. Specifically, VDQS introduces a novel quantization search metric that takes into account both computation and accuracy, and it employs entropy as an accuracy representation to avoid additional training. VDQS also adopts an iterative approach to determine the bitwidth of each feature map to further accelerate the search process. Experimental results on real-world MCU devices show that QuantMCU can reduce computation by 2.2x on average while maintaining comparable model accuracy compared to the state-of-the-art patch-based inference methods.

{{</citation>}}


### (46/123) AMANet: Advancing SAR Ship Detection with Adaptive Multi-Hierarchical Attention Network (Xiaolin Ma et al., 2024)

{{<citation>}}

Xiaolin Ma, Junkai Cheng, Aihua Li, Yuhua Zhang, Zhilong Lin. (2024)  
**AMANet: Advancing SAR Ship Detection with Adaptive Multi-Hierarchical Attention Network**  

---
Primary Category: cs.CV  
Categories: 68T45, I-2-10, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.13214v1)  

---


**ABSTRACT**  
Recently, methods based on deep learning have been successfully applied to ship detection for synthetic aperture radar (SAR) images. Despite the development of numerous ship detection methodologies, detecting small and coastal ships remains a significant challenge due to the limited features and clutter in coastal environments. For that, a novel adaptive multi-hierarchical attention module (AMAM) is proposed to learn multi-scale features and adaptively aggregate salient features from various feature layers, even in complex environments. Specifically, we first fuse information from adjacent feature layers to enhance the detection of smaller targets, thereby achieving multi-scale feature enhancement. Then, to filter out the adverse effects of complex backgrounds, we dissect the previously fused multi-level features on the channel, individually excavate the salient regions, and adaptively amalgamate features originating from different channels. Thirdly, we present a novel adaptive multi-hierarchical attention network (AMANet) by embedding the AMAM between the backbone network and the feature pyramid network (FPN). Besides, the AMAM can be readily inserted between different frameworks to improve object detection. Lastly, extensive experiments on two large-scale SAR ship detection datasets demonstrate that our AMANet method is superior to state-of-the-art methods.

{{</citation>}}


### (47/123) Common-Sense Bias Discovery and Mitigation for Classification Tasks (Miao Zhang et al., 2024)

{{<citation>}}

Miao Zhang, Zee fryer, Ben Colman, Ali Shahriyari, Gaurav Bharaj. (2024)  
**Common-Sense Bias Discovery and Mitigation for Classification Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2401.13213v1)  

---


**ABSTRACT**  
Machine learning model bias can arise from dataset composition: sensitive features correlated to the learning target disturb the model decision rule and lead to performance differences along the features. Existing de-biasing work captures prominent and delicate image features which are traceable in model latent space, like colors of digits or background of animals. However, using the latent space is not sufficient to understand all dataset feature correlations. In this work, we propose a framework to extract feature clusters in a dataset based on image descriptions, allowing us to capture both subtle and coarse features of the images. The feature co-occurrence pattern is formulated and correlation is measured, utilizing a human-in-the-loop for examination. The analyzed features and correlations are human-interpretable, so we name the method Common-Sense Bias Discovery (CSBD). Having exposed sensitive correlations in a dataset, we demonstrate that downstream model bias can be mitigated by adjusting image sampling weights, without requiring a sensitive group label supervision. Experiments show that our method discovers novel biases on multiple classification tasks for two benchmark image datasets, and the intervention outperforms state-of-the-art unsupervised bias mitigation methods.

{{</citation>}}


### (48/123) Boosting the Transferability of Adversarial Examples via Local Mixup and Adaptive Step Size (Junlin Liu et al., 2024)

{{<citation>}}

Junlin Liu, Xinchen Lyu. (2024)  
**Boosting the Transferability of Adversarial Examples via Local Mixup and Adaptive Step Size**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.13205v1)  

---


**ABSTRACT**  
Adversarial examples are one critical security threat to various visual applications, where injected human-imperceptible perturbations can confuse the output.Generating transferable adversarial examples in the black-box setting is crucial but challenging in practice. Existing input-diversity-based methods adopt different image transformations, but may be inefficient due to insufficient input diversity and an identical perturbation step size. Motivated by the fact that different image regions have distinctive weights in classification, this paper proposes a black-box adversarial generative framework by jointly designing enhanced input diversity and adaptive step sizes. We design local mixup to randomly mix a group of transformed adversarial images, strengthening the input diversity. For precise adversarial generation, we project the perturbation into the $tanh$ space to relax the boundary constraint. Moreover, the step sizes of different regions can be dynamically adjusted by integrating a second-order momentum.Extensive experiments on ImageNet validate that our framework can achieve superior transferability compared to state-of-the-art baselines.

{{</citation>}}


### (49/123) MLLMReID: Multimodal Large Language Model-based Person Re-identification (Shan Yang et al., 2024)

{{<citation>}}

Shan Yang, Yongfei Zhang. (2024)  
**MLLMReID: Multimodal Large Language Model-based Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13201v1)  

---


**ABSTRACT**  
Multimodal large language models (MLLM) have achieved satisfactory results in many tasks. However, their performance in the task of person re-identification (ReID) has not been explored to date. This paper will investigate how to adapt them for the task of ReID. An intuitive idea is to fine-tune MLLM with ReID image-text datasets, and then use their visual encoder as a backbone for ReID. However, there still exist two apparent issues: (1) Designing instructions for ReID, MLLMs may overfit specific instructions, and designing a variety of instructions will lead to higher costs. (2) Latent image feature vectors from LLMs are not involved in loss computation. Instructional learning, aligning image-text features, results in indirect optimization and a learning objective that inadequately utilizes features, limiting effectiveness in person feature learning. To address these problems, this paper proposes MLLMReID: Multimodal Large Language Model-based ReID. Firstly, we proposed Common Instruction, a simple approach that leverages the essence ability of LLMs to continue writing, avoiding complex and diverse instruction design. Secondly, we proposed DirectReID, which effectively employs the latent image feature vectors of images outputted by LLMs in ReID tasks. The experimental results demonstrate the superiority of our method. We will open-source the code on GitHub.

{{</citation>}}


### (50/123) Boundary and Relation Distillation for Semantic Segmentation (Dong Zhang et al., 2024)

{{<citation>}}

Dong Zhang, Pingcheng Dong, Xinting Hu, Long Chen, Kwang-Ting Cheng. (2024)  
**Boundary and Relation Distillation for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.13174v1)  

---


**ABSTRACT**  
Recently, it has been revealed that small semantic segmentation (SS) models exhibit a tendency to make errors in maintaining boundary region completeness and preserving target region connectivity, despite their effective segmentation of the main object regions. To address these errors, we propose a targeted boundary and relation distillation (BRD) strategy using knowledge distillation from large teacher models to small student models. Specifically, the boundary distillation extracts explicit object boundaries from the hierarchical feature maps of the backbone network, subsequently enhancing the student model's mask quality in boundary regions. Concurrently, the relation distillation transfers implicit relations from the teacher model to the student model using pixel-level self-relation as a bridge, ensuring that the student's mask has strong target region connectivity. The proposed BRD is designed concretely for SS and is characterized by simplicity and efficiency. Through experimental evaluations on multiple SS datasets, including Pascal VOC 2012, Cityscapes, ADE20K, and COCO-Stuff 10K, we demonstrated that BRD significantly surpasses the current methods without increasing the inference costs, generating crisp region boundaries and smooth connecting regions that are challenging for small models.

{{</citation>}}


### (51/123) ADMap: Anti-disturbance framework for reconstructing online vectorized HD map (Haotian Hu et al., 2024)

{{<citation>}}

Haotian Hu, Fanyi Wang, Yaonong Wang, Laifeng Hu, Jingwei Xu, Zhiwang Zhang. (2024)  
**ADMap: Anti-disturbance framework for reconstructing online vectorized HD map**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.13172v1)  

---


**ABSTRACT**  
In the field of autonomous driving, online high-definition (HD) map reconstruction is crucial for planning tasks. Recent research has developed several high-performance HD map reconstruction models to meet this necessity. However, the point sequences within the instance vectors may be jittery or jagged due to prediction bias, which can impact subsequent tasks. Therefore, this paper proposes the Anti-disturbance Map reconstruction framework (ADMap). To mitigate point-order jitter, the framework consists of three modules: Multi-Scale Perception Neck, Instance Interactive Attention (IIA), and Vector Direction Difference Loss (VDDL). By exploring the point-order relationships between and within instances in a cascading manner, the model can monitor the point-order prediction process more effectively. ADMap achieves state-of-the-art performance on the nuScenes and Argoverse2 datasets. Extensive results demonstrate its ability to produce stable and reliable map elements in complex and changing driving scenarios. Code and more demos are available at https://github.com/hht1996ok/ADMap.

{{</citation>}}


## cs.SD (1)



### (52/123) Scaling NVIDIA's Multi-speaker Multi-lingual TTS Systems with Zero-Shot TTS to Indic Languages (Akshit Arora et al., 2024)

{{<citation>}}

Akshit Arora, Rohan Badlani, Sungwon Kim, Rafael Valle, Bryan Catanzaro. (2024)  
**Scaling NVIDIA's Multi-speaker Multi-lingual TTS Systems with Zero-Shot TTS to Indic Languages**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.13851v2)  

---


**ABSTRACT**  
In this paper, we describe the TTS models developed by NVIDIA for the MMITS-VC (Multi-speaker, Multi-lingual Indic TTS with Voice Cloning) 2024 Challenge. In Tracks 1 and 2, we utilize RAD-MMM to perform few-shot TTS by training additionally on 5 minutes of target speaker data. In Track 3, we utilize P-Flow to perform zero-shot TTS by training on the challenge dataset as well as external datasets. We use HiFi-GAN vocoders for all submissions. RAD-MMM performs competitively on Tracks 1 and 2, while P-Flow ranks first on Track 3, with mean opinion score (MOS) 4.4 and speaker similarity score (SMOS) of 3.62.

{{</citation>}}


## cs.CY (5)



### (53/123) PADTHAI-MM: A Principled Approach for Designing Trustable, Human-centered AI systems using the MAST Methodology (Nayoung Kim et al., 2024)

{{<citation>}}

Nayoung Kim, Myke C. Cohen, Yang Ba, Anna Pan, Shawaiz Bhatti, Pouria Salehi, James Sung, Erik Blasch, Michelle V. Mancenido, Erin K. Chiou. (2024)  
**PADTHAI-MM: A Principled Approach for Designing Trustable, Human-centered AI systems using the MAST Methodology**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13850v1)  

---


**ABSTRACT**  
Designing for AI trustworthiness is challenging, with a lack of practical guidance despite extensive literature on trust. The Multisource AI Scorecard Table (MAST), a checklist rating system, addresses this gap in designing and evaluating AI-enabled decision support systems. We propose the Principled Approach for Designing Trustable Human-centered AI systems using MAST Methodology (PADTHAI-MM), a nine-step framework what we demonstrate through the iterative design of a text analysis platform called the REporting Assistant for Defense and Intelligence Tasks (READIT). We designed two versions of READIT, high-MAST including AI context and explanations, and low-MAST resembling a "black box" type system. Participant feedback and state-of-the-art AI knowledge was integrated in the design process, leading to a redesigned prototype tested by participants in an intelligence reporting task. Results show that MAST-guided design can improve trust perceptions, and that MAST criteria can be linked to performance, process, and purpose information, providing a practical and theory-informed basis for AI system design.

{{</citation>}}


### (54/123) Regulating AI-Based Remote Biometric Identification. Investigating the Public Demand for Bans, Audits, and Public Database Registrations (Kimon Kieslich et al., 2024)

{{<citation>}}

Kimon Kieslich, Marco Lünich. (2024)  
**Regulating AI-Based Remote Biometric Identification. Investigating the Public Demand for Bans, Audits, and Public Database Registrations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13605v1)  

---


**ABSTRACT**  
AI is increasingly being used in the public sector, including public security. In this context, the use of AI-powered remote biometric identification (RBI) systems is a much-discussed technology. RBI systems are used to identify criminal activity in public spaces, but are criticised for inheriting biases and violating fundamental human rights. It is therefore important to ensure that such systems are developed in the public interest, which means that any technology that is deployed for public use needs to be scrutinised. While there is a consensus among business leaders, policymakers and scientists that AI must be developed in an ethical and trustworthy manner, scholars have argued that ethical guidelines do not guarantee ethical AI, but rather prevent stronger regulation of AI. As a possible counterweight, public opinion can have a decisive influence on policymakers to establish boundaries and conditions under which AI systems should be used -- if at all. However, we know little about the conditions that lead to regulatory demand for AI systems. In this study, we focus on the role of trust in AI as well as trust in law enforcement as potential factors that may lead to demands for regulation of AI technology. In addition, we explore the mediating effects of discrimination perceptions regarding RBI. We test the effects on four different use cases of RBI varying the temporal aspect (real-time vs. post hoc analysis) and purpose of use (persecution of criminals vs. safeguarding public events) in a survey among German citizens. We found that German citizens do not differentiate between the different modes of application in terms of their demand for RBI regulation. Furthermore, we show that perceptions of discrimination lead to a demand for stronger regulation, while trust in AI and trust in law enforcement lead to opposite effects in terms of demand for a ban on RBI systems.

{{</citation>}}


### (55/123) How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment (Joshua Ashkinaze et al., 2024)

{{<citation>}}

Joshua Ashkinaze, Julia Mendelsohn, Li Qiwei, Ceren Budak, Eric Gilbert. (2024)  
**How AI Ideas Affect the Creativity, Diversity, and Evolution of Human Ideas: Evidence From a Large, Dynamic Experiment**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.13481v1)  

---


**ABSTRACT**  
Exposure to large language model output is rapidly increasing. How will seeing AI-generated ideas affect human ideas? We conducted an experiment (800+ participants, 40+ countries) where participants viewed creative ideas that were from ChatGPT or prior experimental participants and then brainstormed their own idea. We varied the number of AI-generated examples (none, low, or high exposure) and if the examples were labeled as 'AI' (disclosure). Our dynamic experiment design -- ideas from prior participants in an experimental condition are used as stimuli for future participants in the same experimental condition -- mimics the interdependent process of cultural creation: creative ideas are built upon prior ideas. Hence, we capture the compounding effects of having LLMs 'in the culture loop'. We find that high AI exposure (but not low AI exposure) did not affect the creativity of individual ideas but did increase the average amount and rate of change of collective idea diversity. AI made ideas different, not better. There were no main effects of disclosure. We also found that self-reported creative people were less influenced by knowing an idea was from AI, and that participants were more likely to knowingly adopt AI ideas when the task was difficult. Our findings suggest that introducing AI ideas into society may increase collective diversity but not individual creativity.

{{</citation>}}


### (56/123) 'Here's Your Evidence': False Consensus in Public Twitter Discussions of COVID-19 Science (Alexandros Efstratiou et al., 2024)

{{<citation>}}

Alexandros Efstratiou, Marina Efstratiou, Satrio Yudhoatmojo, Jeremy Blackburn, Emiliano De Cristofaro. (2024)  
**'Here's Your Evidence': False Consensus in Public Twitter Discussions of COVID-19 Science**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.13248v1)  

---


**ABSTRACT**  
The COVID-19 pandemic brought about an extraordinary rate of scientific papers on the topic that were discussed among the general public, although often in biased or misinformed ways. In this paper, we present a mixed-methods analysis aimed at examining whether public discussions were commensurate with the scientific consensus on several COVID-19 issues. We estimate scientific consensus based on samples of abstracts from preprint servers and compare against the volume of public discussions on Twitter mentioning these papers. We find that anti-consensus posts and users, though overall less numerous than pro-consensus ones, are vastly over-represented on Twitter, thus producing a false consensus effect. This transpires with favorable papers being disproportionately amplified, along with an influx of new anti-consensus user sign-ups. Finally, our content analysis highlights that anti-consensus users misrepresent scientific findings or question scientists' integrity in their efforts to substantiate their claims.

{{</citation>}}


### (57/123) Better Call GPT, Comparing Large Language Models Against Lawyers (Lauren Martin et al., 2024)

{{<citation>}}

Lauren Martin, Nick Whitehouse, Stephanie Yiu, Lizzie Catterson, Rivindu Perera. (2024)  
**Better Call GPT, Comparing Large Language Models Against Lawyers**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: GPT, Language Model, Legal  
[Paper Link](http://arxiv.org/abs/2401.16212v1)  

---


**ABSTRACT**  
This paper presents a groundbreaking comparison between Large Language Models and traditional legal contract reviewers, Junior Lawyers and Legal Process Outsourcers. We dissect whether LLMs can outperform humans in accuracy, speed, and cost efficiency during contract review. Our empirical analysis benchmarks LLMs against a ground truth set by Senior Lawyers, uncovering that advanced models match or exceed human accuracy in determining legal issues. In speed, LLMs complete reviews in mere seconds, eclipsing the hours required by their human counterparts. Cost wise, LLMs operate at a fraction of the price, offering a staggering 99.97 percent reduction in cost over traditional methods. These results are not just statistics, they signal a seismic shift in legal practice. LLMs stand poised to disrupt the legal industry, enhancing accessibility and efficiency of legal services. Our research asserts that the era of LLM dominance in legal contract review is upon us, challenging the status quo and calling for a reimagined future of legal workflows.

{{</citation>}}


## cs.CL (27)



### (58/123) TPD: Enhancing Student Language Model Reasoning via Principle Discovery and Guidance (Haorui Wang et al., 2024)

{{<citation>}}

Haorui Wang, Rongzhi Zhang, Yinghao Li, Lingkai Kong, Yuchen Zhuang, Xiusi Chen, Chao Zhang. (2024)  
**TPD: Enhancing Student Language Model Reasoning via Principle Discovery and Guidance**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.13849v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have recently showcased remarkable reasoning abilities. However, larger models often surpass their smaller counterparts in reasoning tasks, posing the challenge of effectively transferring these capabilities from larger models. Existing approaches heavily rely on extensive fine-tuning data or continuous interactions with a superior teacher LLM during inference. We introduce a principle-based teacher-student framework called ``Teaching via Principle Discovery'' (TPD) to address these limitations. Inspired by human learning mechanisms, TPD mimics the interaction between a teacher and a student using a principle-based approach. The teacher LLM generates problem-solving instructions and corrective principles based on the student LLM's errors. These principles guide the refinement of instructions and the selection of instructive examples from a validation set. This enables the student model to learn from both the teacher's guidance and its own mistakes. Once the student model begins making inferences, TPD requires no further intervention from the teacher LLM or humans. Through extensive experiments across eight reasoning tasks, we demonstrate the effectiveness of TPD. Compared to standard chain-of-thought prompting, TPD significantly improves the student model's performance, achieving $6.2\%$ improvement on average.

{{</citation>}}


### (59/123) Automated Root Causing of Cloud Incidents using In-Context Learning with GPT-4 (Xuchao Zhang et al., 2024)

{{<citation>}}

Xuchao Zhang, Supriyo Ghosh, Chetan Bansal, Rujia Wang, Minghua Ma, Yu Kang, Saravan Rajmohan. (2024)  
**Automated Root Causing of Cloud Incidents using In-Context Learning with GPT-4**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SE, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13810v1)  

---


**ABSTRACT**  
Root Cause Analysis (RCA) plays a pivotal role in the incident diagnosis process for cloud services, requiring on-call engineers to identify the primary issues and implement corrective actions to prevent future recurrences. Improving the incident RCA process is vital for minimizing service downtime, customer impact and manual toil. Recent advances in artificial intelligence have introduced state-of-the-art Large Language Models (LLMs) like GPT-4, which have proven effective in tackling various AIOps problems, ranging from code authoring to incident management. Nonetheless, the GPT-4 model's immense size presents challenges when trying to fine-tune it on user data because of the significant GPU resource demand and the necessity for continuous model fine-tuning with the emergence of new data. To address the high cost of fine-tuning LLM, we propose an in-context learning approach for automated root causing, which eliminates the need for fine-tuning. We conduct extensive study over 100,000 production incidents, comparing several large language models using multiple metrics. The results reveal that our in-context learning approach outperforms the previous fine-tuned large language models such as GPT-3 by an average of 24.8\% across all metrics, with an impressive 49.7\% improvement over the zero-shot model. Moreover, human evaluation involving actual incident owners demonstrates its superiority over the fine-tuned model, achieving a 43.5\% improvement in correctness and an 8.7\% enhancement in readability. The impressive results demonstrate the viability of utilizing a vanilla GPT model for the RCA task, thereby avoiding the high computational and maintenance costs associated with a fine-tuned model.

{{</citation>}}


### (60/123) A Unified Approach to Emotion Detection and Task-Oriented Dialogue Modeling (Armand Stricker et al., 2024)

{{<citation>}}

Armand Stricker, Patrick Paroubek. (2024)  
**A Unified Approach to Emotion Detection and Task-Oriented Dialogue Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2401.13789v1)  

---


**ABSTRACT**  
In current text-based task-oriented dialogue (TOD) systems, user emotion detection (ED) is often overlooked or is typically treated as a separate and independent task, requiring additional training. In contrast, our work demonstrates that seamlessly unifying ED and TOD modeling brings about mutual benefits, and is therefore an alternative to be considered. Our method consists in augmenting SimpleToD, an end-to-end TOD system, by extending belief state tracking to include ED, relying on a single language model. We evaluate our approach using GPT-2 and Llama-2 on the EmoWOZ benchmark, a version of MultiWOZ annotated with emotions. Our results reveal a general increase in performance for ED and task results. Our findings also indicate that user emotions provide useful contextual conditioning for system responses, and can be leveraged to further refine responses in terms of empathy.

{{</citation>}}


### (61/123) MambaByte: Token-free Selective State Space Model (Junxiong Wang et al., 2024)

{{<citation>}}

Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, Alexander M Rush. (2024)  
**MambaByte: Token-free Selective State Space Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.13660v1)  

---


**ABSTRACT**  
Token-free language models learn directly from raw bytes and remove the bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences, and standard autoregressive Transformers scale poorly in such settings. We experiment with MambaByte, a token-free adaptation of the Mamba state space model, trained autoregressively on byte sequences. Our experiments indicate the computational efficiency of MambaByte compared to other byte-level models. We also find MambaByte to be competitive with and even outperform state-of-the-art subword Transformers. Furthermore, owing to linear scaling in length, MambaByte benefits from fast inference compared to Transformers. Our findings establish the viability of MambaByte in enabling token-free language modeling.

{{</citation>}}


### (62/123) DenoSent: A Denoising Objective for Self-Supervised Sentence Representation Learning (Xinghao Wang et al., 2024)

{{<citation>}}

Xinghao Wang, Junliang He, Pengyu Wang, Yunhua Zhou, Tianxiang Sun, Xipeng Qiu. (2024)  
**DenoSent: A Denoising Objective for Self-Supervised Sentence Representation Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.13621v1)  

---


**ABSTRACT**  
Contrastive-learning-based methods have dominated sentence representation learning. These methods regularize the representation space by pulling similar sentence representations closer and pushing away the dissimilar ones and have been proven effective in various NLP tasks, e.g., semantic textual similarity (STS) tasks. However, it is challenging for these methods to learn fine-grained semantics as they only learn from the inter-sentence perspective, i.e., their supervision signal comes from the relationship between data samples. In this work, we propose a novel denoising objective that inherits from another perspective, i.e., the intra-sentence perspective. By introducing both discrete and continuous noise, we generate noisy sentences and then train our model to restore them to their original form. Our empirical evaluations demonstrate that this approach delivers competitive results on both semantic textual similarity (STS) and a wide range of transfer tasks, standing up well in comparison to contrastive-learning-based methods. Notably, the proposed intra-sentence denoising objective complements existing inter-sentence contrastive methodologies and can be integrated with them to further enhance performance. Our code is available at https://github.com/xinghaow99/DenoSent.

{{</citation>}}


### (63/123) MM-LLMs: Recent Advances in MultiModal Large Language Models (Duzhen Zhang et al., 2024)

{{<citation>}}

Duzhen Zhang, Yahan Yu, Chenxing Li, Jiahua Dong, Dan Su, Chenhui Chu, Dong Yu. (2024)  
**MM-LLMs: Recent Advances in MultiModal Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13601v2)  

---


**ABSTRACT**  
In the past year, MultiModal Large Language Models (MM-LLMs) have undergone substantial advancements, augmenting off-the-shelf LLMs to support MM inputs or outputs via cost-effective training strategies. The resulting models not only preserve the inherent reasoning and decision-making capabilities of LLMs but also empower a diverse range of MM tasks. In this paper, we provide a comprehensive survey aimed at facilitating further research of MM-LLMs. Specifically, we first outline general design formulations for model architecture and training pipeline. Subsequently, we provide brief introductions of $26$ existing MM-LLMs, each characterized by its specific formulations. Additionally, we review the performance of MM-LLMs on mainstream benchmarks and summarize key training recipes to enhance the potency of MM-LLMs. Lastly, we explore promising directions for MM-LLMs while concurrently maintaining a real-time tracking website for the latest developments in the field. We hope that this survey contributes to the ongoing advancement of the MM-LLMs domain.

{{</citation>}}


### (64/123) Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction (Qi Sun et al., 2024)

{{<citation>}}

Qi Sun, Kun Huang, Xiaocui Yang, Rong Tong, Kun Zhang, Soujanya Poria. (2024)  
**Consistency Guided Knowledge Retrieval and Denoising in LLMs for Zero-shot Document-level Relation Triplet Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13598v1)  

---


**ABSTRACT**  
Document-level Relation Triplet Extraction (DocRTE) is a fundamental task in information systems that aims to simultaneously extract entities with semantic relations from a document. Existing methods heavily rely on a substantial amount of fully labeled data. However, collecting and annotating data for newly emerging relations is time-consuming and labor-intensive. Recent advanced Large Language Models (LLMs), such as ChatGPT and LLaMA, exhibit impressive long-text generation capabilities, inspiring us to explore an alternative approach for obtaining auto-labeled documents with new relations. In this paper, we propose a Zero-shot Document-level Relation Triplet Extraction (ZeroDocRTE) framework, which generates labeled data by retrieval and denoising knowledge from LLMs, called GenRDK. Specifically, we propose a chain-of-retrieval prompt to guide ChatGPT to generate labeled long-text data step by step. To improve the quality of synthetic data, we propose a denoising strategy based on the consistency of cross-document knowledge. Leveraging our denoised synthetic data, we proceed to fine-tune the LLaMA2-13B-Chat for extracting document-level relation triplets. We perform experiments for both zero-shot document-level relation and triplet extraction on two public datasets. The experimental results illustrate that our GenRDK framework outperforms strong baselines.

{{</citation>}}


### (65/123) Graph Guided Question Answer Generation for Procedural Question-Answering (Hai X. Pham et al., 2024)

{{<citation>}}

Hai X. Pham, Isma Hadji, Xinnuo Xu, Ziedune Degutyte, Jay Rainey, Evangelos Kazakos, Afsaneh Fazly, Georgios Tzimiropoulos, Brais Martinez. (2024)  
**Graph Guided Question Answer Generation for Procedural Question-Answering**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, QA  
[Paper Link](http://arxiv.org/abs/2401.13594v1)  

---


**ABSTRACT**  
In this paper, we focus on task-specific question answering (QA). To this end, we introduce a method for generating exhaustive and high-quality training data, which allows us to train compact (e.g., run on a mobile device), task-specific QA models that are competitive against GPT variants. The key technological enabler is a novel mechanism for automatic question-answer generation from procedural text which can ingest large amounts of textual instructions and produce exhaustive in-domain QA training data. While current QA data generation methods can produce well-formed and varied data, their non-exhaustive nature is sub-optimal for training a QA model. In contrast, we leverage the highly structured aspect of procedural text and represent each step and the overall flow of the procedure as graphs. We then condition on graph nodes to automatically generate QA pairs in an exhaustive and controllable manner. Comprehensive evaluations of our method show that: 1) small models trained with our data achieve excellent performance on the target QA task, even exceeding that of GPT3 and ChatGPT despite being several orders of magnitude smaller. 2) semantic coverage is the key indicator for downstream QA performance. Crucially, while large language models excel at syntactic diversity, this does not necessarily result in improvements on the end QA model. In contrast, the higher semantic coverage provided by our method is critical for QA performance.

{{</citation>}}


### (66/123) Evaluation of General Large Language Models in Contextually Assessing Semantic Concepts Extracted from Adult Critical Care Electronic Health Record Notes (Darren Liu et al., 2024)

{{<citation>}}

Darren Liu, Cheng Ding, Delgersuren Bold, Monique Bouvier, Jiaying Lu, Benjamin Shickel, Craig S. Jabaley, Wenhui Zhang, Soojin Park, Michael J. Young, Mark S. Wainwright, Gilles Clermont, Parisa Rashidi, Eric S. Rosenthal, Laurie Dimisko, Ran Xiao, Joo Heung Yoon, Carl Yang, Xiao Hu. (2024)  
**Evaluation of General Large Language Models in Contextually Assessing Semantic Concepts Extracted from Adult Critical Care Electronic Health Record Notes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-SE, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13588v1)  

---


**ABSTRACT**  
The field of healthcare has increasingly turned its focus towards Large Language Models (LLMs) due to their remarkable performance. However, their performance in actual clinical applications has been underexplored. Traditional evaluations based on question-answering tasks don't fully capture the nuanced contexts. This gap highlights the need for more in-depth and practical assessments of LLMs in real-world healthcare settings. Objective: We sought to evaluate the performance of LLMs in the complex clinical context of adult critical care medicine using systematic and comprehensible analytic methods, including clinician annotation and adjudication. Methods: We investigated the performance of three general LLMs in understanding and processing real-world clinical notes. Concepts from 150 clinical notes were identified by MetaMap and then labeled by 9 clinicians. Each LLM's proficiency was evaluated by identifying the temporality and negation of these concepts using different prompts for an in-depth analysis. Results: GPT-4 showed overall superior performance compared to other LLMs. In contrast, both GPT-3.5 and text-davinci-003 exhibit enhanced performance when the appropriate prompting strategies are employed. The GPT family models have demonstrated considerable efficiency, evidenced by their cost-effectiveness and time-saving capabilities. Conclusion: A comprehensive qualitative performance evaluation framework for LLMs is developed and operationalized. This framework goes beyond singular performance aspects. With expert annotations, this methodology not only validates LLMs' capabilities in processing complex medical data but also establishes a benchmark for future LLM evaluations across specialized domains.

{{</citation>}}


### (67/123) Large Malaysian Language Model Based on Mistral for Enhanced Local Language Understanding (Husein Zolkepli et al., 2024)

{{<citation>}}

Husein Zolkepli, Aisyah Razak, Kamarul Adha, Ariff Nazhan. (2024)  
**Large Malaysian Language Model Based on Mistral for Enhanced Local Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13565v2)  

---


**ABSTRACT**  
In this paper, we present significant advancements in the pretraining of Mistral 7B, a large-scale language model, using a dataset of 32.6 GB, equivalent to 1.1 billion tokens. We explore the impact of extending the context length, releasing models with context lengths of 4096 and 32768 tokens, and further refining performance with a specialized 16384 context length instruction-tuned model, we called it Malaysian Mistral.   Our experiments demonstrate the efficacy of continue pretraining and the influence of extended context lengths on Mistral 7B's language understanding capabilities. Additionally, we release a model specifically tuned with a 16384 context length instruction, showcasing its potential for capturing nuanced language intricacies.   Furthermore, our research contributes to the benchmarking of Malaysian Mistral against prominent language models, including ChatGPT3.5 and Claude 2. We present compelling results indicating Malaysian Mistral's superior performance on Tatabahasa (Malay grammar) test set, particularly when fine-tuned with instructions.   All models released at https://huggingface.co/collections/mesolitica/malaysian-mistral-7b-6528f2ec825f4bba46c1700c

{{</citation>}}


### (68/123) SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation (Dong Zhang et al., 2024)

{{<citation>}}

Dong Zhang, Xin Zhang, Jun Zhan, Shimin Li, Yaqian Zhou, Xipeng Qiu. (2024)  
**SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13527v2)  

---


**ABSTRACT**  
Benefiting from effective speech modeling, current Speech Large Language Models (SLLMs) have demonstrated exceptional capabilities in in-context speech generation and efficient generalization to unseen speakers. However, the prevailing information modeling process is encumbered by certain redundancies, leading to inefficiencies in speech generation. We propose Chain-of-Information Generation (CoIG), a method for decoupling semantic and perceptual information in large-scale speech generation. Building on this, we develop SpeechGPT-Gen, an 8-billion-parameter SLLM efficient in semantic and perceptual information modeling. It comprises an autoregressive model based on LLM for semantic information modeling and a non-autoregressive model employing flow matching for perceptual information modeling. Additionally, we introduce the novel approach of infusing semantic information into the prior distribution to enhance the efficiency of flow matching. Extensive experimental results demonstrate that SpeechGPT-Gen markedly excels in zero-shot text-to-speech, zero-shot voice conversion, and speech-to-speech dialogue, underscoring CoIG's remarkable proficiency in capturing and modeling speech's semantic and perceptual dimensions. Code and models are available at https://github.com/0nutation/SpeechGPT.

{{</citation>}}


### (69/123) Can GPT-3.5 Generate and Code Discharge Summaries? (Matúš Falis et al., 2024)

{{<citation>}}

Matúš Falis, Aryo Pradipta Gema, Hang Dong, Luke Daines, Siddharth Basetti, Michael Holder, Rose S Penfold, Alexandra Birch, Beatrice Alex. (2024)  
**Can GPT-3.5 Generate and Code Discharge Summaries?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Clinical, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2401.13512v1)  

---


**ABSTRACT**  
Objective: To investigate GPT-3.5 in generating and coding medical documents with ICD-10 codes for data augmentation on low-resources labels.   Materials and Methods: Employing GPT-3.5 we generated and coded 9,606 discharge summaries based on lists of ICD-10 code descriptions of patients with infrequent (generation) codes within the MIMIC-IV dataset. Combined with the baseline training set, this formed an augmented training set. Neural coding models were trained on baseline and augmented data and evaluated on a MIMIC-IV test set. We report micro- and macro-F1 scores on the full codeset, generation codes, and their families. Weak Hierarchical Confusion Matrices were employed to determine within-family and outside-of-family coding errors in the latter codesets. The coding performance of GPT-3.5 was evaluated both on prompt-guided self-generated data and real MIMIC-IV data. Clinical professionals evaluated the clinical acceptability of the generated documents.   Results: Augmentation slightly hinders the overall performance of the models but improves performance for the generation candidate codes and their families, including one unseen in the baseline training data. Augmented models display lower out-of-family error rates. GPT-3.5 can identify ICD-10 codes by the prompted descriptions, but performs poorly on real data. Evaluators note the correctness of generated concepts while suffering in variety, supporting information, and narrative.   Discussion and Conclusion: GPT-3.5 alone is unsuitable for ICD-10 coding. Augmentation positively affects generation code families but mainly benefits codes with existing examples. Augmentation reduces out-of-family errors. Discharge summaries generated by GPT-3.5 state prompted concepts correctly but lack variety, and authenticity in narratives. They are unsuitable for clinical practice.

{{</citation>}}


### (70/123) SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering (Chyi-Jiunn Lin et al., 2024)

{{<citation>}}

Chyi-Jiunn Lin, Guan-Ting Lin, Yung-Sung Chuang, Wei-Lun Wu, Shang-Wen Li, Abdelrahman Mohamed, Hung-yi Lee, Lin-shan Lee. (2024)  
**SpeechDPR: End-to-End Spoken Passage Retrieval for Open-Domain Spoken Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs-SD, cs.CL, eess-AS  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.13463v1)  

---


**ABSTRACT**  
Spoken Question Answering (SQA) is essential for machines to reply to user's question by finding the answer span within a given spoken passage. SQA has been previously achieved without ASR to avoid recognition errors and Out-of-Vocabulary (OOV) problems. However, the real-world problem of Open-domain SQA (openSQA), in which the machine needs to first retrieve passages that possibly contain the answer from a spoken archive in addition, was never considered. This paper proposes the first known end-to-end framework, Speech Dense Passage Retriever (SpeechDPR), for the retrieval component of the openSQA problem. SpeechDPR learns a sentence-level semantic representation by distilling knowledge from the cascading model of unsupervised ASR (UASR) and text dense retriever (TDR). No manually transcribed speech data is needed. Initial experiments showed performance comparable to the cascading model of UASR and TDR, and significantly better when UASR was poor, verifying this approach is more robust to speech recognition errors.

{{</citation>}}


### (71/123) Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption (Dehao Tao et al., 2024)

{{<citation>}}

Dehao Tao, Feng Huang, Yongfeng Huang, Minghu Jiang. (2024)  
**Clue-Guided Path Exploration: An Efficient Knowledge Base Question-Answering Framework with Low Computational Resource Consumption**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLM, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.13444v1)  

---


**ABSTRACT**  
In recent times, large language models (LLMs) have showcased remarkable capabilities. However, updating their knowledge poses challenges, potentially leading to inaccuracies when confronted with unfamiliar queries. While integrating knowledge graphs with LLMs has been explored, existing approaches treat LLMs as primary decision-makers, imposing high demands on their capabilities. This is particularly unsuitable for LLMs with lower computational costs and relatively poorer performance. In this paper, we introduce a Clue-Guided Path Exploration framework (CGPE) that efficiently merges a knowledge base with an LLM, placing less stringent requirements on the model's capabilities. Inspired by the method humans use to manually retrieve knowledge, CGPE employs information from the question as clues to systematically explore the required knowledge path within the knowledge base. Experiments on open-source datasets reveal that CGPE outperforms previous methods and is highly applicable to LLMs with fewer parameters. In some instances, even ChatGLM3, with its 6 billion parameters, can rival the performance of GPT-4. Furthermore, the results indicate a minimal invocation frequency of CGPE on LLMs, suggesting reduced computational overhead. For organizations and individuals facing constraints in computational resources, our research offers significant practical value.

{{</citation>}}


### (72/123) Text Categorization Can Enhance Domain-Agnostic Stopword Extraction (Houcemeddine Turki et al., 2024)

{{<citation>}}

Houcemeddine Turki, Naome A. Etori, Mohamed Ali Hadj Taieb, Abdul-Hakeem Omotayo, Chris Chinenye Emezue, Mohamed Ben Aouicha, Ayodele Awokoya, Falalu Ibrahim Lawan, Doreen Nixdorf. (2024)  
**Text Categorization Can Enhance Domain-Agnostic Stopword Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2401.13398v1)  

---


**ABSTRACT**  
This paper investigates the role of text categorization in streamlining stopword extraction in natural language processing (NLP), specifically focusing on nine African languages alongside French. By leveraging the MasakhaNEWS, African Stopwords Project, and MasakhaPOS datasets, our findings emphasize that text categorization effectively identifies domain-agnostic stopwords with over 80% detection success rate for most examined languages. Nevertheless, linguistic variances result in lower detection rates for certain languages. Interestingly, we find that while over 40% of stopwords are common across news categories, less than 15% are unique to a single category. Uncommon stopwords add depth to text but their classification as stopwords depends on context. Therefore combining statistical and linguistic approaches creates comprehensive stopword lists, highlighting the value of our hybrid method. This research enhances NLP for African languages and underscores the importance of text categorization in stopword extraction.

{{</citation>}}


### (73/123) MaLA-500: Massive Language Adaptation of Large Language Models (Peiqin Lin et al., 2024)

{{<citation>}}

Peiqin Lin, Shaoxiong Ji, Jörg Tiedemann, André F. T. Martins, Hinrich Schütze. (2024)  
**MaLA-500: Massive Language Adaptation of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13303v1)  

---


**ABSTRACT**  
Large language models have advanced the state of the art in natural language processing. However, their predominant design for English or a limited set of languages creates a substantial gap in their effectiveness for low-resource languages. To bridge this gap, we introduce MaLA-500, a novel large language model designed to cover an extensive range of 534 languages. To train MaLA-500, we employ vocabulary extension and continued pretraining on LLaMA 2 with Glot500-c. Our experiments on SIB-200 show that MaLA-500 achieves state-of-the-art in-context learning results. We release MaLA-500 at https://huggingface.co/MaLA-LM

{{</citation>}}


### (74/123) Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models (Hongzhan Lin et al., 2024)

{{<citation>}}

Hongzhan Lin, Ziyang Luo, Wei Gao, Jing Ma, Bo Wang, Ruichao Yang. (2024)  
**Towards Explainable Harmful Meme Detection through Multimodal Debate between Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13298v1)  

---


**ABSTRACT**  
The age of social media is flooded with Internet memes, necessitating a clear grasp and effective identification of harmful ones. This task presents a significant challenge due to the implicit meaning embedded in memes, which is not explicitly conveyed through the surface text and image. However, existing harmful meme detection methods do not present readable explanations that unveil such implicit meaning to support their detection decisions. In this paper, we propose an explainable approach to detect harmful memes, achieved through reasoning over conflicting rationales from both harmless and harmful positions. Specifically, inspired by the powerful capacity of Large Language Models (LLMs) on text generation and reasoning, we first elicit multimodal debate between LLMs to generate the explanations derived from the contradictory arguments. Then we propose to fine-tune a small language model as the debate judge for harmfulness inference, to facilitate multimodal fusion between the harmfulness rationales and the intrinsic multimodal information within memes. In this way, our model is empowered to perform dialectical reasoning over intricate and implicit harm-indicative patterns, utilizing multimodal explanations originating from both harmless and harmful arguments. Extensive experiments on three public meme datasets demonstrate that our harmful meme detection approach achieves much better performance than state-of-the-art methods and exhibits a superior capacity for explaining the meme harmfulness of the model predictions.

{{</citation>}}


### (75/123) Can AI Assistants Know What They Don't Know? (Qinyuan Cheng et al., 2024)

{{<citation>}}

Qinyuan Cheng, Tianxiang Sun, Xiangyang Liu, Wenwei Zhang, Zhangyue Yin, Shimin Li, Linyang Li, Zhengfu He, Kai Chen, Xipeng Qiu. (2024)  
**Can AI Assistants Know What They Don't Know?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13275v2)  

---


**ABSTRACT**  
Recently, AI assistants based on large language models (LLMs) show surprising performance in many tasks, such as dialogue, solving math problems, writing code, and using tools. Although LLMs possess intensive world knowledge, they still make factual errors when facing some knowledge intensive tasks, like open-domain question answering. These untruthful responses from the AI assistant may cause significant risks in practical applications. We believe that an AI assistant's refusal to answer questions it does not know is a crucial method for reducing hallucinations and making the assistant truthful. Therefore, in this paper, we ask the question "Can AI assistants know what they don't know and express them through natural language?" To answer this question, we construct a model-specific "I don't know" (Idk) dataset for an assistant, which contains its known and unknown questions, based on existing open-domain question answering datasets. Then we align the assistant with its corresponding Idk dataset and observe whether it can refuse to answer its unknown questions after alignment. Experimental results show that after alignment with Idk datasets, the assistant can refuse to answer most its unknown questions. For questions they attempt to answer, the accuracy is significantly higher than before the alignment.

{{</citation>}}


### (76/123) MF-AED-AEC: Speech Emotion Recognition by Leveraging Multimodal Fusion, ASR Error Detection, and ASR Error Correction (Jiajun He et al., 2024)

{{<citation>}}

Jiajun He, Xiaohan Shi, Xingfeng Li, Tomoki Toda. (2024)  
**MF-AED-AEC: Speech Emotion Recognition by Leveraging Multimodal Fusion, ASR Error Detection, and ASR Error Correction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs-SD, cs.CL, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.13260v1)  

---


**ABSTRACT**  
The prevalent approach in speech emotion recognition (SER) involves integrating both audio and textual information to comprehensively identify the speaker's emotion, with the text generally obtained through automatic speech recognition (ASR). An essential issue of this approach is that ASR errors from the text modality can worsen the performance of SER. Previous studies have proposed using an auxiliary ASR error detection task to adaptively assign weights of each word in ASR hypotheses. However, this approach has limited improvement potential because it does not address the coherence of semantic information in the text. Additionally, the inherent heterogeneity of different modalities leads to distribution gaps between their representations, making their fusion challenging. Therefore, in this paper, we incorporate two auxiliary tasks, ASR error detection (AED) and ASR error correction (AEC), to enhance the semantic coherence of ASR text, and further introduce a novel multi-modal fusion (MF) method to learn shared representations across modalities. We refer to our method as MF-AED-AEC. Experimental results indicate that MF-AED-AEC significantly outperforms the baseline model by a margin of 4.1\%.

{{</citation>}}


### (77/123) UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems (Hongru Wang et al., 2024)

{{<citation>}}

Hongru Wang, Wenyu Huang, Yang Deng, Rui Wang, Zezhong Wang, Yufei Wang, Fei Mi, Jeff Z. Pan, Kam-Fai Wong. (2024)  
**UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13256v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) has shown exceptional capabilities in many natual language understanding and generation tasks. However, the personalization issue still remains a much-coveted property, especially when it comes to the multiple sources involved in the dialogue system. To better plan and incorporate the use of multiple sources in generating personalized response, we firstly decompose it into three sub-tasks: Knowledge Source Selection, Knowledge Retrieval, and Response Generation. We then propose a novel Unified Multi-Source Retrieval-Augmented Generation system (UniMS-RAG) Specifically, we unify these three sub-tasks with different formulations into the same sequence-to-sequence paradigm during the training, to adaptively retrieve evidences and evaluate the relevance on-demand using special tokens, called acting tokens and evaluation tokens. Enabling language models to generate acting tokens facilitates interaction with various knowledge sources, allowing them to adapt their behavior to diverse task requirements. Meanwhile, evaluation tokens gauge the relevance score between the dialogue context and the retrieved evidence. In addition, we carefully design a self-refinement mechanism to iteratively refine the generated response considering 1) the consistency scores between the generated response and retrieved evidence; and 2) the relevance scores. Experiments on two personalized datasets (DuLeMon and KBP) show that UniMS-RAG achieves state-of-the-art performance on the knowledge source selection and response generation task with itself as a retriever in a unified manner. Extensive analyses and discussions are provided for shedding some new perspectives for personalized dialogue systems.

{{</citation>}}


### (78/123) SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning (Guoxin Chen et al., 2024)

{{<citation>}}

Guoxin Chen, Kexin Tang, Chao Yang, Fuying Ye, Yu Qiao, Yiming Qian. (2024)  
**SEER: Facilitating Structured Reasoning and Explanation via Reinforcement Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.13246v1)  

---


**ABSTRACT**  
Elucidating the reasoning process with structured explanations from question to answer is fundamentally crucial, as it significantly enhances the interpretability and trustworthiness of question-answering (QA) systems. However, structured explanations demand models to perform intricate structured reasoning, which poses great challenges. Most existing methods focus on single-step reasoning through supervised learning, ignoring logical dependencies between steps. Meanwhile, existing reinforcement learning (RL)-based methods overlook the structured relationships, impeding RL's potential in structured reasoning. In this paper, we propose SEER, a novel method that maximizes a structure-based return to facilitate structured reasoning and explanation. Our proposed structure-based return precisely describes the hierarchical and branching structure inherent in structured reasoning, effectively capturing the intricate relationships between states. We also introduce a fine-grained reward function to meticulously delineate diverse reasoning steps. Extensive experiments show that SEER significantly outperforms state-of-the-art methods, achieving an absolute improvement of 6.9% over RL-based methods on EntailmentBank, a 4.4% average improvement on STREET benchmark, and exhibiting outstanding efficiency and cross-dataset generalization performance.

{{</citation>}}


### (79/123) From Random to Informed Data Selection: A Diversity-Based Approach to Optimize Human Annotation and Few-Shot Learning (Alexandre Alcoforado et al., 2024)

{{<citation>}}

Alexandre Alcoforado, Thomas Palmeira Ferraz, Lucas Hideki Okamura, Israel Campos Fama, Arnold Moya Lavado, Bárbara Dias Bueno, Bruno Veloso, Anna Helena Reali Costa. (2024)  
**From Random to Informed Data Selection: A Diversity-Based Approach to Optimize Human Annotation and Few-Shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.13229v1)  

---


**ABSTRACT**  
A major challenge in Natural Language Processing is obtaining annotated data for supervised learning. An option is the use of crowdsourcing platforms for data annotation. However, crowdsourcing introduces issues related to the annotator's experience, consistency, and biases. An alternative is to use zero-shot methods, which in turn have limitations compared to their few-shot or fully supervised counterparts. Recent advancements driven by large language models show potential, but struggle to adapt to specialized domains with severely limited data. The most common approaches therefore involve the human itself randomly annotating a set of datapoints to build initial datasets. But randomly sampling data to be annotated is often inefficient as it ignores the characteristics of the data and the specific needs of the model. The situation worsens when working with imbalanced datasets, as random sampling tends to heavily bias towards the majority classes, leading to excessive annotated data. To address these issues, this paper contributes an automatic and informed data selection architecture to build a small dataset for few-shot learning. Our proposal minimizes the quantity and maximizes diversity of data selected for human annotation, while improving model performance.

{{</citation>}}


### (80/123) Scalable Link Prediction on Large-Scale Heterogeneous Graphs with Large Language Models (Baolong Bi et al., 2024)

{{<citation>}}

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Xueqi Chen. (2024)  
**Scalable Link Prediction on Large-Scale Heterogeneous Graphs with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-SI, cs.CL  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.13227v1)  

---


**ABSTRACT**  
Exploring the application of large-scale language models to graph learning is a novel endeavor. However, the vast amount of information inherent in large graphs poses significant challenges to this process. This paper focuses on the link prediction task and introduces LPNL (Link Prediction via Natural Language), a framework based on a large language model designed for scalable link prediction on large-scale heterogeneous graphs.We design novel prompts for link prediction that articulate graph details in natural language. We propose a two-stage sampling pipeline to extract crucial information from large-scale heterogeneous graphs, and a divide-and-conquer strategy to control the input token count within predefined limits, addressing the challenge of overwhelming information. We fine-tune a T5 model based on our self-supervised learning designed for for link prediction. Extensive experiments on a large public heterogeneous graphs demonstrate that LPNL outperforms various advanced baselines, highlighting its remarkable performance in link prediction tasks on large-scale graphs.

{{</citation>}}


### (81/123) TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data (Fengbin Zhu et al., 2024)

{{<citation>}}

Fengbin Zhu, Ziyang Liu, Fuli Feng, Chao Wang, Moxin Li, Tat-Seng Chua. (2024)  
**TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, LLaMA, Language Model, QA, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.13223v1)  

---


**ABSTRACT**  
In this work, we address question answering (QA) over a hybrid of tabular and textual data that are very common content on the Web (e.g. SEC filings), where discrete reasoning capabilities are often required. Recently, large language models (LLMs) like GPT-4 have demonstrated strong multi-step reasoning capabilities. We then consider harnessing the amazing power of LLMs to solve our task. We abstract a Step-wise Pipeline for tabular and textual QA, which consists of three key steps, including Extractor, Reasoner and Executor, and initially design an instruction to instantiate the pipeline and validate that GPT-4 outperforms all existing methods. However, utilizing an online LLM like GPT-4 holds various challenges in terms of cost, latency, and data security risk, which motivates us to specialize smaller LLMs in this task. We develop a TAT-LLM language model by fine-tuning LLaMA 2 with the training data generated automatically from existing expert-annotated datasets following the Step-wise Pipeline. The experimental results have verified that our TAT-LLM model can outperform all baseline models, including the previous best fine-tuned models and very large-scale LLMs like GPT-4 on FinQA, TAT-QA and TAT-DQA benchmarks. We hope our work can serve as a pioneering example of specializing smaller language models for specific tasks.

{{</citation>}}


### (82/123) ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Refinement (Xinliang Frederick Zhang et al., 2024)

{{<citation>}}

Xinliang Frederick Zhang, Carter Blum, Temma Choji, Shalin Shah, Alakananda Vempala. (2024)  
**ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Refinement**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13218v1)  

---


**ABSTRACT**  
Structural extraction of events within discourse is critical since it avails a deeper understanding of communication patterns and behavior trends. Event argument extraction (EAE), at the core of event-centric understanding, is the task of identifying role-specific text spans (i.e., arguments) for a given event. Document-level EAE (DocEAE) focuses on arguments that are scattered across an entire document. In this work, we explore the capabilities of open source Large Language Models (LLMs), i.e., Flan-UL2, for the DocEAE task. To this end, we propose ULTRA, a hierarchical framework that extracts event arguments more cost-effectively -- the method needs as few as 50 annotations and doesn't require hitting costly API endpoints. Further, it alleviates the positional bias issue intrinsic to LLMs. ULTRA first sequentially reads text chunks of a document to generate a candidate argument set, upon which ULTRA learns to drop non-pertinent candidates through self-refinement. We further introduce LEAFER to address the challenge LLMs face in locating the exact boundary of an argument span. ULTRA outperforms strong baselines, which include strong supervised models and ChatGPT, by 9.8% when evaluated by the exact match (EM) metric.

{{</citation>}}


### (83/123) CFMatch: Aligning Automated Answer Equivalence Evaluation with Expert Judgments For Open-Domain Question Answering (Zongxia Li et al., 2024)

{{<citation>}}

Zongxia Li, Ishani Mondal, Yijun Liang, Huy Nghiem, Jordan Boyd-Graber. (2024)  
**CFMatch: Aligning Automated Answer Equivalence Evaluation with Expert Judgments For Open-Domain Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.13170v1)  

---


**ABSTRACT**  
Question answering (QA) can only make progress if we know if an answer is correct, but for many of the most challenging and interesting QA examples, current evaluation metrics to determine answer equivalence (AE) often do not align with human judgments, particularly more verbose, free-form answers from large language models (LLM). There are two challenges: a lack of data and that models are too big: LLM-based scorers can correlate better with human judges, but this task has only been tested on limited QA datasets, and even when available, update of the model is limited because LLMs are large and often expensive. We rectify both of these issues by providing clear and consistent guidelines for evaluating AE in machine QA adopted from professional human QA contests. We also introduce a combination of standard evaluation and a more efficient, robust, and lightweight discriminate AE classifier-based matching method (CFMatch, smaller than 1 MB), trained and validated to more accurately evaluate answer correctness in accordance with adopted expert AE rules that are more aligned with human judgments.

{{</citation>}}


### (84/123) Misgendering and Assuming Gender in Machine Translation when Working with Low-Resource Languages (Sourojit Ghosh et al., 2024)

{{<citation>}}

Sourojit Ghosh, Srishti Chatterjee. (2024)  
**Misgendering and Assuming Gender in Machine Translation when Working with Low-Resource Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Low-Resource, Machine Translation  
[Paper Link](http://arxiv.org/abs/2401.13165v2)  

---


**ABSTRACT**  
This chapter focuses on gender-related errors in machine translation (MT) in the context of low-resource languages. We begin by explaining what low-resource languages are, examining the inseparable social and computational factors that create such linguistic hierarchies. We demonstrate through a case study of our mother tongue Bengali, a global language spoken by almost 300 million people but still classified as low-resource, how gender is assumed and inferred in translations to and from the high(est)-resource English when no such information is provided in source texts. We discuss the postcolonial and societal impacts of such errors leading to linguistic erasure and representational harms, and conclude by discussing potential solutions towards uplifting languages by providing them more agency in MT conversations.

{{</citation>}}


## cs.IR (9)



### (85/123) Algorithmically Curated Lies: How Search Engines Handle Misinformation about US Biolabs in Ukraine (Elizaveta Kuznetsova et al., 2024)

{{<citation>}}

Elizaveta Kuznetsova, Mykola Makhortykh, Maryna Sydorova, Aleksandra Urman, Ilaria Vitulano, Martha Stolze. (2024)  
**Algorithmically Curated Lies: How Search Engines Handle Misinformation about US Biolabs in Ukraine**  

---
Primary Category: cs.IR  
Categories: cs-CY, cs-IR, cs.IR  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2401.13832v1)  

---


**ABSTRACT**  
The growing volume of online content prompts the need for adopting algorithmic systems of information curation. These systems range from web search engines to recommender systems and are integral for helping users stay informed about important societal developments. However, unlike journalistic editing the algorithmic information curation systems (AICSs) are known to be subject to different forms of malperformance which make them vulnerable to possible manipulation. The risk of manipulation is particularly prominent in the case when AICSs have to deal with information about false claims that underpin propaganda campaigns of authoritarian regimes. Using as a case study of the Russian disinformation campaign concerning the US biolabs in Ukraine, we investigate how one of the most commonly used forms of AICSs - i.e. web search engines - curate misinformation-related content. For this aim, we conduct virtual agent-based algorithm audits of Google, Bing, and Yandex search outputs in June 2022. Our findings highlight the troubling performance of search engines. Even though some search engines, like Google, were less likely to return misinformation results, across all languages and locations, the three search engines still mentioned or promoted a considerable share of false content (33% on Google; 44% on Bing, and 70% on Yandex). We also find significant disparities in misinformation exposure based on the language of search, with all search engines presenting a higher number of false stories in Russian. Location matters as well with users from Germany being more likely to be exposed to search results promoting false information. These observations stress the possibility of AICSs being vulnerable to manipulation, in particular in the case of the unfolding propaganda campaigns, and underline the importance of monitoring performance of these systems to prevent it.

{{</citation>}}


### (86/123) Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation (Ludovico Boratto et al., 2024)

{{<citation>}}

Ludovico Boratto, Francesco Fabbri, Gianni Fenu, Mirko Marras, Giacomo Medda. (2024)  
**Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.13823v2)  

---


**ABSTRACT**  
Efforts in the recommendation community are shifting from the sole emphasis on utility to considering beyond-utility factors, such as fairness and robustness. Robustness of recommendation models is typically linked to their ability to maintain the original utility when subjected to attacks. Limited research has explored the robustness of a recommendation model in terms of fairness, e.g., the parity in performance across groups, under attack scenarios. In this paper, we aim to assess the robustness of graph-based recommender systems concerning fairness, when exposed to attacks based on edge-level perturbations. To this end, we considered four different fairness operationalizations, including both consumer and provider perspectives. Experiments on three datasets shed light on the impact of perturbations on the targeted fairness notion, uncovering key shortcomings in existing evaluation protocols for robustness. As an example, we observed perturbations affect consumer fairness on a higher extent than provider fairness, with alarming unfairness for the former. Source code: https://github.com/jackmedda/CPFairRobust

{{</citation>}}


### (87/123) Building Contextual Knowledge Graphs for Personalized Learning Recommendations using Text Mining and Semantic Graph Completion (Hasan Abu-Rasheed et al., 2024)

{{<citation>}}

Hasan Abu-Rasheed, Mareike Dornhöfer, Christian Weber, Gábor Kismihók, Ulrike Buchmann, Madjid Fathi. (2024)  
**Building Contextual Knowledge Graphs for Personalized Learning Recommendations using Text Mining and Semantic Graph Completion**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.13609v1)  

---


**ABSTRACT**  
Modelling learning objects (LO) within their context enables the learner to advance from a basic, remembering-level, learning objective to a higher-order one, i.e., a level with an application- and analysis objective. While hierarchical data models are commonly used in digital learning platforms, using graph-based models enables representing the context of LOs in those platforms. This leads to a foundation for personalized recommendations of learning paths. In this paper, the transformation of hierarchical data models into knowledge graph (KG) models of LOs using text mining is introduced and evaluated. We utilize custom text mining pipelines to mine semantic relations between elements of an expert-curated hierarchical model. We evaluate the KG structure and relation extraction using graph quality-control metrics and the comparison of algorithmic semantic-similarities to expert-defined ones. The results show that the relations in the KG are semantically comparable to those defined by domain experts, and that the proposed KG improves representing and linking the contexts of LOs through increasing graph communities and betweenness centrality.

{{</citation>}}


### (88/123) Fine-grained Contract NER using instruction based model (Hiranmai Sri Adibhatla et al., 2024)

{{<citation>}}

Hiranmai Sri Adibhatla, Pavan Baswani, Manish Shrivastava. (2024)  
**Fine-grained Contract NER using instruction based model**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model, NER, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2401.13545v1)  

---


**ABSTRACT**  
Lately, instruction-based techniques have made significant strides in improving performance in few-shot learning scenarios. They achieve this by bridging the gap between pre-trained language models and fine-tuning for specific downstream tasks. Despite these advancements, the performance of Large Language Models (LLMs) in information extraction tasks like Named Entity Recognition (NER), using prompts or instructions, still falls short of supervised baselines. The reason for this performance gap can be attributed to the fundamental disparity between NER and LLMs. NER is inherently a sequence labeling task, where the model must assign entity-type labels to individual tokens within a sentence. In contrast, LLMs are designed as a text generation task. This distinction between semantic labeling and text generation leads to subpar performance. In this paper, we transform the NER task into a text-generation task that can be readily adapted by LLMs. This involves enhancing source sentences with task-specific instructions and answer choices, allowing for the identification of entities and their types within natural language. We harness the strength of LLMs by integrating supervised learning within them. The goal of this combined strategy is to boost the performance of LLMs in extraction tasks like NER while simultaneously addressing hallucination issues often observed in LLM-generated content. A novel corpus Contract NER comprising seven frequently observed contract categories, encompassing named entities associated with 18 distinct legal entity types is released along with our baseline models. Our models and dataset are available to the community for future research * .

{{</citation>}}


### (89/123) TPRF: A Transformer-based Pseudo-Relevance Feedback Model for Efficient and Effective Retrieval (Chuting Yu et al., 2024)

{{<citation>}}

Chuting Yu, Hang Li, Ahmed Mourad, Bevan Koopman, Guido Zuccon. (2024)  
**TPRF: A Transformer-based Pseudo-Relevance Feedback Model for Efficient and Effective Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.13509v1)  

---


**ABSTRACT**  
This paper considers Pseudo-Relevance Feedback (PRF) methods for dense retrievers in a resource constrained environment such as that of cheap cloud instances or embedded systems (e.g., smartphones and smartwatches), where memory and CPU are limited and GPUs are not present. For this, we propose a transformer-based PRF method (TPRF), which has a much smaller memory footprint and faster inference time compared to other deep language models that employ PRF mechanisms, with a marginal effectiveness loss. TPRF learns how to effectively combine the relevance feedback signals from dense passage representations. Specifically, TPRF provides a mechanism for modelling relationships and weights between the query and the relevance feedback signals. The method is agnostic to the specific dense representation used and thus can be generally applied to any dense retriever.

{{</citation>}}


### (90/123) SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval (Siwei Wu et al., 2024)

{{<citation>}}

Siwei Wu, Yizhi Li, Kang Zhu, Ge Zhang, Yiming Liang, Kaijing Ma, Chenghao Xiao, Haoran Zhang, Bohao Yang, Wenhu Chen, Wenhao Huang, Noura Al Moubayed, Jie Fu, Chenghua Lin. (2024)  
**SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-CV, cs-IR, cs-MM, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.13478v1)  

---


**ABSTRACT**  
Multi-modal information retrieval (MMIR) is a rapidly evolving field, where significant progress, particularly in image-text pairing, has been made through advanced representation learning and cross-modality alignment research. However, current benchmarks for evaluating MMIR performance in image-text pairing within the scientific domain show a notable gap, where chart and table images described in scholarly language usually do not play a significant role. To bridge this gap, we develop a specialised scientific MMIR (SciMMIR) benchmark by leveraging open-access paper collections to extract data relevant to the scientific domain. This benchmark comprises 530K meticulously curated image-text pairs, extracted from figures and tables with detailed captions in scientific documents. We further annotate the image-text pairs with two-level subset-subcategory hierarchy annotations to facilitate a more comprehensive evaluation of the baselines. We conducted zero-shot and fine-tuning evaluations on prominent multi-modal image-captioning and visual language models, such as CLIP and BLIP. Our analysis offers critical insights for MMIR in the scientific domain, including the impact of pre-training and fine-tuning settings and the influence of the visual and textual encoders. All our data and checkpoints are publicly available at https://github.com/Wusiwei0410/SciMMIR.

{{</citation>}}


### (91/123) Decentralized Collaborative Learning with Adaptive Reference Data for On-Device POI Recommendation (Ruiqi Zheng et al., 2024)

{{<citation>}}

Ruiqi Zheng, Liang Qu, Tong Chen, Lizhen Cui, Yuhui Shi, Hongzhi Yin. (2024)  
**Decentralized Collaborative Learning with Adaptive Reference Data for On-Device POI Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2401.13448v2)  

---


**ABSTRACT**  
In Location-based Social Networks, Point-of-Interest (POI) recommendation helps users discover interesting places. There is a trend to move from the cloud-based model to on-device recommendations for privacy protection and reduced server reliance. Due to the scarcity of local user-item interactions on individual devices, solely relying on local instances is not adequate. Collaborative Learning (CL) emerges to promote model sharing among users, where reference data is an intermediary that allows users to exchange their soft decisions without directly sharing their private data or parameters, ensuring privacy and benefiting from collaboration. However, existing CL-based recommendations typically use a single reference for all users. Reference data valuable for one user might be harmful to another, given diverse user preferences. Users may not offer meaningful soft decisions on items outside their interest scope. Consequently, using the same reference data for all collaborations can impede knowledge exchange and lead to sub-optimal performance. To address this gap, we introduce the Decentralized Collaborative Learning with Adaptive Reference Data (DARD) framework, which crafts adaptive reference data for effective user collaboration. It first generates a desensitized public reference data pool with transformation and probability data generation methods. For each user, the selection of adaptive reference data is executed in parallel by training loss tracking and influence function. Local models are trained with individual private data and collaboratively with the geographical and semantic neighbors. During the collaboration between two users, they exchange soft decisions based on a combined set of their adaptive reference data. Our evaluations across two real-world datasets highlight DARD's superiority in recommendation performance and addressing the scarcity of available reference data.

{{</citation>}}


### (92/123) Query Exposure Prediction for Groups of Documents in Rankings (Thomas Jaenich et al., 2024)

{{<citation>}}

Thomas Jaenich, Graham McDonald, Iadh Ounis. (2024)  
**Query Exposure Prediction for Groups of Documents in Rankings**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2401.13434v1)  

---


**ABSTRACT**  
The main objective of an Information Retrieval system is to provide a user with the most relevant documents to the user's query. To do this, modern IR systems typically deploy a re-ranking pipeline in which a set of documents is retrieved by a lightweight first-stage retrieval process and then re-ranked by a more effective but expensive model. However, the success of a re-ranking pipeline is heavily dependent on the performance of the first stage retrieval, since new documents are not usually identified during the re-ranking stage. Moreover, this can impact the amount of exposure that a particular group of documents, such as documents from a particular demographic group, can receive in the final ranking. For example, the fair allocation of exposure becomes more challenging or impossible if the first stage retrieval returns too few documents from certain groups, since the number of group documents in the ranking affects the exposure more than the documents' positions. With this in mind, it is beneficial to predict the amount of exposure that a group of documents is likely to receive in the results of the first stage retrieval process, in order to ensure that there are a sufficient number of documents included from each of the groups. In this paper, we introduce the novel task of query exposure prediction (QEP). Specifically, we propose the first approach for predicting the distribution of exposure that groups of documents will receive for a given query. Our new approach, called GEP, uses lexical information from individual groups of documents to estimate the exposure the groups will receive in a ranking. Our experiments on the TREC 2021 and 2022 Fair Ranking Track test collections show that our proposed GEP approach results in exposure predictions that are up to 40 % more accurate than the predictions of adapted existing query performance prediction and resource allocation approaches.

{{</citation>}}


### (93/123) It's About Time: Incorporating Temporality in Retrieval Augmented Language Models (Anoushka Gade et al., 2024)

{{<citation>}}

Anoushka Gade, Jorjeta Jetcheva. (2024)  
**It's About Time: Incorporating Temporality in Retrieval Augmented Language Models**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13222v2)  

---


**ABSTRACT**  
The web serves as a global repository of knowledge, used by billions of people to search for information. Ensuring that users receive the most relevant and up-to-date information, especially in the presence of multiple versions of web content from different time points remains a critical challenge for information retrieval. This challenge has recently been compounded by the increased use of question answering tools trained on Wikipedia or web content and powered by large language models (LLMs) which have been found to make up information (or hallucinate), and in addition have been shown to struggle with the temporal dimensions of information. Even Retriever Augmented Language Models (RALMs) which incorporate a document database to reduce LLM hallucination are unable to handle temporal queries correctly. This leads to instances where RALMs respond to queries such as "Who won the Wimbledon Championship?", by retrieving document passages related to Wimbledon but without the ability to differentiate between them based on how recent they are.   In this paper, we propose and evaluate, TempRALM, a temporally-aware Retriever Augmented Language Model (RALM) with few-shot learning extensions, which takes into account both semantically and temporally relevant documents relative to a given query, rather than relying on semantic similarity alone. We show that our approach results in up to 74% improvement in performance over the baseline RALM model, without requiring model pre-training, recalculating or replacing the RALM document index, or adding other computationally intensive elements.

{{</citation>}}


## cs.SI (2)



### (94/123) Longitudinal Sentiment Topic Modelling of Reddit Posts (Fabian Nwaoha et al., 2024)

{{<citation>}}

Fabian Nwaoha, Ziyad Gaffar, Ho Joon Chun, Marina Sokolova. (2024)  
**Longitudinal Sentiment Topic Modelling of Reddit Posts**  

---
Primary Category: cs.SI  
Categories: I-2-7, cs-IR, cs-SI, cs.SI  
Keywords: Topic Model  
[Paper Link](http://arxiv.org/abs/2401.13805v1)  

---


**ABSTRACT**  
In this study, we analyze texts of Reddit posts written by students of four major Canadian universities. We gauge the emotional tone and uncover prevailing themes and discussions through longitudinal topic modeling of posts textual data. Our study focuses on four years, 2020-2023, covering COVID-19 pandemic and after pandemic years. Our results highlight a gradual uptick in discussions related to mental health.

{{</citation>}}


### (95/123) The Dynamics of (Not) Unfollowing Misinformation Spreaders (Joshua Ashkinaze et al., 2024)

{{<citation>}}

Joshua Ashkinaze, Eric Gilbert, Ceren Budak. (2024)  
**The Dynamics of (Not) Unfollowing Misinformation Spreaders**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-HC, cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.13480v1)  

---


**ABSTRACT**  
Many studies explore how people 'come into' misinformation exposure. But much less is known about how people 'come out of' misinformation exposure. Do people organically sever ties to misinformation spreaders? And what predicts doing so? Over six months, we tracked the frequency and predictors of ~1M followers unfollowing ~5K health misinformation spreaders on Twitter. We found that misinformation ties are persistent. Monthly unfollowing rates are just 0.52%. Users are also 31% more likely to unfollow non-misinformation spreaders than they are to unfollow misinformation spreaders. Although generally infrequent, the factors most associated with unfollowing misinformation spreaders are (1) redundancy and (2) ideology. First, users initially following many spreaders, or who follow spreaders that tweet often, are most likely to unfollow later. Second, liberals are more likely to unfollow than conservatives. Overall, we observe strong persistence of misinformation ties. The fact that users rarely unfollow misinformation spreaders suggests a need for external nudges and the importance of preventing exposure from arising in the first place.

{{</citation>}}


## cs.HC (8)



### (96/123) Exploring Parent's Needs for Children-Centered AI to Support Preschoolers' Storytelling and Reading Activities (Yuling Sun et al., 2024)

{{<citation>}}

Yuling Sun, Jiali Liu, Bingsheng Yao, Jiaju Chen, Dakuo Wang, Xiaojuan Ma, Yuxuan Lu, Ying Xu, Liang He. (2024)  
**Exploring Parent's Needs for Children-Centered AI to Support Preschoolers' Storytelling and Reading Activities**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13804v1)  

---


**ABSTRACT**  
Interactive storytelling is vital for preschooler development. While children's interactive partners have traditionally been their parents and teachers, recent advances in artificial intelligence (AI) have sparked a surge of AI-based storytelling technologies. As these technologies become increasingly ubiquitous in preschoolers' lives, questions arise regarding how they function in practical storytelling scenarios and, in particular, how parents, the most critical stakeholders, experience and perceive these technologies. This paper investigates these questions through a qualitative study with 17 parents of children aged 3-6. Our findings suggest that even though AI-based storytelling technologies provide more immersive and engaging interaction, they still cannot meet parents' expectations due to a series of interactive, functional, and algorithmic challenges. We elaborate on these challenges and discuss the possible implications of future AI-based storytelling technologies for preschoolers. We conclude by highlighting the design implications for future AI-based storytelling technologies.

{{</citation>}}


### (97/123) Synergizing Human Expertise and AI Efficiency with Language Model for Microscopy Operation and Automated Experiment Design (Yongtao Liu et al., 2024)

{{<citation>}}

Yongtao Liu, Marti Checa, Rama K. Vasudevan. (2024)  
**Synergizing Human Expertise and AI Efficiency with Language Model for Microscopy Operation and Automated Experiment Design**  

---
Primary Category: cs.HC  
Categories: cond-mat-mtrl-sci, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13803v1)  

---


**ABSTRACT**  
With the advent of large language models (LLMs), in both the open source and proprietary domains, attention is turning to how to exploit such artificial intelligence (AI) systems in assisting complex scientific tasks, such as material synthesis, characterization, analysis and discovery. Here, we explore the utility of LLM, particularly ChatGPT4, in combination with application program interfaces (APIs) in tasks of experimental design, programming workflows, and data analysis in scanning probe microscopy, using both in-house developed API and API given by a commercial vendor for instrument control. We find that the LLM can be especially useful in converting ideations of experimental workflows to executable code on microscope APIs. Beyond code generation, we find that the GPT4 is capable of analyzing microscopy images in a generic sense. At the same time, we find that GPT4 suffers from inability to extend beyond basic analyses or more in-depth technical experimental design. We argue that a LLM specifically fine-tuned for individual scientific domains can potentially be a better language interface for converting scientific ideations from human experts to executable workflows, such a synergy between human expertise and LLM efficiency in experimentation can open new door for accelerating scientific research, enabling effective experimental protocols archive and sharing in scientific community.

{{</citation>}}


### (98/123) Supporting Sensemaking of Large Language Model Outputs at Scale (Katy Ilonka Gero et al., 2024)

{{<citation>}}

Katy Ilonka Gero, Chelse Swoopes, Ziwei Gu, Jonathan K. Kummerfeld, Elena L. Glassman. (2024)  
**Supporting Sensemaking of Large Language Model Outputs at Scale**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13726v1)  

---


**ABSTRACT**  
Large language models (LLMs) are capable of generating multiple responses to a single prompt, yet little effort has been expended to help end-users or system designers make use of this capability. In this paper, we explore how to present many LLM responses at once. We design five features, which include both pre-existing and novel methods for computing similarities and differences across textual documents, as well as how to render their outputs. We report on a controlled user study (n=24) and eight case studies evaluating these features and how they support users in different tasks. We find that the features support a wide variety of sensemaking tasks and even make tasks previously considered to be too difficult by our participants now tractable. Finally, we present design guidelines to inform future explorations of new LLM interfaces.

{{</citation>}}


### (99/123) Design, Development, and Deployment of Context-Adaptive AI Systems for Enhanced End-User Adoption (Christine P Lee, 2024)

{{<citation>}}

Christine P Lee. (2024)  
**Design, Development, and Deployment of Context-Adaptive AI Systems for Enhanced End-User Adoption**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13643v1)  

---


**ABSTRACT**  
My research centers on the development of context-adaptive AI systems to improve end-user adoption through the integration of technical methods. I deploy these AI systems across various interaction modalities, including user interfaces and embodied agents like robots, to expand their practical applicability. My research unfolds in three key stages: design, development, and deployment. In the design phase, user-centered approaches were used to understand user experiences with AI systems and create design tools for user participation in crafting AI explanations. In the ongoing development stage, a safety-guaranteed AI system for a robot agent was created to automatically provide adaptive solutions and explanations for unforeseen scenarios. The next steps will involve the implementation and evaluation of context-adaptive AI systems in various interaction forms. I seek to prioritize human needs in technology development, creating AI systems that tangibly benefit end-users in real-world applications and enhance interaction experiences.

{{</citation>}}


### (100/123) Proactive Emotion Tracker: AI-Driven Continuous Mood and Emotion Monitoring (Mohammad Asif et al., 2024)

{{<citation>}}

Mohammad Asif, Sudhakar Mishra, Ankush Sonker, Sanidhya Gupta, Somesh Kumar Maurya, Uma Shanker Tiwary. (2024)  
**Proactive Emotion Tracker: AI-Driven Continuous Mood and Emotion Monitoring**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, BERT  
[Paper Link](http://arxiv.org/abs/2401.13722v1)  

---


**ABSTRACT**  
This research project aims to tackle the growing mental health challenges in today's digital age. It employs a modified pre-trained BERT model to detect depressive text within social media and users' web browsing data, achieving an impressive 93% test accuracy. Simultaneously, the project aims to incorporate physiological signals from wearable devices, such as smartwatches and EEG sensors, to provide long-term tracking and prognosis of mood disorders and emotional states. This comprehensive approach holds promise for enhancing early detection of depression and advancing overall mental health outcomes.

{{</citation>}}


### (101/123) Information That Matters: Exploring Information Needs of People Affected by Algorithmic Decisions (Timothée Schmude et al., 2024)

{{<citation>}}

Timothée Schmude, Laura Koesten, Torsten Möller, Sebastian Tschiatschek. (2024)  
**Information That Matters: Exploring Information Needs of People Affected by Algorithmic Decisions**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13324v4)  

---


**ABSTRACT**  
Explanations of AI systems rarely address the information needs of people affected by algorithmic decision-making (ADM). This gap between conveyed information and information that matters to affected stakeholders can impede understanding and adherence to regulatory frameworks such as the AI Act. To address this gap, we present the "XAI Novice Question Bank": A catalog of affected stakeholders' information needs in two ADM use cases (employment prediction and health monitoring), covering the categories data, system context, system usage, and system specifications. Information needs were gathered in an interview study where participants received explanations in response to their inquiries. Participants further reported their understanding and decision confidence, showing that while confidence tended to increase after receiving explanations, participants also met understanding challenges, such as being unable to tell why their understanding felt incomplete. Explanations further influenced participants' perceptions of the systems' risks and benefits, which they confirmed or changed depending on the use case. When risks were perceived as high, participants expressed particular interest in explanations about intention, such as why and to what end a system was put in place. With this work, we aim to support the inclusion of affected stakeholders into explainability by contributing an overview of information and challenges relevant to them when deciding on the adoption of ADM systems. We close by summarizing our findings in a list of six key implications that inform the design of future explanations for affected stakeholder audiences.

{{</citation>}}


### (102/123) No Longer Trending on Artstation: Prompt Analysis of Generative AI Art (Jon McCormack et al., 2024)

{{<citation>}}

Jon McCormack, Maria Teresa Llano, Stephen James Krol, Nina Rajcic. (2024)  
**No Longer Trending on Artstation: Prompt Analysis of Generative AI Art**  

---
Primary Category: cs.HC  
Categories: J-5; I-2, cs-AI, cs-CV, cs-CY, cs-HC, cs-NE, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2401.14425v1)  

---


**ABSTRACT**  
Image generation using generative AI is rapidly becoming a major new source of visual media, with billions of AI generated images created using diffusion models such as Stable Diffusion and Midjourney over the last few years. In this paper we collect and analyse over 3 million prompts and the images they generate. Using natural language processing, topic analysis and visualisation methods we aim to understand collectively how people are using text prompts, the impact of these systems on artists, and more broadly on the visual cultures they promote. Our study shows that prompting focuses largely on surface aesthetics, reinforcing cultural norms, popular conventional representations and imagery. We also find that many users focus on popular topics (such as making colouring books, fantasy art, or Christmas cards), suggesting that the dominant use for the systems analysed is recreational rather than artistic.

{{</citation>}}


### (103/123) GraphiMind: LLM-centric Interface for Information Graphics Design (Qirui Huang et al., 2024)

{{<citation>}}

Qirui Huang, Min Lu, Joel Lanir, Dani Lischinski, Daniel Cohen-Or, Hui Huang. (2024)  
**GraphiMind: LLM-centric Interface for Information Graphics Design**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13245v1)  

---


**ABSTRACT**  
Information graphics are pivotal in effective information dissemination and storytelling. However, creating such graphics is extremely challenging for non-professionals, since the design process requires multifaceted skills and comprehensive knowledge. Thus, despite the many available authoring tools, a significant gap remains in enabling non-experts to produce compelling information graphics seamlessly, especially from scratch. Recent breakthroughs show that Large Language Models (LLMs), especially when tool-augmented, can autonomously engage with external tools, making them promising candidates for enabling innovative graphic design applications. In this work, we propose a LLM-centric interface with the agent GraphiMind for automatic generation, recommendation, and composition of information graphics design resources, based on user intent expressed through natural language. Our GraphiMind integrates a Textual Conversational Interface, powered by tool-augmented LLM, with a traditional Graphical Manipulation Interface, streamlining the entire design process from raw resource curation to composition and refinement. Extensive evaluations highlight our tool's proficiency in simplifying the design process, opening avenues for its use by non-professional users. Moreover, we spotlight the potential of LLMs in reshaping the domain of information graphics design, offering a blend of automation, versatility, and user-centric interactivity.

{{</citation>}}


## cs.SE (3)



### (104/123) Investigating the Efficacy of Large Language Models for Code Clone Detection (Mohamad Khajezade et al., 2024)

{{<citation>}}

Mohamad Khajezade, Jie JW Wu, Fatemeh Hendijani Fard, Gema Rodríguez-Pérez, Mohamed Sami Shehata. (2024)  
**Investigating the Efficacy of Large Language Models for Code Clone Detection**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-CL, cs-LG, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.13802v3)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable success in various natural language processing and software engineering tasks, such as code generation. The LLMs are mainly utilized in the prompt-based zero/few-shot paradigm to guide the model in accomplishing the task. GPT-based models are one of the popular ones studied for tasks such as code comment generation or test generation. These tasks are `generative' tasks. However, there is limited research on the usage of LLMs for `non-generative' tasks such as classification using the prompt-based paradigm. In this preliminary exploratory study, we investigated the applicability of LLMs for Code Clone Detection (CCD), a non-generative task. By building a mono-lingual and cross-lingual CCD dataset derived from CodeNet, we first investigated two different prompts using ChatGPT to detect Type-4 code clones in Java-Java and Java-Ruby pairs in a zero-shot setting. We then conducted an analysis to understand the strengths and weaknesses of ChatGPT in CCD. ChatGPT surpasses the baselines in cross-language CCD attaining an F1-score of 0.877 and achieves comparable performance to fully fine-tuned models for mono-lingual CCD, with an F1-score of 0.878. Also, the prompt and the difficulty level of the problems has an impact on the performance of ChatGPT. Finally we provide insights and future directions based on our initial analysis

{{</citation>}}


### (105/123) What Makes a Great Software Quality Assurance Engineer? (Roselane Silva Farias et al., 2024)

{{<citation>}}

Roselane Silva Farias, Iftekhar Ahmed, Eduardo Santana de Almeida. (2024)  
**What Makes a Great Software Quality Assurance Engineer?**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.13623v1)  

---


**ABSTRACT**  
Software Quality Assurance (SQA) Engineers are responsible for assessing a product during every phase of the software development process to ensure that the outcomes of each phase and the final product possess the desired qualities. In general, a great SQA engineer needs to have a different set of abilities from development engineers to effectively oversee the entire product development process from beginning to end. Recent empirical studies identified important attributes of software engineers and managers, but the quality assurance role is overlooked. As software quality aspects have become more of a priority in the life cycle of software development, employers seek professionals that best suit the company's objectives and new graduates desire to make a valuable contribution through their job as an SQA engineer, but what makes them great? We addressed this knowledge gap by conducting 25 semi-structured interviews and 363 survey respondents with software quality assurance engineers from different companies around the world. We use the data collected from these activities to derive a comprehensive set of attributes that are considered important. As a result of the interviews, twenty-five attributes were identified and grouped into five main categories: personal, social, technical, management, and decision-making attributes. Through a rating survey, we confirmed that the distinguishing characteristics of great SQA engineers are curiosity, the ability to communicate effectively, and critical thinking skills. This work will guide further studies with SQA practitioners, by considering contextual factors and providing some implications for research and practice.

{{</citation>}}


### (106/123) Deep Learning Model Reuse in the HuggingFace Community: Challenges, Benefit and Trends (Mina Taraghi et al., 2024)

{{<citation>}}

Mina Taraghi, Gianolli Dorcelus, Armstrong Foundjem, Florian Tambon, Foutse Khomh. (2024)  
**Deep Learning Model Reuse in the HuggingFace Community: Challenges, Benefit and Trends**  

---
Primary Category: cs.SE  
Categories: cs-CY, cs-LG, cs-SE, cs.SE  
Keywords: Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2401.13177v1)  

---


**ABSTRACT**  
The ubiquity of large-scale Pre-Trained Models (PTMs) is on the rise, sparking interest in model hubs, and dedicated platforms for hosting PTMs. Despite this trend, a comprehensive exploration of the challenges that users encounter and how the community leverages PTMs remains lacking. To address this gap, we conducted an extensive mixed-methods empirical study by focusing on discussion forums and the model hub of HuggingFace, the largest public model hub. Based on our qualitative analysis, we present a taxonomy of the challenges and benefits associated with PTM reuse within this community. We then conduct a quantitative study to track model-type trends and model documentation evolution over time. Our findings highlight prevalent challenges such as limited guidance for beginner users, struggles with model output comprehensibility in training or inference, and a lack of model understanding. We also identified interesting trends among models where some models maintain high upload rates despite a decline in topics related to them. Additionally, we found that despite the introduction of model documentation tools, its quantity has not increased over time, leading to difficulties in model comprehension and selection among users. Our study sheds light on new challenges in reusing PTMs that were not reported before and we provide recommendations for various stakeholders involved in PTM reuse.

{{</citation>}}


## cs.DL (2)



### (107/123) Tweets to Citations: Unveiling the Impact of Social Media Influencers on AI Research Visibility (Iain Xie Weissburg et al., 2024)

{{<citation>}}

Iain Xie Weissburg, Mehir Arora, Liangming Pan, William Yang Wang. (2024)  
**Tweets to Citations: Unveiling the Impact of Social Media Influencers on AI Research Visibility**  

---
Primary Category: cs.DL  
Categories: cs-AI, cs-CL, cs-CV, cs-DL, cs-LG, cs-SI, cs.DL  
Keywords: AI, Social Media  
[Paper Link](http://arxiv.org/abs/2401.13782v1)  

---


**ABSTRACT**  
As the number of accepted papers at AI and ML conferences reaches into the thousands, it has become unclear how researchers access and read research publications. In this paper, we investigate the role of social media influencers in enhancing the visibility of machine learning research, particularly the citation counts of papers they share. We have compiled a comprehensive dataset of over 8,000 papers, spanning tweets from December 2018 to October 2023, alongside 1:1 matched controls based on publication year, venue, and abstract topics. Our analysis reveals a significant increase in citations for papers endorsed by these influencers, with median citation counts 2-3 times higher than those of the control group. Additionally, the study delves into the geographic, gender, and institutional diversity of highlighted authors. These findings highlight the expanding influence of social media in scholarly communication and underscore the importance of an evolving ecosystem in today's digital academic landscape.

{{</citation>}}


### (108/123) Organizing Scientific Knowledge From Energy System Research Using the Open Research Knowledge Graph (Oliver Karras et al., 2024)

{{<citation>}}

Oliver Karras, Jan Göpfert, Patrick Kuckertz, Tristan Pelser, Sören Auer. (2024)  
**Organizing Scientific Knowledge From Energy System Research Using the Open Research Knowledge Graph**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2401.13365v1)  

---


**ABSTRACT**  
Engineering sciences, such as energy system research, play an important role in developing solutions to technical, environmental, economic, and social challenges of our modern society. In this context, the transformation of energy systems into climate-neutral systems is one of the key strategies for mitigating climate change. For the transformation of energy systems, engineers model, simulate and analyze scenarios and transformation pathways to initiate debates about possible transformation strategies. For these debates and research in general, all steps of the research process must be traceable to guarantee the trustworthiness of published results, avoid redundancies, and ensure their social acceptance. However, the analysis of energy systems is an interdisciplinary field as the investigations of large, complex energy systems often require the use of different software applications and large amounts of heterogeneous data. Engineers must therefore communicate, understand, and (re)use heterogeneous scientific knowledge and data. Although the importance of FAIR scientific knowledge and data in the engineering sciences and energy system research is increasing, little research has been conducted on this topic. When it comes to publishing scientific knowledge and data from publications, software, and datasets (such as models, scenarios, and simulations) openly available and transparent, energy system research lags behind other research domains. According to Schmitt et al. and Nie{\ss}e et al., engineers need technical support in the form of infrastructures, services, and terminologies to improve communication, understanding, and (re)use of scientific knowledge and data.

{{</citation>}}


## math.NA (1)



### (109/123) Multi-Function Multi-Way Analog Technology for Sustainable Machine Intelligence Computation (Vassilis Kalantzis et al., 2024)

{{<citation>}}

Vassilis Kalantzis, Mark S. Squillante, Shashanka Ubaru, Tayfun Gokmen, Chai Wah Wu, Anshul Gupta, Haim Avron, Tomasz Nowicki, Malte Rasch, Murat Onen, Vanessa Lopez Marrero, Effendi Leobandung, Yasuteru Kohda, Wilfried Haensch, Lior Horesh. (2024)  
**Multi-Function Multi-Way Analog Technology for Sustainable Machine Intelligence Computation**  

---
Primary Category: math.NA  
Categories: 65F10, C3, G1, G-1-3, cs-ET, cs-NA, math-NA, math.NA  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13754v1)  

---


**ABSTRACT**  
Numerical computation is essential to many areas of artificial intelligence (AI), whose computing demands continue to grow dramatically, yet their continued scaling is jeopardized by the slowdown in Moore's law. Multi-function multi-way analog (MFMWA) technology, a computing architecture comprising arrays of memristors supporting in-memory computation of matrix operations, can offer tremendous improvements in computation and energy, but at the expense of inherent unpredictability and noise. We devise novel randomized algorithms tailored to MFMWA architectures that mitigate the detrimental impact of imperfect analog computations while realizing their potential benefits across various areas of AI, such as applications in computer vision. Through analysis, measurements from analog devices, and simulations of larger systems, we demonstrate orders of magnitude reduction in both computation and energy with accuracy similar to digital computers.

{{</citation>}}


## cs.DC (2)



### (110/123) Enabling Seamless Data Security, Consensus, and Trading in Vehicular Networks (Emanuel Vieira et al., 2024)

{{<citation>}}

Emanuel Vieira, João Almeida, Joaquim Ferreira, Paulo C. Bartolomeu. (2024)  
**Enabling Seamless Data Security, Consensus, and Trading in Vehicular Networks**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.13630v1)  

---


**ABSTRACT**  
Cooperative driving is an emerging paradigm to enhance the safety and efficiency of autonomous vehicles. To ensure successful cooperation, road users must reach a consensus for making collective decisions, while recording vehicular data to analyze and address failures related to such agreements. This data has the potential to provide valuable insights into various vehicular events, while also potentially improving accountability measures. Furthermore, vehicles may benefit from the ability to negotiate and trade services among themselves, adding value to the cooperative driving framework. However, the majority of proposed systems aiming to ensure data security, consensus, or service trading, lack efficient and thoroughly validated mechanisms that consider the distinctive characteristics of vehicular networks. These limitations are amplified by a dependency on the centralized support provided by the infrastructure. Furthermore, corresponding mechanisms must diligently address security concerns, especially regarding potential malicious or misbehaving nodes, while also considering inherent constraints of the wireless medium. We introduce the Verifiable Event Extension (VEE), an applicational extension designed for Intelligent Transportation System (ITS) messages. The VEE operates seamlessly with any existing standardized vehicular communications protocol, addressing crucial aspects of data security, consensus, and trading with minimal overhead. To achieve this, we employ blockchain techniques, Byzantine fault tolerance (BFT) consensus protocols, and cryptocurrency-based mechanics. To assess our proposal's feasibility and lightweight nature, we employed a hardware-in-the-loop setup for analysis. Experimental results demonstrate the viability and efficiency of the VEE extension in overcoming the challenges posed by the distributed and opportunistic nature of wireless vehicular communications.

{{</citation>}}


### (111/123) A Big Data Architecture for Early Identification and Categorization of Dark Web Sites (Javier Pastor-Galindo et al., 2024)

{{<citation>}}

Javier Pastor-Galindo, Hông-Ân Sandlin, Félix Gómez Mármol, Gérôme Bovet, Gregorio Martínez Pérez. (2024)  
**A Big Data Architecture for Early Identification and Categorization of Dark Web Sites**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-IR, cs.DC  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2401.13320v1)  

---


**ABSTRACT**  
The dark web has become notorious for its association with illicit activities and there is a growing need for systems to automate the monitoring of this space. This paper proposes an end-to-end scalable architecture for the early identification of new Tor sites and the daily analysis of their content. The solution is built using an Open Source Big Data stack for data serving with Kubernetes, Kafka, Kubeflow, and MinIO, continuously discovering onion addresses in different sources (threat intelligence, code repositories, web-Tor gateways, and Tor repositories), downloading the HTML from Tor and deduplicating the content using MinHash LSH, and categorizing with the BERTopic modeling (SBERT embedding, UMAP dimensionality reduction, HDBSCAN document clustering and c-TF-IDF topic keywords). In 93 days, the system identified 80,049 onion services and characterized 90% of them, addressing the challenge of Tor volatility. A disproportionate amount of repeated content is found, with only 6.1% unique sites. From the HTML files of the dark sites, 31 different low-topics are extracted, manually labeled, and grouped into 11 high-level topics. The five most popular included sexual and violent content, repositories, search engines, carding, cryptocurrencies, and marketplaces. During the experiments, we identified 14 sites with 13,946 clones that shared a suspiciously similar mirroring rate per day, suggesting an extensive common phishing network. Among the related works, this study is the most representative characterization of onion services based on topics to date.

{{</citation>}}


## cs.CR (2)



### (112/123) Securing the Invisible Thread: A Comprehensive Analysis of BLE Tracker Security in Apple AirTags and Samsung SmartTags (Hosam Alamleh et al., 2024)

{{<citation>}}

Hosam Alamleh, Michael Gogarty, David Ruddell, Ali Abdullah S. AlQahtani. (2024)  
**Securing the Invisible Thread: A Comprehensive Analysis of BLE Tracker Security in Apple AirTags and Samsung SmartTags**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2401.13584v1)  

---


**ABSTRACT**  
This study presents an in-depth analysis of the security landscape in Bluetooth Low Energy (BLE) tracking systems, with a particular emphasis on Apple AirTags and Samsung SmartTags, including their cryptographic frameworks. Our investigation traverses a wide spectrum of attack vectors such as physical tampering, firmware exploitation, signal spoofing, eavesdropping, jamming, app security flaws, Bluetooth security weaknesses, location spoofing, threats to owner devices, and cloud-related vulnerabilities. Moreover, we delve into the security implications of the cryptographic methods utilized in these systems. Our findings reveal that while BLE trackers like AirTags and SmartTags offer substantial utility, they also pose significant security risks. Notably, Apple's approach, which prioritizes user privacy by removing intermediaries, inadvertently leads to device authentication challenges, evidenced by successful AirTag spoofing instances. Conversely, Samsung SmartTags, designed to thwart beacon spoofing, raise critical concerns about cloud security and user privacy. Our analysis also highlights the constraints faced by these devices due to their design focus on battery life conservation, particularly the absence of secure boot processes, which leaves them susceptible to OS modification and a range of potential attacks. The paper concludes with insights into the anticipated evolution of these tracking systems. We predict that future enhancements will likely focus on bolstering security features, especially as these devices become increasingly integrated into the broader IoT ecosystem and face evolving privacy regulations. This shift is imperative to address the intricate balance between functionality and security in next-generation BLE tracking systems.

{{</citation>}}


### (113/123) A Repository-Level Dataset For Detecting, Classifying and Repairing Software Vulnerabilities (Xinchen Wang et al., 2024)

{{<citation>}}

Xinchen Wang, Ruida Hu, Cuiyun Gao, Xin-Cheng Wen, Yujia Chen, Qing Liao. (2024)  
**A Repository-Level Dataset For Detecting, Classifying and Repairing Software Vulnerabilities**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.13169v1)  

---


**ABSTRACT**  
Open-Source Software (OSS) vulnerabilities bring great challenges to the software security and pose potential risks to our society. Enormous efforts have been devoted into automated vulnerability detection, among which deep learning (DL)-based approaches have proven to be the most effective. However, the current labeled data present the following limitations: (1) \textbf{Tangled Patches}: Developers may submit code changes unrelated to vulnerability fixes within patches, leading to tangled patches. (2) \textbf{Lacking Inter-procedural Vulnerabilities}: The existing vulnerability datasets typically contain function-level and file-level vulnerabilities, ignoring the relations between functions, thus rendering the approaches unable to detect the inter-procedural vulnerabilities. (3) \textbf{Outdated Patches}: The existing datasets usually contain outdated patches, which may bias the model during training.   To address the above limitations, in this paper, we propose an automated data collection framework and construct the first repository-level high-quality vulnerability dataset named \textbf{ReposVul}. The proposed framework mainly contains three modules: (1) A vulnerability untangling module, aiming at distinguishing vulnerability-fixing related code changes from tangled patches, in which the Large Language Models (LLMs) and static analysis tools are jointly employed. (2) A multi-granularity dependency extraction module, aiming at capturing the inter-procedural call relationships of vulnerabilities, in which we construct multiple-granularity information for each vulnerability patch, including repository-level, file-level, function-level, and line-level. (3) A trace-based filtering module, aiming at filtering the outdated patches, which leverages the file path trace-based filter and commit time trace-based filter to construct an up-to-date dataset.

{{</citation>}}


## cs.CE (1)



### (114/123) Guided Diffusion for Fast Inverse Design of Density-based Mechanical Metamaterials (Yanyan Yang et al., 2024)

{{<citation>}}

Yanyan Yang, Lili Wang, Xiaoya Zhai, Kai Chen, Wenming Wu, Yunkai Zhao, Ligang Liu, Xiao-Ming Fu. (2024)  
**Guided Diffusion for Fast Inverse Design of Density-based Mechanical Metamaterials**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-LG, cs.CE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13570v1)  

---


**ABSTRACT**  
Mechanical metamaterial is a synthetic material that can possess extraordinary physical characteristics, such as abnormal elasticity, stiffness, and stability, by carefully designing its internal structure. To make metamaterials contain delicate local structures with unique mechanical properties, it is a potential method to represent them through high-resolution voxels. However, it brings a substantial computational burden. To this end, this paper proposes a fast inverse design method, whose core is an advanced deep generative AI algorithm, to generate voxel-based mechanical metamaterials. Specifically, we use the self-conditioned diffusion model, capable of generating a microstructure with a resolution of $128^3$ to approach the specified homogenized tensor matrix in just 3 seconds. Accordingly, this rapid reverse design tool facilitates the exploration of extreme metamaterials, the sequence interpolation in metamaterials, and the generation of diverse microstructures for multi-scale design. This flexible and adaptive generative tool is of great value in structural engineering or other mechanical systems and can stimulate more subsequent research.

{{</citation>}}


## hep-ph (1)



### (115/123) Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models (Lukas Heinrich et al., 2024)

{{<citation>}}

Lukas Heinrich, Tobias Golling, Michael Kagan, Samuel Klein, Matthew Leigh, Margarita Osadchy, John Andrew Raine. (2024)  
**Masked Particle Modeling on Sets: Towards Self-Supervised High Energy Physics Foundation Models**  

---
Primary Category: hep-ph  
Categories: cs-LG, hep-ex, hep-ph, hep-ph, physics-data-an  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2401.13537v2)  

---


**ABSTRACT**  
We propose masked particle modeling (MPM) as a self-supervised method for learning generic, transferable, and reusable representations on unordered sets of inputs for use in high energy physics (HEP) scientific data. This work provides a novel scheme to perform masked modeling based pre-training to learn permutation invariant functions on sets. More generally, this work provides a step towards building large foundation models for HEP that can be generically pre-trained with self-supervised learning and later fine-tuned for a variety of down-stream tasks. In MPM, particles in a set are masked and the training objective is to recover their identity, as defined by a discretized token representation of a pre-trained vector quantized variational autoencoder. We study the efficacy of the method in samples of high energy jets at collider physics experiments, including studies on the impact of discretization, permutation invariance, and ordering. We also study the fine-tuning capability of the model, showing that it can be adapted to tasks such as supervised and weakly supervised jet classification, and that the model can transfer efficiently with small fine-tuning data sets to new classes and new data domains.

{{</citation>}}


## q-fin.RM (1)



### (116/123) Real-time Risk Metrics for Programmatic Stablecoin Crypto Asset-Liability Management (CALM) (Marcel Bluhm et al., 2024)

{{<citation>}}

Marcel Bluhm, Adrian Cachinero Vasiljević, Sébastien Derivaux, Søren Terp Hørlück Jessen. (2024)  
**Real-time Risk Metrics for Programmatic Stablecoin Crypto Asset-Liability Management (CALM)**  

---
Primary Category: q-fin.RM  
Categories: cs-CR, q-fin-GN, q-fin-RM, q-fin.RM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13399v1)  

---


**ABSTRACT**  
Stablecoins have turned out to be the "killer" use case of the growing digital asset space. However, risk management frameworks, including regulatory ones, have been largely absent. In this paper, we address the critical question of measuring and managing risk in stablecoin protocols, which operate on public blockchain infrastructure. The on-chain environment makes it possible to monitor risk and automate its management via transparent smart-contracts in real-time. We propose two risk metrics covering capitalization and liquidity of stablecoin protocols. We then explore in a case-study type analysis how our risk management framework can be applied to DAI, the biggest decentralized stablecoin by market capitalisation to-date, governed by MakerDAO. Based on our findings, we recommend that the protocol explores implementing automatic capital buffer adjustments and dynamic maturity gap matching. Our analysis demonstrates the practical benefits for scalable (prudential) risk management stemming from real-time availability of high-quality, granular, tamper-resistant on-chain data in the digital asset space. We name this approach Crypto Asset-Liability Management (CALM).

{{</citation>}}


## cs.GT (1)



### (117/123) SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification (Patrick Kolpaczki et al., 2024)

{{<citation>}}

Patrick Kolpaczki, Maximilian Muschalik, Fabian Fumagalli, Barbara Hammer, Eyke Hüllermeier. (2024)  
**SVARM-IQ: Efficient Approximation of Any-order Shapley Interactions through Stratification**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13371v2)  

---


**ABSTRACT**  
Addressing the limitations of individual attribution scores via the Shapley value (SV), the field of explainable AI (XAI) has recently explored intricate interactions of features or data points. In particular, extensions of the SV, such as the Shapley Interaction Index (SII), have been proposed as a measure to still benefit from the axiomatic basis of the SV. However, similar to the SV, their exact computation remains computationally prohibitive. Hence, we propose with SVARM-IQ a sampling-based approach to efficiently approximate Shapley-based interaction indices of any order. SVARM-IQ can be applied to a broad class of interaction indices, including the SII, by leveraging a novel stratified representation. We provide non-asymptotic theoretical guarantees on its approximation quality and empirically demonstrate that SVARM-IQ achieves state-of-the-art estimation results in practical XAI scenarios on different model classes and application domains.

{{</citation>}}


## cs.RO (1)



### (118/123) TraKDis: A Transformer-based Knowledge Distillation Approach for Visual Reinforcement Learning with Application to Cloth Manipulation (Wei Chen et al., 2024)

{{<citation>}}

Wei Chen, Nicolas Rojas. (2024)  
**TraKDis: A Transformer-based Knowledge Distillation Approach for Visual Reinforcement Learning with Application to Cloth Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Knowledge Distillation, Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2401.13362v1)  

---


**ABSTRACT**  
Approaching robotic cloth manipulation using reinforcement learning based on visual feedback is appealing as robot perception and control can be learned simultaneously. However, major challenges result due to the intricate dynamics of cloth and the high dimensionality of the corresponding states, what shadows the practicality of the idea. To tackle these issues, we propose TraKDis, a novel Transformer-based Knowledge Distillation approach that decomposes the visual reinforcement learning problem into two distinct stages. In the first stage, a privileged agent is trained, which possesses complete knowledge of the cloth state information. This privileged agent acts as a teacher, providing valuable guidance and training signals for subsequent stages. The second stage involves a knowledge distillation procedure, where the knowledge acquired by the privileged agent is transferred to a vision-based agent by leveraging pre-trained state estimation and weight initialization. TraKDis demonstrates better performance when compared to state-of-the-art RL techniques, showing a higher performance of 21.9%, 13.8%, and 8.3% in cloth folding tasks in simulation. Furthermore, to validate robustness, we evaluate the agent in a noisy environment; the results indicate its ability to handle and adapt to environmental uncertainties effectively. Real robot experiments are also conducted to showcase the efficiency of our method in real-world scenarios.

{{</citation>}}


## cs.OS (1)



### (119/123) Characterizing Network Requirements for GPU API Remoting in AI Applications (Tianxia Wang et al., 2024)

{{<citation>}}

Tianxia Wang, Zhuofu Chen, Xingda Wei, Jinyu Gu, Rong Chen, Haibo Chen. (2024)  
**Characterizing Network Requirements for GPU API Remoting in AI Applications**  

---
Primary Category: cs.OS  
Categories: cs-NI, cs-OS, cs.OS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13354v1)  

---


**ABSTRACT**  
GPU remoting is a promising technique for supporting AI applications. Networking plays a key role in enabling remoting. However, for efficient remoting, the network requirements in terms of latency and bandwidth are unknown. In this paper, we take a GPU-centric approach to derive the minimum latency and bandwidth requirements for GPU remoting, while ensuring no (or little) performance degradation for AI applications. Our study including theoretical model demonstrates that, with careful remoting design, unmodified AI applications can run on the remoting setup using commodity networking hardware without any overhead or even with better performance, with low network demands.

{{</citation>}}


## cs.NI (1)



### (120/123) Past, Present, Future: A Comprehensive Exploration of AI Use Cases in the UMBRELLA IoT Testbed (Peizheng Li et al., 2024)

{{<citation>}}

Peizheng Li, Ioannis Mavromatis, Aftab Khan. (2024)  
**Past, Present, Future: A Comprehensive Exploration of AI Use Cases in the UMBRELLA IoT Testbed**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.13346v1)  

---


**ABSTRACT**  
UMBRELLA is a large-scale, open-access Internet of Things (IoT) ecosystem incorporating over 200 multi-sensor multi-wireless nodes, 20 collaborative robots, and edge-intelligence-enabled devices. This paper provides a guide to the implemented and prospective artificial intelligence (AI) capabilities of UMBRELLA in real-world IoT systems. Four existing UMBRELLA applications are presented in detail: 1) An automated streetlight monitoring for detecting issues and triggering maintenance alerts; 2) A Digital twin of building environments providing enhanced air quality sensing with reduced cost; 3) A large-scale Federated Learning framework for reducing communication overhead; and 4) An intrusion detection for containerised applications identifying malicious activities. Additionally, the potential of UMBRELLA is outlined for future smart city and multi-robot crowdsensing applications enhanced by semantic communications and multi-agent planning. Finally, to realise the above use-cases we discuss the need for a tailored MLOps platform to automate UMBRELLA model pipelines and establish trust.

{{</citation>}}


## cs.AR (1)



### (121/123) SpecLLM: Exploring Generation and Review of VLSI Design Specification with Large Language Model (Mengming Li et al., 2024)

{{<citation>}}

Mengming Li, Wenji Fang, Qijun Zhang, Zhiyao Xie. (2024)  
**SpecLLM: Exploring Generation and Review of VLSI Design Specification with Large Language Model**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.13266v1)  

---


**ABSTRACT**  
The development of architecture specifications is an initial and fundamental stage of the integrated circuit (IC) design process. Traditionally, architecture specifications are crafted by experienced chip architects, a process that is not only time-consuming but also error-prone. Mistakes in these specifications may significantly affect subsequent stages of chip design. Despite the presence of advanced electronic design automation (EDA) tools, effective solutions to these specification-related challenges remain scarce. Since writing architecture specifications is naturally a natural language processing (NLP) task, this paper pioneers the automation of architecture specification development with the advanced capabilities of large language models (LLMs). Leveraging our definition and dataset, we explore the application of LLMs in two key aspects of architecture specification development: (1) Generating architecture specifications, which includes both writing specifications from scratch and converting RTL code into detailed specifications. (2) Reviewing existing architecture specifications. We got promising results indicating that LLMs may revolutionize how these critical specification documents are developed in IC design nowadays. By reducing the effort required, LLMs open up new possibilities for efficiency and accuracy in this crucial aspect of chip design.

{{</citation>}}


## eess.IV (1)



### (122/123) Segment Any Cell: A SAM-based Auto-prompting Fine-tuning Framework for Nuclei Segmentation (Saiyang Na et al., 2024)

{{<citation>}}

Saiyang Na, Yuzhi Guo, Feng Jiang, Hehuan Ma, Junzhou Huang. (2024)  
**Segment Any Cell: A SAM-based Auto-prompting Fine-tuning Framework for Nuclei Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Attention, BERT, ChatGPT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2401.13220v1)  

---


**ABSTRACT**  
In the rapidly evolving field of AI research, foundational models like BERT and GPT have significantly advanced language and vision tasks. The advent of pretrain-prompting models such as ChatGPT and Segmentation Anything Model (SAM) has further revolutionized image segmentation. However, their applications in specialized areas, particularly in nuclei segmentation within medical imaging, reveal a key challenge: the generation of high-quality, informative prompts is as crucial as applying state-of-the-art (SOTA) fine-tuning techniques on foundation models. To address this, we introduce Segment Any Cell (SAC), an innovative framework that enhances SAM specifically for nuclei segmentation. SAC integrates a Low-Rank Adaptation (LoRA) within the attention layer of the Transformer to improve the fine-tuning process, outperforming existing SOTA methods. It also introduces an innovative auto-prompt generator that produces effective prompts to guide segmentation, a critical factor in handling the complexities of nuclei segmentation in biomedical imaging. Our extensive experiments demonstrate the superiority of SAC in nuclei segmentation tasks, proving its effectiveness as a tool for pathologists and researchers. Our contributions include a novel prompt generation strategy, automated adaptability for diverse segmentation tasks, the innovative application of Low-Rank Attention Adaptation in SAM, and a versatile framework for semantic segmentation challenges.

{{</citation>}}


## q-bio.GN (1)



### (123/123) TEPI: Taxonomy-aware Embedding and Pseudo-Imaging for Scarcely-labeled Zero-shot Genome Classification (Sathyanarayanan Aakur et al., 2024)

{{<citation>}}

Sathyanarayanan Aakur, Vishalini R. Laguduva, Priyadharsini Ramamurthy, Akhilesh Ramachandran. (2024)  
**TEPI: Taxonomy-aware Embedding and Pseudo-Imaging for Scarcely-labeled Zero-shot Genome Classification**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, cs-LG, q-bio-GN, q-bio.GN  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.13219v1)  

---


**ABSTRACT**  
A species' genetic code or genome encodes valuable evolutionary, biological, and phylogenetic information that aids in species recognition, taxonomic classification, and understanding genetic predispositions like drug resistance and virulence. However, the vast number of potential species poses significant challenges in developing a general-purpose whole genome classification tool. Traditional bioinformatics tools have made notable progress but lack scalability and are computationally expensive. Machine learning-based frameworks show promise but must address the issue of large classification vocabularies with long-tail distributions. In this study, we propose addressing this problem through zero-shot learning using TEPI, Taxonomy-aware Embedding and Pseudo-Imaging. We represent each genome as pseudo-images and map them to a taxonomy-aware embedding space for reasoning and classification. This embedding space captures compositional and phylogenetic relationships of species, enabling predictions in extensive search spaces. We evaluate TEPI using two rigorous zero-shot settings and demonstrate its generalization capabilities qualitatively on curated, large-scale, publicly sourced data.

{{</citation>}}
