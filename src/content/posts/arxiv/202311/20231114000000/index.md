---
draft: false
title: "arXiv @ 2023.11.14"
date: 2023-11-14
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.14"
    identifier: arxiv_20231114
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.SE (1)](#csse-1)
- [cs.CL (18)](#cscl-18)
- [cs.AI (2)](#csai-2)
- [cs.CV (11)](#cscv-11)
- [cs.LG (14)](#cslg-14)
- [cs.CY (2)](#cscy-2)
- [cs.RO (3)](#csro-3)
- [eess.SY (1)](#eesssy-1)
- [cs.AR (1)](#csar-1)
- [cs.SI (1)](#cssi-1)
- [cs.HC (3)](#cshc-3)
- [cs.IR (2)](#csir-2)
- [eess.IV (1)](#eessiv-1)
- [cs.NI (1)](#csni-1)
- [cs.IT (1)](#csit-1)
- [physics.chem-ph (1)](#physicschem-ph-1)

## cs.SE (1)



### (1/63) Creating a Discipline-specific Commons for Infectious Disease Epidemiology (Michael M. Wagner et al., 2023)

{{<citation>}}

Michael M. Wagner, William Hogan, John Levander, Adam Darr, Matt Diller, Max Sibilla, Alexander T. Loiacono. Terence Sperringer, Jr., Shawn T. Brown. (2023)  
**Creating a Discipline-specific Commons for Infectious Disease Epidemiology**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06989v1)  

---


**ABSTRACT**  
Objective: To create a commons for infectious disease (ID) epidemiology in which epidemiologists, public health officers, data producers, and software developers can not only share data and software, but receive assistance in improving their interoperability. Materials and Methods: We represented 586 datasets, 54 software, and 24 data formats in OWL 2 and then used logical queries to infer potentially interoperable combinations of software and datasets, as well as statistics about the FAIRness of the collection. We represented the objects in DATS 2.2 and a software metadata schema of our own design. We used these representations as the basis for the Content, Search, FAIR-o-meter, and Workflow pages that constitute the MIDAS Digital Commons. Results: Interoperability was limited by lack of standardization of input and output formats of software. When formats existed, they were human-readable specifications (22/24; 92%); only 3 formats (13%) had machine-readable specifications. Nevertheless, logical search of a triple store based on named data formats was able to identify scores of potentially interoperable combinations of software and datasets. Discussion: We improved the findability and availability of a sample of software and datasets and developed metrics for assessing interoperability. The barriers to interoperability included poor documentation of software input/output formats and little attention to standardization of most types of data in this field. Conclusion: Centralizing and formalizing the representation of digital objects within a commons promotes FAIRness, enables its measurement over time and the identification of potentially interoperable combinations of data and software.

{{</citation>}}


## cs.CL (18)



### (2/63) SELF-EXPLAIN: Teaching Large Language Models to Reason Complex Questions by Themselves (Jiachen Zhao et al., 2023)

{{<citation>}}

Jiachen Zhao, Zonghai Yao, Zhichao Yang, Hong Yu. (2023)  
**SELF-EXPLAIN: Teaching Large Language Models to Reason Complex Questions by Themselves**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06985v1)  

---


**ABSTRACT**  
Large language models (LLMs) can generate intermediate reasoning steps. To elicit the reliable reasoning, the common practice is to employ few-shot chain-of-thought prompting, where several in-context demonstrations for reasoning are prepended to the question. However, such chain-of-thought examples are expensive to craft, especially for professional domains, and can have high variance depending on human annotators. Therefore, this work investigates whether LLMs can teach themselves to reason without human-crafted demonstrations. We propose SELF-EXPLAIN to generate CoT examples by LLMs inspired by "encoding specificity" in human memory retrieval. We find using self-explanations makes LLMs more confident, more calibrated and less biased when answering complex questions. Moreover, we find prompting with self-explanations can even significantly outperform using human-crafted CoTs on several complex question answering dataset.

{{</citation>}}


### (3/63) Flames: Benchmarking Value Alignment of Chinese Large Language Models (Kexin Huang et al., 2023)

{{<citation>}}

Kexin Huang, Xiangyang Liu, Qianyu Guo, Tianxiang Sun, Jiawei Sun, Yaru Wang, Zeyang Zhou, Yixu Wang, Yan Teng, Xipeng Qiu, Yingchun Wang, Dahua Lin. (2023)  
**Flames: Benchmarking Value Alignment of Chinese Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06899v1)  

---


**ABSTRACT**  
The widespread adoption of large language models (LLMs) across various regions underscores the urgent need to evaluate their alignment with human values. Current benchmarks, however, fall short of effectively uncovering safety vulnerabilities in LLMs. Despite numerous models achieving high scores and 'topping the chart' in these evaluations, there is still a significant gap in LLMs' deeper alignment with human values and achieving genuine harmlessness. To this end, this paper proposes the first highly adversarial benchmark named Flames, consisting of 2,251 manually crafted prompts, ~18.7K model responses with fine-grained annotations, and a specified scorer. Our framework encompasses both common harmlessness principles, such as fairness, safety, legality, and data protection, and a unique morality dimension that integrates specific Chinese values such as harmony. Based on the framework, we carefully design adversarial prompts that incorporate complex scenarios and jailbreaking methods, mostly with implicit malice. By prompting mainstream LLMs with such adversarially constructed prompts, we obtain model responses, which are then rigorously annotated for evaluation. Our findings indicate that all the evaluated LLMs demonstrate relatively poor performance on Flames, particularly in the safety and fairness dimensions. Claude emerges as the best-performing model overall, but with its harmless rate being only 63.08% while GPT-4 only scores 39.04%. The complexity of Flames has far exceeded existing benchmarks, setting a new challenge for contemporary LLMs and highlighting the need for further alignment of LLMs. To efficiently evaluate new models on the benchmark, we develop a specified scorer capable of scoring LLMs across multiple dimensions, achieving an accuracy of 77.4%. The Flames Benchmark is publicly available on https://github.com/AIFlames/Flames.

{{</citation>}}


### (4/63) Retrieval and Generative Approaches for a Pregnancy Chatbot in Nepali with Stemmed and Non-Stemmed Data : A Comparative Study (Sujan Poudel et al., 2023)

{{<citation>}}

Sujan Poudel, Nabin Ghimire, Bipesh Subedi, Saugat Singh. (2023)  
**Retrieval and Generative Approaches for a Pregnancy Chatbot in Nepali with Stemmed and Non-Stemmed Data : A Comparative Study**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, BLEU, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.06898v1)  

---


**ABSTRACT**  
The field of Natural Language Processing which involves the use of artificial intelligence to support human languages has seen tremendous growth due to its high-quality features. Its applications such as language translation, chatbots, virtual assistants, search autocomplete, and autocorrect are widely used in various domains including healthcare, advertising, customer service, and target advertising. To provide pregnancy-related information a health domain chatbot has been proposed and this work explores two different NLP-based approaches for developing the chatbot. The first approach is a multiclass classification-based retrieval approach using BERTbased multilingual BERT and multilingual DistilBERT while the other approach employs a transformer-based generative chatbot for pregnancy-related information. The performance of both stemmed and non-stemmed datasets in Nepali language has been analyzed for each approach. The experimented results indicate that BERT-based pre-trained models perform well on non-stemmed data whereas scratch transformer models have better performance on stemmed data. Among the models tested the DistilBERT model achieved the highest training and validation accuracy and testing accuracy of 0.9165 on the retrieval-based model architecture implementation on the non-stemmed dataset. Similarly, in the generative approach architecture implementation with transformer 1 gram BLEU and 2 gram BLEU scores of 0.3570 and 0.1413 respectively were achieved.

{{</citation>}}


### (5/63) Can Large Language Models Augment a Biomedical Ontology with missing Concepts and Relations? (Antonio Zaitoun et al., 2023)

{{<citation>}}

Antonio Zaitoun, Tomer Sagi, Szymon Wilk, Mor Peleg. (2023)  
**Can Large Language Models Augment a Biomedical Ontology with missing Concepts and Relations?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06858v1)  

---


**ABSTRACT**  
Ontologies play a crucial role in organizing and representing knowledge. However, even current ontologies do not encompass all relevant concepts and relationships. Here, we explore the potential of large language models (LLM) to expand an existing ontology in a semi-automated fashion. We demonstrate our approach on the biomedical ontology SNOMED-CT utilizing semantic relation types from the widely used UMLS semantic network. We propose a method that uses conversational interactions with an LLM to analyze clinical practice guidelines (CPGs) and detect the relationships among the new medical concepts that are not present in SNOMED-CT. Our initial experimentation with the conversational prompts yielded promising preliminary results given a manually generated gold standard, directing our future potential improvements.

{{</citation>}}


### (6/63) Automatic Textual Normalization for Hate Speech Detection (Anh Thi-Hoang Nguyen et al., 2023)

{{<citation>}}

Anh Thi-Hoang Nguyen, Dung Ha Nguyen, Nguyet Thi Nguyen, Khanh Thanh-Duy Ho, Kiet Van Nguyen. (2023)  
**Automatic Textual Normalization for Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Hate Speech Detection, NLP, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2311.06851v2)  

---


**ABSTRACT**  
Social media data is a valuable resource for research, yet it contains a wide range of non-standard words (NSW). These irregularities hinder the effective operation of NLP tools. Current state-of-the-art methods for the Vietnamese language address this issue as a problem of lexical normalization, involving the creation of manual rules or the implementation of multi-staged deep learning frameworks, which necessitate extensive efforts to craft intricate rules. In contrast, our approach is straightforward, employing solely a sequence-to-sequence (Seq2Seq) model. In this research, we provide a dataset for textual normalization, comprising 2,181 human-annotated comments with an inter-annotator agreement of 0.9014. By leveraging the Seq2Seq model for textual normalization, our results reveal that the accuracy achieved falls slightly short of 70%. Nevertheless, textual normalization enhances the accuracy of the Hate Speech Detection (HSD) task by approximately 2%, demonstrating its potential to improve the performance of complex NLP tasks. Our dataset is accessible for research purposes.

{{</citation>}}


### (7/63) GIELLM: Japanese General Information Extraction Large Language Model Utilizing Mutual Reinforcement Effect (Chengguang Gan et al., 2023)

{{<citation>}}

Chengguang Gan, Qinghao Zhang, Tatsunori Mori. (2023)  
**GIELLM: Japanese General Information Extraction Large Language Model Utilizing Mutual Reinforcement Effect**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Event Extraction, GPT, GPT-3.5, Information Extraction, Language Model, Named Entity Recognition, Relation Extraction, Sentiment Analysis, Text Classification  
[Paper Link](http://arxiv.org/abs/2311.06838v1)  

---


**ABSTRACT**  
Information Extraction (IE) stands as a cornerstone in natural language processing, traditionally segmented into distinct sub-tasks. The advent of Large Language Models (LLMs) heralds a paradigm shift, suggesting the feasibility of a singular model addressing multiple IE subtasks. In this vein, we introduce the General Information Extraction Large Language Model (GIELLM), which integrates text Classification, Sentiment Analysis, Named Entity Recognition, Relation Extraction, and Event Extraction using a uniform input-output schema. This innovation marks the first instance of a model simultaneously handling such a diverse array of IE subtasks. Notably, the GIELLM leverages the Mutual Reinforcement Effect (MRE), enhancing performance in integrated tasks compared to their isolated counterparts. Our experiments demonstrate State-of-the-Art (SOTA) results in five out of six Japanese mixed datasets, significantly surpassing GPT-3.5-Turbo. Further, an independent evaluation using the novel Text Classification Relation and Event Extraction(TCREE) dataset corroborates the synergistic advantages of MRE in text and word classification. This breakthrough paves the way for most IE subtasks to be subsumed under a singular LLM framework. Specialized fine-tune task-specific models are no longer needed.

{{</citation>}}


### (8/63) Evaluation of GPT-4 for chest X-ray impression generation: A reader study on performance and perception (Sebastian Ziegelmayer et al., 2023)

{{<citation>}}

Sebastian Ziegelmayer, Alexander W. Marka, Nicolas Lenhart, Nadja Nehls, Stefan Reischl, Felix Harder, Andreas Sauter, Marcus Makowski, Markus Graf, Joshua Gawlitza. (2023)  
**Evaluation of GPT-4 for chest X-ray impression generation: A reader study on performance and perception**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.06815v1)  

---


**ABSTRACT**  
The remarkable generative capabilities of multimodal foundation models are currently being explored for a variety of applications. Generating radiological impressions is a challenging task that could significantly reduce the workload of radiologists. In our study we explored and analyzed the generative abilities of GPT-4 for Chest X-ray impression generation. To generate and evaluate impressions of chest X-rays based on different input modalities (image, text, text and image), a blinded radiological report was written for 25-cases of the publicly available NIH-dataset. GPT-4 was given image, finding section or both sequentially to generate an input dependent impression. In a blind randomized reading, 4-radiologists rated the impressions and were asked to classify the impression origin (Human, AI), providing justification for their decision. Lastly text model evaluation metrics and their correlation with the radiological score (summation of the 4 dimensions) was assessed. According to the radiological score, the human-written impression was rated highest, although not significantly different to text-based impressions. The automated evaluation metrics showed moderate to substantial correlations to the radiological score for the image impressions, however individual scores were highly divergent among inputs, indicating insufficient representation of radiological quality. Detection of AI-generated impressions varied by input and was 61% for text-based impressions. Impressions classified as AI-generated had significantly worse radiological scores even when written by a radiologist, indicating potential bias. Our study revealed significant discrepancies between a radiological assessment and common automatic evaluation metrics depending on the model input. The detection of AI-generated findings is subject to bias that highly rated impressions are perceived as human-written.

{{</citation>}}


### (9/63) On the Robustness of Question Rewriting Systems to Questions of Varying Hardness (Hai Ye et al., 2023)

{{<citation>}}

Hai Ye, Hwee Tou Ng, Wenjuan Han. (2023)  
**On the Robustness of Question Rewriting Systems to Questions of Varying Hardness**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.06807v1)  

---


**ABSTRACT**  
In conversational question answering (CQA), the task of question rewriting~(QR) in context aims to rewrite a context-dependent question into an equivalent self-contained question that gives the same answer. In this paper, we are interested in the robustness of a QR system to questions varying in rewriting hardness or difficulty. Since there is a lack of questions classified based on their rewriting hardness, we first propose a heuristic method to automatically classify questions into subsets of varying hardness, by measuring the discrepancy between a question and its rewrite. To find out what makes questions hard or easy for rewriting, we then conduct a human evaluation to annotate the rewriting hardness of questions. Finally, to enhance the robustness of QR systems to questions of varying hardness, we propose a novel learning framework for QR that first trains a QR model independently on each subset of questions of a certain level of hardness, then combines these QR models as one joint model for inference. Experimental results on two datasets show that our framework improves the overall performance compared to the baselines.

{{</citation>}}


### (10/63) Learning Globally Optimized Language Structure via Adversarial Training (Xuwang Yin, 2023)

{{<citation>}}

Xuwang Yin. (2023)  
**Learning Globally Optimized Language Structure via Adversarial Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2311.06771v1)  

---


**ABSTRACT**  
Recent work has explored integrating autoregressive language models with energy-based models (EBMs) to enhance text generation capabilities. However, learning effective EBMs for text is challenged by the discrete nature of language. This work proposes an adversarial training strategy to address limitations in prior efforts. Specifically, an iterative adversarial attack algorithm is presented to generate negative samples for training the EBM by perturbing text from the autoregressive model. This aims to enable the EBM to suppress spurious modes outside the support of the data distribution. Experiments on an arithmetic sequence generation task demonstrate that the proposed adversarial training approach can substantially enhance the quality of generated sequences compared to prior methods. The results highlight the promise of adversarial techniques to improve discrete EBM training. Key contributions include: (1) an adversarial attack strategy tailored to text to generate negative samples, circumventing MCMC limitations; (2) an adversarial training algorithm for EBMs leveraging these attacks; (3) empirical validation of performance improvements on a sequence generation task.

{{</citation>}}


### (11/63) Learning Knowledge-Enhanced Contextual Language Representations for Domain Natural Language Understanding (Ruyao Xu et al., 2023)

{{<citation>}}

Ruyao Xu, Taolin Zhang, Chengyu Wang, Zhongjie Duan, Cen Chen, Minghui Qiu, Dawei Cheng, Xiaofeng He, Weining Qian. (2023)  
**Learning Knowledge-Enhanced Contextual Language Representations for Domain Natural Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Graph, Language Model, NLP, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2311.06761v1)  

---


**ABSTRACT**  
Knowledge-Enhanced Pre-trained Language Models (KEPLMs) improve the performance of various downstream NLP tasks by injecting knowledge facts from large-scale Knowledge Graphs (KGs). However, existing methods for pre-training KEPLMs with relational triples are difficult to be adapted to close domains due to the lack of sufficient domain graph semantics. In this paper, we propose a Knowledge-enhanced lANGuAge Representation learning framework for various clOsed dOmains (KANGAROO) via capturing the implicit graph structure among the entities. Specifically, since the entity coverage rates of closed-domain KGs can be relatively low and may exhibit the global sparsity phenomenon for knowledge injection, we consider not only the shallow relational representations of triples but also the hyperbolic embeddings of deep hierarchical entity-class structures for effective knowledge fusion.Moreover, as two closed-domain entities under the same entity-class often have locally dense neighbor subgraphs counted by max point biconnected component, we further propose a data augmentation strategy based on contrastive learning over subgraphs to construct hard negative samples of higher quality. It makes the underlying KELPMs better distinguish the semantics of these neighboring entities to further complement the global semantic sparsity. In the experiments, we evaluate KANGAROO over various knowledge-aware and general NLP tasks in both full and few-shot learning settings, outperforming various KEPLM training paradigms performance in closed-domains significantly.

{{</citation>}}


### (12/63) Sharing, Teaching and Aligning: Knowledgeable Transfer Learning for Cross-Lingual Machine Reading Comprehension (Tingfeng Cao et al., 2023)

{{<citation>}}

Tingfeng Cao, Chengyu Wang, Chuanqi Tan, Jun Huang, Jinhui Zhu. (2023)  
**Sharing, Teaching and Aligning: Knowledgeable Transfer Learning for Cross-Lingual Machine Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Reading Comprehension  
[Paper Link](http://arxiv.org/abs/2311.06758v1)  

---


**ABSTRACT**  
In cross-lingual language understanding, machine translation is often utilized to enhance the transferability of models across languages, either by translating the training data from the source language to the target, or from the target to the source to aid inference. However, in cross-lingual machine reading comprehension (MRC), it is difficult to perform a deep level of assistance to enhance cross-lingual transfer because of the variation of answer span positions in different languages. In this paper, we propose X-STA, a new approach for cross-lingual MRC. Specifically, we leverage an attentive teacher to subtly transfer the answer spans of the source language to the answer output space of the target. A Gradient-Disentangled Knowledge Sharing technique is proposed as an improved cross-attention block. In addition, we force the model to learn semantic alignments from multiple granularities and calibrate the model outputs with teacher guidance to enhance cross-lingual transferability. Experiments on three multi-lingual MRC datasets show the effectiveness of our method, outperforming state-of-the-art approaches.

{{</citation>}}


### (13/63) From Complex to Simple: Unraveling the Cognitive Tree for Reasoning with Small Language Models (Junbing Yan et al., 2023)

{{<citation>}}

Junbing Yan, Chengyu Wang, Taolin Zhang, Xiaofeng He, Jun Huang, Wei Zhang. (2023)  
**From Complex to Simple: Unraveling the Cognitive Tree for Reasoning with Small Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.06754v1)  

---


**ABSTRACT**  
Reasoning is a distinctive human capacity, enabling us to address complex problems by breaking them down into a series of manageable cognitive steps. Yet, complex logical reasoning is still cumbersome for language models. Based on the dual process theory in cognitive science, we are the first to unravel the cognitive reasoning abilities of language models. Our framework employs an iterative methodology to construct a Cognitive Tree (CogTree). The root node of this tree represents the initial query, while the leaf nodes consist of straightforward questions that can be answered directly. This construction involves two main components: the implicit extraction module (referred to as the intuitive system) and the explicit reasoning module (referred to as the reflective system). The intuitive system rapidly generates multiple responses by utilizing in-context examples, while the reflective system scores these responses using comparative learning. The scores guide the intuitive system in its subsequent generation step. Our experimental results on two popular and challenging reasoning tasks indicate that it is possible to achieve a performance level comparable to that of GPT-3.5 (with 175B parameters), using a significantly smaller language model that contains fewer parameters (<=7B) than 5% of GPT-3.5.

{{</citation>}}


### (14/63) Towards General-Purpose Speech Abilities for Large Language Models Using Unpaired Data (Yassir Fathullah et al., 2023)

{{<citation>}}

Yassir Fathullah, Chunyang Wu, Egor Lakomkin, Junteng Jia, Yuan Shangguan, Jay Mahadeokar, Ozlem Kalinli, Christian Fuegen, Mike Seltzer. (2023)  
**Towards General-Purpose Speech Abilities for Large Language Models Using Unpaired Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06753v1)  

---


**ABSTRACT**  
In this work, we extend the instruction-tuned Llama-2 model with end-to-end general-purpose speech processing and reasoning abilities while maintaining the wide range of LLM capabilities, without using any carefully curated paired data. The proposed model can utilize audio prompts as a replacement for text and sustain a conversation. Such a model also has extended cross-modal capabilities such as being able to perform speech question answering, speech translation, and audio summarization amongst many other closed and open-domain tasks. This is unlike prior approaches in speech, in which LLMs are extended to handle audio for a limited number of pre-designated tasks. Experiments show that our end-to-end approach is on par with or outperforms a cascaded system (speech recognizer + LLM) in terms of modeling the response to a prompt. Furthermore, unlike a cascade, our approach shows the ability to interchange text and audio modalities and utilize the prior context in a conversation to provide better results.

{{</citation>}}


### (15/63) BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis (Tingfeng Cao et al., 2023)

{{<citation>}}

Tingfeng Cao, Chengyu Wang, Bingyan Liu, Ziheng Wu, Jinhui Zhu, Jun Huang. (2023)  
**BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06752v1)  

---


**ABSTRACT**  
Recently, diffusion-based deep generative models (e.g., Stable Diffusion) have shown impressive results in text-to-image synthesis. However, current text-to-image models often require multiple passes of prompt engineering by humans in order to produce satisfactory results for real-world applications. We propose BeautifulPrompt, a deep generative model to produce high-quality prompts from very simple raw descriptions, which enables diffusion-based models to generate more beautiful images. In our work, we first fine-tuned the BeautifulPrompt model over low-quality and high-quality collecting prompt pairs. Then, to ensure that our generated prompts can generate more beautiful images, we further propose a Reinforcement Learning with Visual AI Feedback technique to fine-tune our model to maximize the reward values of the generated prompts, where the reward values are calculated based on the PickScore and the Aesthetic Scores. Our results demonstrate that learning from visual AI feedback promises the potential to improve the quality of generated prompts and images significantly. We further showcase the integration of BeautifulPrompt to a cloud-native AI platform to provide better text-to-image generation service in the cloud.

{{</citation>}}


### (16/63) Detecting and Correcting Hate Speech in Multimodal Memes with Large Visual Language Model (Minh-Hao Van et al., 2023)

{{<citation>}}

Minh-Hao Van, Xintao Wu. (2023)  
**Detecting and Correcting Hate Speech in Multimodal Memes with Large Visual Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.06737v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs) have taken the spotlight in natural language processing. Further, integrating LLMs with vision enables the users to explore more emergent abilities in multimodality. Visual language models (VLMs), such as LLaVA, Flamingo, or GPT-4, have demonstrated impressive performance on various visio-linguistic tasks. Consequently, there are enormous applications of large models that could be potentially used on social media platforms. Despite that, there is a lack of related work on detecting or correcting hateful memes with VLMs. In this work, we study the ability of VLMs on hateful meme detection and hateful meme correction tasks with zero-shot prompting. From our empirical experiments, we show the effectiveness of the pretrained LLaVA model and discuss its strengths and weaknesses in these tasks.

{{</citation>}}


### (17/63) Comprehending Lexical and Affective Ontologies in the Demographically Diverse Spatial Social Media Discourse (Salim Sazzed, 2023)

{{<citation>}}

Salim Sazzed. (2023)  
**Comprehending Lexical and Affective Ontologies in the Demographically Diverse Spatial Social Media Discourse**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2311.06729v1)  

---


**ABSTRACT**  
This study aims to comprehend linguistic and socio-demographic features, encompassing English language styles, conveyed sentiments, and lexical diversity within spatial online social media review data. To this end, we undertake a case study that scrutinizes reviews composed by two distinct and demographically diverse groups. Our analysis entails the extraction and examination of various statistical, grammatical, and sentimental features from these two groups. Subsequently, we leverage these features with machine learning (ML) classifiers to discern their potential in effectively differentiating between the groups. Our investigation unveils substantial disparities in certain linguistic attributes between the two groups. When integrated into ML classifiers, these attributes exhibit a marked efficacy in distinguishing the groups, yielding a macro F1 score of approximately 0.85. Furthermore, we conduct a comparative evaluation of these linguistic features with word n-gram-based lexical features in discerning demographically diverse review data. As expected, the n-gram lexical features, coupled with fine-tuned transformer-based models, show superior performance, attaining accuracies surpassing 95\% and macro F1 scores exceeding 0.96. Our meticulous analysis and comprehensive evaluations substantiate the efficacy of linguistic and sentimental features in effectively discerning demographically diverse review data. The findings of this study provide valuable guidelines for future research endeavors concerning the analysis of demographic patterns in textual content across various social media platforms.

{{</citation>}}


### (18/63) Controllable Topic-Focused Abstractive Summarization (Seyed Ali Bahrainian et al., 2023)

{{<citation>}}

Seyed Ali Bahrainian, Martin Jaggi, Carsten Eickhoff. (2023)  
**Controllable Topic-Focused Abstractive Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Summarization, T5, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06724v1)  

---


**ABSTRACT**  
Controlled abstractive summarization focuses on producing condensed versions of a source article to cover specific aspects by shifting the distribution of generated text towards a desired style, e.g., a set of topics. Subsequently, the resulting summaries may be tailored to user-defined requirements. This paper presents a new Transformer-based architecture capable of producing topic-focused summaries. The architecture modifies the cross-attention mechanism of the Transformer to bring topic-focus control to the generation process while not adding any further parameters to the model. We show that our model sets a new state of the art on the NEWTS dataset in terms of topic-focused abstractive summarization as well as a topic-prevalence score. Moreover, we show via extensive experiments that our proposed topical cross-attention mechanism can be plugged into various Transformer models, such as BART and T5, improving their performance on the CNN/Dailymail and XSum benchmark datasets for abstractive summarization. This is achieved via fine-tuning, without requiring training from scratch. Finally, we show through human evaluation that our model generates more faithful summaries outperforming the state-of-the-art Frost model.

{{</citation>}}


### (19/63) Trusted Source Alignment in Large Language Models (Vasilisa Bashlovkina et al., 2023)

{{<citation>}}

Vasilisa Bashlovkina, Zhaobin Kuang, Riley Matthews, Edward Clifford, Yennie Jun, William W. Cohen, Simon Baumgartner. (2023)  
**Trusted Source Alignment in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, PaLM, QA  
[Paper Link](http://arxiv.org/abs/2311.06697v1)  

---


**ABSTRACT**  
Large language models (LLMs) are trained on web-scale corpora that inevitably include contradictory factual information from sources of varying reliability. In this paper, we propose measuring an LLM property called trusted source alignment (TSA): the model's propensity to align with content produced by trusted publishers in the face of uncertainty or controversy. We present FactCheckQA, a TSA evaluation dataset based on a corpus of fact checking articles. We describe a simple protocol for evaluating TSA and offer a detailed analysis of design considerations including response extraction, claim contextualization, and bias in prompt formulation. Applying the protocol to PaLM-2, we find that as we scale up the model size, the model performance on FactCheckQA improves from near-random to up to 80% balanced accuracy in aligning with trusted sources.

{{</citation>}}


## cs.AI (2)



### (20/63) Assessing the Interpretability of Programmatic Policies with Large Language Models (Zahra Bashir et al., 2023)

{{<citation>}}

Zahra Bashir, Michael Bowling, Levi H. S. Lelis. (2023)  
**Assessing the Interpretability of Programmatic Policies with Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-PL, cs-SE, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.06979v1)  

---


**ABSTRACT**  
Although the synthesis of programs encoding policies often carries the promise of interpretability, systematic evaluations to assess the interpretability of these policies were never performed, likely because of the complexity of such an evaluation. In this paper, we introduce a novel metric that uses large-language models (LLM) to assess the interpretability of programmatic policies. For our metric, an LLM is given both a program and a description of its associated programming language. The LLM then formulates a natural language explanation of the program. This explanation is subsequently fed into a second LLM, which tries to reconstruct the program from the natural language explanation. Our metric measures the behavioral similarity between the reconstructed program and the original. We validate our approach using obfuscated programs that are used to solve classic programming problems. We also assess our metric with programmatic policies synthesized for playing a real-time strategy game, comparing the interpretability scores of programmatic policies synthesized by an existing system to lightly obfuscated versions of the same programs. Our LLM-based interpretability score consistently ranks less interpretable programs lower and more interpretable ones higher. These findings suggest that our metric could serve as a reliable and inexpensive tool for evaluating the interpretability of programmatic policies.

{{</citation>}}


### (21/63) Enabling Human-Centered AI: A Methodological Perspective (Wei Xu et al., 2023)

{{<citation>}}

Wei Xu, Zaifeng Gao. (2023)  
**Enabling Human-Centered AI: A Methodological Perspective**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-SE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06703v2)  

---


**ABSTRACT**  
Human-centered AI (HCAI) is a design philosophy that advocates prioritizing humans in designing, developing, and deploying intelligent systems, aiming to maximize the benefits of AI to humans and avoid potential adverse impacts. While HCAI continues to influence, the lack of guidance on methodology in practice makes its adoption challenging. This paper proposes a comprehensive HCAI framework based on our previous work with integrated components, including design goals, design principles, implementation approaches, interdisciplinary teams, HCAI methods, and HCAI processes. This paper also presents a "three-layer" approach to facilitate the implementation of the framework. We believe this systematic and executable framework can overcome the weaknesses in current HCAI frameworks and the challenges currently faced in practice, putting it into action to enable HCAI further.

{{</citation>}}


## cs.CV (11)



### (22/63) CD-COCO: A Versatile Complex Distorted COCO Database for Scene-Context-Aware Computer Vision (Ayman Beghdadi et al., 2023)

{{<citation>}}

Ayman Beghdadi, Azeddine Beghdadi, Malik Mallem, Lotfi Beji, Faouzi Alaya Cheikh. (2023)  
**CD-COCO: A Versatile Complex Distorted COCO Database for Scene-Context-Aware Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-DB, cs.CV  
Keywords: Computer Vision, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.06976v1)  

---


**ABSTRACT**  
The recent development of deep learning methods applied to vision has enabled their increasing integration into real-world applications to perform complex Computer Vision (CV) tasks. However, image acquisition conditions have a major impact on the performance of high-level image processing. A possible solution to overcome these limitations is to artificially augment the training databases or to design deep learning models that are robust to signal distortions. We opt here for the first solution by enriching the database with complex and realistic distortions which were ignored until now in the existing databases. To this end, we built a new versatile database derived from the well-known MS-COCO database to which we applied local and global photo-realistic distortions. These new local distortions are generated by considering the scene context of the images that guarantees a high level of photo-realism. Distortions are generated by exploiting the depth information of the objects in the scene as well as their semantics. This guarantees a high level of photo-realism and allows to explore real scenarios ignored in conventional databases dedicated to various CV applications. Our versatile database offers an efficient solution to improve the robustness of various CV tasks such as Object Detection (OD), scene segmentation, and distortion-type classification methods. The image database, scene classification index, and distortion generation codes are publicly available \footnote{\url{https://github.com/Aymanbegh/CD-COCO}}

{{</citation>}}


### (23/63) DialMAT: Dialogue-Enabled Transformer with Moment-Based Adversarial Training (Kanta Kaneda et al., 2023)

{{<citation>}}

Kanta Kaneda, Ryosuke Korekata, Yuiga Wada, Shunya Nagashima, Motonari Kambara, Yui Iioka, Haruka Matsuo, Yuto Imai, Takayuki Nishimura, Komei Sugiura. (2023)  
**DialMAT: Dialogue-Enabled Transformer with Moment-Based Adversarial Training**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-RO, cs.CV  
Keywords: AI, Adversarial Training, Dialog, Dialogue, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06855v1)  

---


**ABSTRACT**  
This paper focuses on the DialFRED task, which is the task of embodied instruction following in a setting where an agent can actively ask questions about the task. To address this task, we propose DialMAT. DialMAT introduces Moment-based Adversarial Training, which incorporates adversarial perturbations into the latent space of language, image, and action. Additionally, it introduces a crossmodal parallel feature extraction mechanism that applies foundation models to both language and image. We evaluated our model using a dataset constructed from the DialFRED dataset and demonstrated superior performance compared to the baseline method in terms of success rate and path weighted success rate. The model secured the top position in the DialFRED Challenge, which took place at the CVPR 2023 Embodied AI workshop.

{{</citation>}}


### (24/63) Contrastive Learning of View-Invariant Representations for Facial Expressions Recognition (Shuvendu Roy et al., 2023)

{{<citation>}}

Shuvendu Roy, Ali Etemad. (2023)  
**Contrastive Learning of View-Invariant Representations for Facial Expressions Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.06852v1)  

---


**ABSTRACT**  
Although there has been much progress in the area of facial expression recognition (FER), most existing methods suffer when presented with images that have been captured from viewing angles that are non-frontal and substantially different from those used in the training process. In this paper, we propose ViewFX, a novel view-invariant FER framework based on contrastive learning, capable of accurately classifying facial expressions regardless of the input viewing angles during inference. ViewFX learns view-invariant features of expression using a proposed self-supervised contrastive loss which brings together different views of the same subject with a particular expression in the embedding space. We also introduce a supervised contrastive loss to push the learnt view-invariant features of each expression away from other expressions. Since facial expressions are often distinguished with very subtle differences in the learned feature space, we incorporate the Barlow twins loss to reduce the redundancy and correlations of the representations in the learned representations. The proposed method is a substantial extension of our previously proposed CL-MEx, which only had a self-supervised loss. We test the proposed framework on two public multi-view facial expression recognition datasets, KDEF and DDCF. The experiments demonstrate that our approach outperforms previous works in the area and sets a new state-of-the-art for both datasets while showing considerably less sensitivity to challenging angles and the number of output labels used for training. We also perform detailed sensitivity and ablation experiments to evaluate the impact of different components of our model as well as its sensitivity to different parameters.

{{</citation>}}


### (25/63) Dual-Branch Reconstruction Network for Industrial Anomaly Detection with RGB-D Data (Chenyang Bi et al., 2023)

{{<citation>}}

Chenyang Bi, Yueyang Li, Haichi Luo. (2023)  
**Dual-Branch Reconstruction Network for Industrial Anomaly Detection with RGB-D Data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.06797v1)  

---


**ABSTRACT**  
Unsupervised anomaly detection methods are at the forefront of industrial anomaly detection efforts and have made notable progress. Previous work primarily used 2D information as input, but multi-modal industrial anomaly detection based on 3D point clouds and RGB images is just beginning to emerge. The regular approach involves utilizing large pre-trained models for feature representation and storing them in memory banks. However, the above methods require a longer inference time and higher memory usage, which cannot meet the real-time requirements of the industry. To overcome these issues, we propose a lightweight dual-branch reconstruction network(DBRN) based on RGB-D input, learning the decision boundary between normal and abnormal examples. The requirement for alignment between the two modalities is eliminated by using depth maps instead of point cloud input. Furthermore, we introduce an importance scoring module in the discriminative network to assist in fusing features from these two modalities, thereby obtaining a comprehensive discriminative result. DBRN achieves 92.8% AUROC with high inference efficiency on the MVTec 3D-AD dataset without large pre-trained models and memory banks.

{{</citation>}}


### (26/63) Deep Perspective Transformation Based Vehicle Localization on Bird's Eye View (Abtin Mahyar et al., 2023)

{{<citation>}}

Abtin Mahyar, Hossein Motamednia, Dara Rahmati. (2023)  
**Deep Perspective Transformation Based Vehicle Localization on Bird's Eye View**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.06796v1)  

---


**ABSTRACT**  
An accurate understanding of a self-driving vehicle's surrounding environment is crucial for its navigation system. To enhance the effectiveness of existing algorithms and facilitate further research, it is essential to provide comprehensive data to the routing system. Traditional approaches rely on installing multiple sensors to simulate the environment, leading to high costs and complexity. In this paper, we propose an alternative solution by generating a top-down representation of the scene, enabling the extraction of distances and directions of other cars relative to the ego vehicle. We introduce a new synthesized dataset that offers extensive information about the ego vehicle and its environment in each frame, providing valuable resources for similar downstream tasks. Additionally, we present an architecture that transforms perspective view RGB images into bird's-eye-view maps with segmented surrounding vehicles. This approach offers an efficient and cost-effective method for capturing crucial environmental information for self-driving cars. Code and dataset are available at https://github.com/IPM-HPC/Perspective-BEV-Transformer.

{{</citation>}}


### (27/63) InfMLLM: A Unified Framework for Visual-Language Tasks (Qiang Zhou et al., 2023)

{{<citation>}}

Qiang Zhou, Zhibin Wang, Wei Chu, Yinghui Xu, Hao Li, Yuan Qi. (2023)  
**InfMLLM: A Unified Framework for Visual-Language Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2311.06791v1)  

---


**ABSTRACT**  
Large language models (LLMs) have proven their remarkable versatility in handling a comprehensive range of language-centric applications. To expand LLMs' capabilities to a broader spectrum of modal inputs, multimodal large language models (MLLMs) have attracted growing interest. This work delves into enabling LLMs to tackle more vision-language-related tasks, particularly image captioning, visual question answering (VQA,) and visual grounding. To this end, we implemented a three-stage training scheme: starting with lightweight alignment pretraining, then moderate-weight multitask hybrid training, and finally, LLM fine-tuning to improve instruction following capability. Throughout the training process, the requirements on GPU memory gradually increase. To effectively manage the number of visual embeddings passed to the LLM while preserving their positional information, we introduce a straightforward visual adapter module dubbed pool-adapter. Our experiments demonstrate that preserving the positional information of visual embeddings through the pool-adapter is particularly beneficial for tasks like visual grounding. We name our proposed approach InfMLLM and have evaluated it extensively on various benchmark datasets. Our results demonstrate that InfMLLM achieves either state-of-the-art (SOTA) performance or performance comparable to recent MLLMs. The code and model will be made open-source at: \url{https://github.com/mightyzau/InfMLLM}.

{{</citation>}}


### (28/63) Explainability of Vision Transformers: A Comprehensive Review and New Perspectives (Rojina Kashefi et al., 2023)

{{<citation>}}

Rojina Kashefi, Leili Barekatain, Mohammad Sabokrou, Fatemeh Aghaeipoor. (2023)  
**Explainability of Vision Transformers: A Comprehensive Review and New Perspectives**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.06786v1)  

---


**ABSTRACT**  
Transformers have had a significant impact on natural language processing and have recently demonstrated their potential in computer vision. They have shown promising results over convolution neural networks in fundamental computer vision tasks. However, the scientific community has not fully grasped the inner workings of vision transformers, nor the basis for their decision-making, which underscores the importance of explainability methods. Understanding how these models arrive at their decisions not only improves their performance but also builds trust in AI systems. This study explores different explainability methods proposed for visual transformers and presents a taxonomy for organizing them according to their motivations, structures, and application scenarios. In addition, it provides a comprehensive review of evaluation criteria that can be used for comparing explanation results, as well as explainability tools and frameworks. Finally, the paper highlights essential but unexplored aspects that can enhance the explainability of visual transformers, and promising research directions are suggested for future investment.

{{</citation>}}


### (29/63) Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models (Haoning Wu et al., 2023)

{{<citation>}}

Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Kaixin Xu, Chunyi Li, Jingwen Hou, Guangtao Zhai, Geng Xue, Wenxiu Sun, Qiong Yan, Weisi Lin. (2023)  
**Q-Instruct: Improving Low-level Visual Abilities for Multi-modality Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.06783v1)  

---


**ABSTRACT**  
Multi-modality foundation models, as represented by GPT-4V, have brought a new paradigm for low-level visual perception and understanding tasks, that can respond to a broad range of natural human instructions in a model. While existing foundation models have shown exciting potentials on low-level visual tasks, their related abilities are still preliminary and need to be improved. In order to enhance these models, we conduct a large-scale subjective experiment collecting a vast number of real human feedbacks on low-level vision. Each feedback follows a pathway that starts with a detailed description on the low-level visual appearance (*e.g. clarity, color, brightness* of an image, and ends with an overall conclusion, with an average length of 45 words. The constructed **Q-Pathway** dataset includes 58K detailed human feedbacks on 18,973 images with diverse low-level appearance. Moreover, to enable foundation models to robustly respond to diverse types of questions, we design a GPT-participated conversion to process these feedbacks into diverse-format 200K instruction-response pairs. Experimental results indicate that the **Q-Instruct** consistently elevates low-level perception and understanding abilities across several foundational models. We anticipate that our datasets can pave the way for a future that general intelligence can perceive, understand low-level visual appearance and evaluate visual quality like a human. Our dataset, model zoo, and demo is published at: https://q-future.github.io/Q-Instruct.

{{</citation>}}


### (30/63) ReIDTracker Sea: the technical report of BoaTrack and SeaDronesSee-MOT challenge at MaCVi of WACV24 (Kaer Huang et al., 2023)

{{<citation>}}

Kaer Huang, Weitu Chong. (2023)  
**ReIDTracker Sea: the technical report of BoaTrack and SeaDronesSee-MOT challenge at MaCVi of WACV24**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Drone, ImageNet  
[Paper Link](http://arxiv.org/abs/2311.07616v1)  

---


**ABSTRACT**  
Multi-Object Tracking is one of the most important technologies in maritime computer vision. Our solution tries to explore Multi-Object Tracking in maritime Unmanned Aerial vehicles (UAVs) and Unmanned Surface Vehicles (USVs) usage scenarios. Most of the current Multi-Object Tracking algorithms require complex association strategies and association information (2D location and motion, 3D motion, 3D depth, 2D appearance) to achieve better performance, which makes the entire tracking system extremely complex and heavy. At the same time, most of the current Multi-Object Tracking algorithms still require video annotation data which is costly to obtain for training. Our solution tries to explore Multi-Object Tracking in a completely unsupervised way. The scheme accomplishes instance representation learning by using self-supervision on ImageNet. Then, by cooperating with high-quality detectors, the multi-target tracking task can be completed simply and efficiently. The scheme achieved top 3 performance on both UAV-based Multi-Object Tracking with Reidentification and USV-based Multi-Object Tracking benchmarks and the solution won the championship in many multiple Multi-Object Tracking competitions. such as BDD100K MOT,MOTS, Waymo 2D MOT

{{</citation>}}


### (31/63) Aggregate, Decompose, and Fine-Tune: A Simple Yet Effective Factor-Tuning Method for Vision Transformer (Dongping Chen, 2023)

{{<citation>}}

Dongping Chen. (2023)  
**Aggregate, Decompose, and Fine-Tune: A Simple Yet Effective Factor-Tuning Method for Vision Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.06749v1)  

---


**ABSTRACT**  
Recent advancements have illuminated the efficacy of some tensorization-decomposition Parameter-Efficient Fine-Tuning methods like LoRA and FacT in the context of Vision Transformers (ViT). However, these methods grapple with the challenges of inadequately addressing inner- and cross-layer redundancy. To tackle this issue, we introduce EFfective Factor-Tuning (EFFT), a simple yet effective fine-tuning method. Within the VTAB-1K dataset, our EFFT surpasses all baselines, attaining state-of-the-art performance with a categorical average of 75.9% in top-1 accuracy with only 0.28% of the parameters for full fine-tuning. Considering the simplicity and efficacy of EFFT, it holds the potential to serve as a foundational benchmark. The code and model are now available at https://github.com/Dongping-Chen/EFFT-EFfective-Factor-Tuning.

{{</citation>}}


### (32/63) Two Stream Scene Understanding on Graph Embedding (Wenkai Yang et al., 2023)

{{<citation>}}

Wenkai Yang, Wenyuan Sun, Runxaing Huang. (2023)  
**Two Stream Scene Understanding on Graph Embedding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Embedding, Graph Attention Network, Graph Convolutional Network, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06746v1)  

---


**ABSTRACT**  
The paper presents a novel two-stream network architecture for enhancing scene understanding in computer vision. This architecture utilizes a graph feature stream and an image feature stream, aiming to merge the strengths of both modalities for improved performance in image classification and scene graph generation tasks. The graph feature stream network comprises a segmentation structure, scene graph generation, and a graph representation module. The segmentation structure employs the UPSNet architecture with a backbone that can be a residual network, Vit, or Swin Transformer. The scene graph generation component focuses on extracting object labels and neighborhood relationships from the semantic map to create a scene graph. Graph Convolutional Networks (GCN), GraphSAGE, and Graph Attention Networks (GAT) are employed for graph representation, with an emphasis on capturing node features and their interconnections. The image feature stream network, on the other hand, focuses on image classification through the use of Vision Transformer and Swin Transformer models. The two streams are fused using various data fusion methods. This fusion is designed to leverage the complementary strengths of graph-based and image-based features.Experiments conducted on the ADE20K dataset demonstrate the effectiveness of the proposed two-stream network in improving image classification accuracy compared to conventional methods. This research provides a significant contribution to the field of computer vision, particularly in the areas of scene understanding and image classification, by effectively combining graph-based and image-based approaches.

{{</citation>}}


## cs.LG (14)



### (33/63) Anchor Data Augmentation (Nora Schneider et al., 2023)

{{<citation>}}

Nora Schneider, Shirin Goshtasbpour, Fernando Perez-Cruz. (2023)  
**Anchor Data Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.06965v1)  

---


**ABSTRACT**  
We propose a novel algorithm for data augmentation in nonlinear over-parametrized regression. Our data augmentation algorithm borrows from the literature on causality and extends the recently proposed Anchor regression (AR) method for data augmentation, which is in contrast to the current state-of-the-art domain-agnostic solutions that rely on the Mixup literature. Our Anchor Data Augmentation (ADA) uses several replicas of the modified samples in AR to provide more training examples, leading to more robust regression predictions. We apply ADA to linear and nonlinear regression problems using neural networks. ADA is competitive with state-of-the-art C-Mixup solutions.

{{</citation>}}


### (34/63) Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks (Moshe Eliasof et al., 2023)

{{<citation>}}

Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb. (2023)  
**Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.06942v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

{{</citation>}}


### (35/63) Attention for Causal Relationship Discovery from Biological Neural Dynamics (Ziyu Lu et al., 2023)

{{<citation>}}

Ziyu Lu, Anika Tabassum, Shruti Kulkarni, Lu Mi, J. Nathan Kutz, Eric Shea-Brown, Seung-Hwan Lim. (2023)  
**Attention for Causal Relationship Discovery from Biological Neural Dynamics**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.06928v2)  

---


**ABSTRACT**  
This paper explores the potential of the transformer models for learning Granger causality in networks with complex nonlinear dynamics at every node, as in neurobiological and biophysical networks. Our study primarily focuses on a proof-of-concept investigation based on simulated neural dynamics, for which the ground-truth causality is known through the underlying connectivity matrix. For transformer models trained to forecast neuronal population dynamics, we show that the cross attention module effectively captures the causal relationship among neurons, with an accuracy equal or superior to that for the most popular Granger causality analysis method. While we acknowledge that real-world neurobiology data will bring further challenges, including dynamic connectivity and unobserved variability, this research offers an encouraging preliminary glimpse into the utility of the transformer model for causal representation learning in neuroscience.

{{</citation>}}


### (36/63) FLASH-RL: Federated Learning Addressing System and Static Heterogeneity using Reinforcement Learning (Sofiane Bouaziz et al., 2023)

{{<citation>}}

Sofiane Bouaziz, Hadjer Benmeziane, Youcef Imine, Leila Hamdad, Smail Niar, Hamza Ouarnoughi. (2023)  
**FLASH-RL: Federated Learning Addressing System and Static Heterogeneity using Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06917v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a promising Machine Learning paradigm, enabling multiple users to collaboratively train a shared model while preserving their local data. To minimize computing and communication costs associated with parameter transfer, it is common practice in FL to select a subset of clients in each training round. This selection must consider both system and static heterogeneity. Therefore, we propose FLASH-RL, a framework that utilizes Double Deep QLearning (DDQL) to address both system and static heterogeneity in FL. FLASH-RL introduces a new reputation-based utility function to evaluate client contributions based on their current and past performances. Additionally, an adapted DDQL algorithm is proposed to expedite the learning process. Experimental results on MNIST and CIFAR-10 datasets have shown FLASH-RL's effectiveness in achieving a balanced trade-off between model performance and end-to-end latency against existing solutions. Indeed, FLASH-RL reduces latency by up to 24.83% compared to FedAVG and 24.67% compared to FAVOR. It also reduces the training rounds by up to 60.44% compared to FedAVG and +76% compared to FAVOR. In fall detection using the MobiAct dataset, FLASH-RL outperforms FedAVG by up to 2.82% in model's performance and reduces latency by up to 34.75%. Additionally, FLASH-RL achieves the target performance faster, with up to a 45.32% reduction in training rounds compared to FedAVG.

{{</citation>}}


### (37/63) Preserving Node-level Privacy in Graph Neural Networks (Zihang Xiang et al., 2023)

{{<citation>}}

Zihang Xiang, Tianhao Wang, Di Wang. (2023)  
**Preserving Node-level Privacy in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.06888v1)  

---


**ABSTRACT**  
Differential privacy (DP) has seen immense applications in learning on tabular, image, and sequential data where instance-level privacy is concerned. In learning on graphs, contrastingly, works on node-level privacy are highly sparse. Challenges arise as existing DP protocols hardly apply to the message-passing mechanism in Graph Neural Networks (GNNs).   In this study, we propose a solution that specifically addresses the issue of node-level privacy. Our protocol consists of two main components: 1) a sampling routine called HeterPoisson, which employs a specialized node sampling strategy and a series of tailored operations to generate a batch of sub-graphs with desired properties, and 2) a randomization routine that utilizes symmetric multivariate Laplace (SML) noise instead of the commonly used Gaussian noise. Our privacy accounting shows this particular combination provides a non-trivial privacy guarantee. In addition, our protocol enables GNN learning with good performance, as demonstrated by experiments on five real-world datasets; compared with existing baselines, our method shows significant advantages, especially in the high privacy regime. Experimentally, we also 1) perform membership inference attacks against our protocol and 2) apply privacy audit techniques to confirm our protocol's privacy integrity.   In the sequel, we present a study on a seemingly appealing approach \cite{sajadmanesh2023gap} (USENIX'23) that protects node-level privacy via differentially private node/instance embeddings. Unfortunately, such work has fundamental privacy flaws, which are identified through a thorough case study. More importantly, we prove an impossibility result of achieving both (strong) privacy and (acceptable) utility through private instance embedding. The implication is that such an approach has intrinsic utility barriers when enforcing differential privacy.

{{</citation>}}


### (38/63) Inference and Interference: The Role of Clipping, Pruning and Loss Landscapes in Differentially Private Stochastic Gradient Descent (Lauren Watson et al., 2023)

{{<citation>}}

Lauren Watson, Eric Gan, Mohan Dantam, Baharan Mirzasoleiman, Rik Sarkar. (2023)  
**Inference and Interference: The Role of Clipping, Pruning and Loss Landscapes in Differentially Private Stochastic Gradient Descent**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.06839v1)  

---


**ABSTRACT**  
Differentially private stochastic gradient descent (DP-SGD) is known to have poorer training and test performance on large neural networks, compared to ordinary stochastic gradient descent (SGD). In this paper, we perform a detailed study and comparison of the two processes and unveil several new insights. By comparing the behavior of the two processes separately in early and late epochs, we find that while DP-SGD makes slower progress in early stages, it is the behavior in the later stages that determines the end result. This separate analysis of the clipping and noise addition steps of DP-SGD shows that while noise introduces errors to the process, gradient descent can recover from these errors when it is not clipped, and clipping appears to have a larger impact than noise. These effects are amplified in higher dimensions (large neural networks), where the loss basin occupies a lower dimensional space. We argue theoretically and using extensive experiments that magnitude pruning can be a suitable dimension reduction technique in this regard, and find that heavy pruning can improve the test accuracy of DPSGD.

{{</citation>}}


### (39/63) GraNNDis: Efficient Unified Distributed Training Framework for Deep GNNs on Large Clusters (Jaeyong Song et al., 2023)

{{<citation>}}

Jaeyong Song, Hongsun Jang, Jaewon Jung, Youngsok Kim, Jinho Lee. (2023)  
**GraNNDis: Efficient Unified Distributed Training Framework for Deep GNNs on Large Clusters**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2311.06837v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are one of the most rapidly growing fields within deep learning. According to the growth in the dataset and the model size used for GNNs, an important problem is that it becomes nearly impossible to keep the whole network on GPU memory. Among numerous attempts, distributed training is one popular approach to address the problem. However, due to the nature of GNNs, existing distributed approaches suffer from poor scalability, mainly due to the slow external server communications.   In this paper, we propose GraNNDis, an efficient distributed GNN training framework for training GNNs on large graphs and deep layers. GraNNDis introduces three new techniques. First, shared preloading provides a training structure for a cluster of multi-GPU servers. We suggest server-wise preloading of essential vertex dependencies to reduce the low-bandwidth external server communications. Second, we present expansion-aware sampling. Because shared preloading alone has limitations because of the neighbor explosion, expansion-aware sampling reduces vertex dependencies that span across server boundaries. Third, we propose cooperative batching to create a unified framework for full-graph and minibatch training. It significantly reduces redundant memory usage in mini-batch training. From this, GraNNDis enables a reasonable trade-off between full-graph and mini-batch training through unification especially when the entire graph does not fit into the GPU memory. With experiments conducted on a multi-server/multi-GPU cluster, we show that GraNNDis provides superior speedup over the state-of-the-art distributed GNN training frameworks.

{{</citation>}}


### (40/63) Open-Set Graph Anomaly Detection via Normal Structure Regularisation (Qizhou Wang et al., 2023)

{{<citation>}}

Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray Buntine, Christopher Leckie. (2023)  
**Open-Set Graph Anomaly Detection via Normal Structure Regularisation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.06835v1)  

---


**ABSTRACT**  
This paper considers an under-explored Graph Anomaly Detection (GAD) task, namely open-set GAD, which aims to detect anomalous nodes using a small number of labelled training normal and anomaly nodes (known as seen anomalies) that cannot illustrate all possible inference-time abnormalities. The task has attracted growing attention due to the availability of anomaly prior knowledge from the label information that can help to substantially reduce detection errors. However, current methods tend to over-emphasise fitting the seen anomalies, leading to a weak generalisation ability to detect unseen anomalies, i.e., those that are not illustrated by the labelled anomaly nodes. Further, they were introduced to handle Euclidean data, failing to effectively capture important non-Euclidean features for GAD. In this work, we propose a novel open-set GAD approach, namely normal structure regularisation (NSReg), to leverage the rich normal graph structure embedded in the labelled nodes to tackle the aforementioned two issues. In particular, NSReg trains an anomaly-discriminative supervised graph anomaly detector, with a plug-and-play regularisation term to enforce compact, semantically-rich representations of normal nodes. To this end, the regularisation is designed to differentiate various types of normal nodes, including labelled normal nodes that are connected in their local neighbourhood, and those that are not connected. By doing so, it helps incorporate strong normality into the supervised anomaly detector learning, mitigating their overfitting to the seen anomalies. Extensive empirical results on real-world datasets demonstrate the superiority of our proposed NSReg for open-set GAD.

{{</citation>}}


### (41/63) Fairness Hacking: The Malicious Practice of Shrouding Unfairness in Algorithms (Kristof Meding et al., 2023)

{{<citation>}}

Kristof Meding, Thilo Hagendorff. (2023)  
**Fairness Hacking: The Malicious Practice of Shrouding Unfairness in Algorithms**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06826v1)  

---


**ABSTRACT**  
Fairness in machine learning (ML) is an ever-growing field of research due to the manifold potential for harm from algorithmic discrimination. To prevent such harm, a large body of literature develops new approaches to quantify fairness. Here, we investigate how one can divert the quantification of fairness by describing a practice we call "fairness hacking" for the purpose of shrouding unfairness in algorithms. This impacts end-users who rely on learning algorithms, as well as the broader community interested in fair AI practices. We introduce two different categories of fairness hacking in reference to the established concept of p-hacking. The first category, intra-metric fairness hacking, describes the misuse of a particular metric by adding or removing sensitive attributes from the analysis. In this context, countermeasures that have been developed to prevent or reduce p-hacking can be applied to similarly prevent or reduce fairness hacking. The second category of fairness hacking is inter-metric fairness hacking. Inter-metric fairness hacking is the search for a specific fair metric with given attributes. We argue that countermeasures to prevent or reduce inter-metric fairness hacking are still in their infancy. Finally, we demonstrate both types of fairness hacking using real datasets. Our paper intends to serve as a guidance for discussions within the fair ML community to prevent or reduce the misuse of fairness metrics, and thus reduce overall harm from ML applications.

{{</citation>}}


### (42/63) MetaMix: Meta-state Precision Searcher for Mixed-precision Activation Quantization (Han-Byul Kim et al., 2023)

{{<citation>}}

Han-Byul Kim, Joo Hyung Lee, Sungjoo Yoo, Hong-Seok Kim. (2023)  
**MetaMix: Meta-state Precision Searcher for Mixed-precision Activation Quantization**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2311.06798v1)  

---


**ABSTRACT**  
Mixed-precision quantization of efficient networks often suffer from activation instability encountered in the exploration of bit selections. To address this problem, we propose a novel method called MetaMix which consists of bit selection and weight training phases. The bit selection phase iterates two steps, (1) the mixed-precision-aware weight update, and (2) the bit-search training with the fixed mixed-precision-aware weights, both of which combined reduce activation instability in mixed-precision quantization and contribute to fast and high-quality bit selection. The weight training phase exploits the weights and step sizes trained in the bit selection phase and fine-tunes them thereby offering fast training. Our experiments with efficient and hard-to-quantize networks, i.e., MobileNet v2 and v3, and ResNet-18 on ImageNet show that our proposed method pushes the boundary of mixed-precision quantization, in terms of accuracy vs. operations, by outperforming both mixed- and single-precision SOTA methods.

{{</citation>}}


### (43/63) Large Language Models' Understanding of Math: Source Criticism and Extrapolation (Roozbeh Yousefzadeh et al., 2023)

{{<citation>}}

Roozbeh Yousefzadeh, Xuenan Cao. (2023)  
**Large Language Models' Understanding of Math: Source Criticism and Extrapolation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, math-HO  
Keywords: AI, GPT, GPT-4, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2311.07618v1)  

---


**ABSTRACT**  
It has been suggested that large language models such as GPT-4 have acquired some form of understanding beyond the correlations among the words in text including some understanding of mathematics as well. Here, we perform a critical inquiry into this claim by evaluating the mathematical understanding of the GPT-4 model. Considering that GPT-4's training set is a secret, it is not straightforward to evaluate whether the model's correct answers are based on a mathematical understanding or based on replication of proofs that the model has seen before. We specifically craft mathematical questions which their formal proofs are not readily available on the web, proofs that are more likely not seen by the GPT-4. We see that GPT-4 is unable to solve those problems despite their simplicity. It is hard to find scientific evidence suggesting that GPT-4 has acquired an understanding of even basic mathematical concepts. A straightforward way to find failure modes of GPT-4 in theorem proving is to craft questions where their formal proofs are not available on the web. Our finding suggests that GPT-4's ability is to reproduce, rephrase, and polish the mathematical proofs that it has seen before, and not in grasping mathematical concepts. We also see that GPT-4's ability to prove mathematical theorems is continuously expanding over time despite the claim that it is a fixed model. We suggest that the task of proving mathematical theorems in formal language is comparable to the methods used in search engines such as Google while predicting the next word in a sentence may be a misguided approach, a recipe that often leads to excessive extrapolation and eventual failures. Prompting the GPT-4 over and over may benefit the GPT-4 and the OpenAI, but we question whether it is valuable for machine learning or for theorem proving.

{{</citation>}}


### (44/63) Application of a Dense Fusion Attention Network in Fault Diagnosis of Centrifugal Fan (Ruijun Wang et al., 2023)

{{<citation>}}

Ruijun Wang, Yuan Liu, Zhixia Fan, Xiaogang Xu, Huijie Wang. (2023)  
**Application of a Dense Fusion Attention Network in Fault Diagnosis of Centrifugal Fan**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.07614v1)  

---


**ABSTRACT**  
Although the deep learning recognition model has been widely used in the condition monitoring of rotating machinery. However, it is still a challenge to understand the correspondence between the structure and function of the model and the diagnosis process. Therefore, this paper discusses embedding distributed attention modules into dense connections instead of traditional dense cascading operations. It not only decouples the influence of space and channel on fault feature adaptive recalibration feature weights, but also forms a fusion attention function. The proposed dense fusion focuses on the visualization of the network diagnosis process, which increases the interpretability of model diagnosis. How to continuously and effectively integrate different functions to enhance the ability to extract fault features and the ability to resist noise is answered. Centrifugal fan fault data is used to verify this network. Experimental results show that the network has stronger diagnostic performance than other advanced fault diagnostic models.

{{</citation>}}


### (45/63) DeepQC: A Deep Learning System for Automatic Quality Control of In-situ Soil Moisture Sensor Time Series Data (Lahari Bandaru et al., 2023)

{{<citation>}}

Lahari Bandaru, Bharat C Irigireddy, Brian Davis. (2023)  
**DeepQC: A Deep Learning System for Automatic Quality Control of In-situ Soil Moisture Sensor Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2311.06735v1)  

---


**ABSTRACT**  
Amidst changing climate, real-time soil moisture monitoring is vital for the development of in-season decision support tools to help farmers manage weather related risks. Precision Sustainable Agriculture (PSA) recently established a real-time soil moisture monitoring network across the central, Midwest, and eastern U.S., but field-scale sensor observations often come with data gaps and anomalies. To maintain the data quality needed for development of decision tools, a quality control system is necessary. The International Soil Moisture Network (ISMN) introduced the Flagit module for anomaly detection in soil moisture observations. However, under certain conditions, Flagit's quality control approaches may underperform in identifying anomalies. Recently deep learning methods have been successfully applied to detect anomalies in time series data in various disciplines. However, their use in agriculture has not been yet investigated. This study focuses on developing a Bi-directional Long Short-Term Memory (LSTM) model, referred to as DeepQC, to identify anomalies in soil moisture data. Manual flagged PSA observations were used for training, validation, and testing the model, following an 80:10:10 split. The study then compared the DeepQC and Flagit based estimates to assess their relative performance. Flagit corrected flagged 95.5% of the corrected observations and 50.3% of the anomaly observations, indicating its limitations in identifying anomalies. On the other hand, the DeepQC correctly flagged 99.7% of the correct observations and 95.6% of the anomalies in significantly less time, demonstrating its superiority over Flagit approach. Importantly, DeepQC's performance remained consistent regardless of the number of anomalies. Given the promising results obtained with the DeepQC, future studies will focus on implementing this model on national and global soil moisture networks.

{{</citation>}}


### (46/63) Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer (Bowen Tan et al., 2023)

{{<citation>}}

Bowen Tan, Yun Zhu, Lijuan Liu, Eric Xing, Zhiting Hu, Jindong Chen. (2023)  
**Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: PaLM, T5  
[Paper Link](http://arxiv.org/abs/2311.06720v1)  

---


**ABSTRACT**  
Large language models (LLMs) such as T0, FLAN, and OPT-IML, excel in multi-tasking under a unified instruction-following paradigm, where they also exhibit remarkable generalization abilities to unseen tasks. Despite their impressive performance, these LLMs, with sizes ranging from several billion to hundreds of billions of parameters, demand substantial computational resources, making their training and inference expensive and inefficient. Furthermore, adapting these models to downstream applications, particularly complex tasks, is often unfeasible due to the extensive hardware requirements for finetuning, even when utilizing parameter-efficient approaches such as prompt tuning. Additionally, the most powerful multi-task LLMs, such as OPT-IML-175B and FLAN-PaLM-540B, are not publicly accessible, severely limiting their customization potential. To address these challenges, we introduce a pretrained small scorer, Cappy, designed to enhance the performance and efficiency of multi-task LLMs. With merely 360 million parameters, Cappy functions either independently on classification tasks or serve as an auxiliary component for LLMs, boosting their performance. Moreover, Cappy enables efficiently integrating downstream supervision without requiring LLM finetuning nor the access to their parameters. Our experiments demonstrate that, when working independently on 11 language understanding tasks from PromptSource, Cappy outperforms LLMs that are several orders of magnitude larger. Besides, on 45 complex tasks from BIG-Bench, Cappy boosts the performance of the advanced multi-task LLM, FLAN-T5, by a large margin. Furthermore, Cappy is flexible to cooperate with other LLM adaptations, including finetuning and in-context learning, offering additional performance enhancement.

{{</citation>}}


## cs.CY (2)



### (47/63) Empowering Learning: Standalone, Browser-Only Courses for Seamless Education (Babak Moghadas et al., 2023)

{{<citation>}}

Babak Moghadas, Brian S. Caffo. (2023)  
**Empowering Learning: Standalone, Browser-Only Courses for Seamless Education**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06961v1)  

---


**ABSTRACT**  
Massive Open Online Courses (MOOCs) have transformed the educational landscape, offering scalable and flexible learning opportunities, particularly in data-centric fields like data science and artificial intelligence. Incorporating AI and data science into MOOCs is a potential means of enhancing the learning experience through adaptive learning approaches. In this context, we introduce PyGlide, a proof-of-concept open-source MOOC delivery system that underscores autonomy, transparency, and collaboration in maintaining course content. We provide a user-friendly, step-by-step guide for PyGlide, emphasizing its distinct advantage of not requiring any local software installation for students. Highlighting its potential to enhance accessibility, inclusivity, and the manageability of course materials, we showcase PyGlide's practical application in a continuous integration pipeline on GitHub. We believe that PyGlide charts a promising course for the future of open-source MOOCs, effectively addressing crucial challenges in online education.

{{</citation>}}


### (48/63) Simulating Public Administration Crisis: A Novel Generative Agent-Based Simulation System to Lower Technology Barriers in Social Science Research (Bushi Xiao et al., 2023)

{{<citation>}}

Bushi Xiao, Ziyuan Yin, Zixuan Shan. (2023)  
**Simulating Public Administration Crisis: A Novel Generative Agent-Based Simulation System to Lower Technology Barriers in Social Science Research**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2311.06957v1)  

---


**ABSTRACT**  
This article proposes a social simulation paradigm based on the GPT-3.5 large language model. It involves constructing Generative Agents that emulate human cognition, memory, and decision-making frameworks, along with establishing a virtual social system capable of stable operation and an insertion mechanism for standardized public events. The project focuses on simulating a township water pollution incident, enabling the comprehensive examination of a virtual government's response to a specific public administration event. Controlled variable experiments demonstrate that the stored memory in generative agents significantly influences both individual decision-making and social networks.   The Generative Agent-Based Simulation System introduces a novel approach to social science and public administration research. Agents exhibit personalized customization, and public events are seamlessly incorporated through natural language processing. Its high flexibility and extensive social interaction render it highly applicable in social science investigations. The system effectively reduces the complexity associated with building intricate social simulations while enhancing its interpretability.

{{</citation>}}


## cs.RO (3)



### (49/63) Multimodal Learning of Soft Robot Dynamics using Differentiable Filters (Xiao Liu et al., 2023)

{{<citation>}}

Xiao Liu, Yifan Zhou, Shuhei Ikemoto, Heni Ben Amor. (2023)  
**Multimodal Learning of Soft Robot Dynamics using Differentiable Filters**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.06954v1)  

---


**ABSTRACT**  
Differentiable Filters, as recursive Bayesian estimators, possess the ability to learn complex dynamics by deriving state transition and measurement models exclusively from data. This data-driven approach eliminates the reliance on explicit analytical models while maintaining the essential algorithmic components of the filtering process. However, the gain mechanism remains non-differentiable, limiting its adaptability to specific task requirements and contextual variations. To address this limitation, this paper introduces an innovative approach called {\alpha}-MDF (Attention-based Multimodal Differentiable Filter). {\alpha}-MDF leverages modern attention mechanisms to learn multimodal latent representations for accurate state estimation in soft robots. By incorporating attention mechanisms, {\alpha}-MDF offers the flexibility to tailor the gain mechanism to the unique nature of the task and context. The effectiveness of {\alpha}-MDF is validated through real-world state estimation tasks on soft robots. Our experimental results demonstrate significant reductions in state estimation errors, consistently surpassing differentiable filter baselines by up to 45% in the domain of soft robotics.

{{</citation>}}


### (50/63) Model-assisted Reinforcement Learning of a Quadrotor (Arshad Javeed, 2023)

{{<citation>}}

Arshad Javeed. (2023)  
**Model-assisted Reinforcement Learning of a Quadrotor**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06914v1)  

---


**ABSTRACT**  
In recent times, reinforcement learning has produced baffling results when it comes to performing control tasks with highly non-linear systems. The impressive results always outweigh the potential vulnerabilities or uncertainties associated with the agents when deployed in the real-world. While the performance is remarkable compared to the classical control algorithms, the reinforcement learning-based methods suffer from two flaws, robustness and interpretability, which are vital for contemporary real-world applications. The paper attempts to alleviate such problems with reinforcement learning and proposes the concept of model-assisted reinforcement learning to induce a notion of conservativeness in the agents. The control task considered for the experiment involves navigating a CrazyFlie quadrotor. The paper also describes a way of reformulating the task to have the flexibility of tuning the level of conservativeness via multi-objective reinforcement learning. The results include a comparison of the vanilla reinforcement learning approaches and the proposed approach. The metrics are evaluated by systematically injecting disturbances to classify the inherent robustness and conservativeness of the agents. More concrete arguments are made by computing and comparing the backward reachability tubes of the RL policies by solving the Hamilton-Jacobi-Bellman partial differential equation (HJ PDE).

{{</citation>}}


### (51/63) Towards Continual Reinforcement Learning for Quadruped Robots (Giovanni Minelli et al., 2023)

{{<citation>}}

Giovanni Minelli, Vassilis Vassiliades. (2023)  
**Towards Continual Reinforcement Learning for Quadruped Robots**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06828v1)  

---


**ABSTRACT**  
Quadruped robots have emerged as an evolving technology that currently leverages simulators to develop a robust controller capable of functioning in the real-world without the need for further training. However, since it is impossible to predict all possible real-world situations, our research explores the possibility of enabling them to continue learning even after their deployment. To this end, we designed two continual learning scenarios, sequentially training the robot on different environments while simultaneously evaluating its performance across all of them. Our approach sheds light on the extent of both forward and backward skill transfer, as well as the degree to which the robot might forget previously acquired skills. By addressing these factors, we hope to enhance the adaptability and performance of quadruped robots in real-world scenarios.

{{</citation>}}


## eess.SY (1)



### (52/63) TSViT: A Time Series Vision Transformer for Fault Diagnosis (Shouhua Zhang et al., 2023)

{{<citation>}}

Shouhua Zhang, Jiehan Zhou, Xue Ma, Chenglin Wen, Susanna Pirttikangas, Chen Yu, Weishan Zhang, Chunsheng Yang. (2023)  
**TSViT: A Time Series Vision Transformer for Fault Diagnosis**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-SY, eess-SY, eess.SY  
Keywords: Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06916v1)  

---


**ABSTRACT**  
Traditional fault diagnosis methods using Convolutional Neural Networks (CNNs) face limitations in capturing temporal features (i.e., the variation of vibration signals over time). To address this issue, this paper introduces a novel model, the Time Series Vision Transformer (TSViT), specifically designed for fault diagnosis. On one hand, TSViT model integrates a convolutional layer to segment vibration signals and capture local features. On the other hand, it employs a transformer encoder to learn long-term temporal information. The experimental results with other methods on two distinct datasets validate the effectiveness and generalizability of TSViT with a comparative analysis of its hyperparameters' impact on model performance, computational complexity, and overall parameter quantity. TSViT reaches average accuracies of 100% and 99.99% on two test sets, correspondingly.

{{</citation>}}


## cs.AR (1)



### (53/63) EPIM: Efficient Processing-In-Memory Accelerators based on Epitome (Chenyu Wang et al., 2023)

{{<citation>}}

Chenyu Wang, Zhen Dong, Daquan Zhou, Zhenhua Zhu, Yu Wang, Jiashi Feng, Kurt Keutzer. (2023)  
**EPIM: Efficient Processing-In-Memory Accelerators based on Epitome**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-LG, cs.AR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.07620v1)  

---


**ABSTRACT**  
The exploration of Processing-In-Memory (PIM) accelerators has garnered significant attention within the research community. However, the utilization of large-scale neural networks on Processing-In-Memory (PIM) accelerators encounters challenges due to constrained on-chip memory capacity. To tackle this issue, current works explore model compression algorithms to reduce the size of Convolutional Neural Networks (CNNs). Most of these algorithms either aim to represent neural operators with reduced-size parameters (e.g., quantization) or search for the best combinations of neural operators (e.g., neural architecture search). Designing neural operators to align with PIM accelerators' specifications is an area that warrants further study. In this paper, we introduce the Epitome, a lightweight neural operator offering convolution-like functionality, to craft memory-efficient CNN operators for PIM accelerators (EPIM). On the software side, we evaluate epitomes' latency and energy on PIM accelerators and introduce a PIM-aware layer-wise design method to enhance their hardware efficiency. We apply epitome-aware quantization to further reduce the size of epitomes. On the hardware side, we modify the datapath of current PIM accelerators to accommodate epitomes and implement a feature map reuse technique to reduce computation cost. Experimental results reveal that our 3-bit quantized EPIM-ResNet50 attains 71.59% top-1 accuracy on ImageNet, reducing crossbar areas by 30.65 times. EPIM surpasses the state-of-the-art pruning methods on PIM.

{{</citation>}}


## cs.SI (1)



### (54/63) Spotting Fake Profiles in Social Networks via Keystroke Dynamics (Alvin Kuruvilla et al., 2023)

{{<citation>}}

Alvin Kuruvilla, Rojanaye Daley, Rajesh Kumar. (2023)  
**Spotting Fake Profiles in Social Networks via Keystroke Dynamics**  

---
Primary Category: cs.SI  
Categories: cs-HC, cs-SI, cs.SI  
Keywords: Social Network, Twitter  
[Paper Link](http://arxiv.org/abs/2311.06903v1)  

---


**ABSTRACT**  
Spotting and removing fake profiles could curb the menace of fake news in society. This paper, thus, investigates fake profile detection in social networks via users' typing patterns. We created a novel dataset of 468 posts from 26 users on three social networks: Facebook, Instagram, and X (previously Twitter) over six sessions. Then, we extract a series of features from keystroke timings and use them to predict whether two posts originated from the same users using three prominent statistical methods and their score-level fusion. The models' performance is evaluated under same, cross, and combined-cross-platform scenarios. We report the performance using k-rank accuracy for k varying from 1 to 5. The best-performing model obtained accuracies between 91.6-100% on Facebook (Fusion), 70.8-87.5% on Instagram (Fusion), and 75-87.5% on X (Fusion) for k from 1 to 5. Under a cross-platform scenario, the fusion model achieved mean accuracies of 79.1-91.6%, 87.5-91.6%, and 83.3-87.5% when trained on Facebook, Instagram, and Twitter posts, respectively. In combined cross-platform, which involved mixing two platforms' data for model training while testing happened on the third platform's data, the best model achieved accuracy ranges of 75-95.8% across different scenarios. The results highlight the potential of the presented method in uncovering fake profiles across social network platforms.

{{</citation>}}


## cs.HC (3)



### (55/63) Anticipating User Needs: Insights from Design Fiction on Conversational Agents for Computational Thinking (Jacob Penney et al., 2023)

{{<citation>}}

Jacob Penney, João Felipe Pimentel, Igor Steinmacher, Marco A. Gerosa. (2023)  
**Anticipating User Needs: Insights from Design Fiction on Conversational Agents for Computational Thinking**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06887v1)  

---


**ABSTRACT**  
Computational thinking, and by extension, computer programming, is notoriously challenging to learn. Conversational agents and generative artificial intelligence (genAI) have the potential to facilitate this learning process by offering personalized guidance, interactive learning experiences, and code generation. However, current genAI-based chatbots focus on professional developers and may not adequately consider educational needs. Involving educators in conceiving educational tools is critical for ensuring usefulness and usability. We enlisted \numParticipants{} instructors to engage in design fiction sessions in which we elicited abilities such a conversational agent supported by genAI should display. Participants envisioned a conversational agent that guides students stepwise through exercises, tuning its method of guidance with an awareness of the educational background, skills and deficits, and learning preferences. The insights obtained in this paper can guide future implementations of tutoring conversational agents oriented toward teaching computational thinking and computer programming.

{{</citation>}}


### (56/63) Evaluating the Efficacy of Interactive Language Therapy Based on LLM for High-Functioning Autistic Adolescent Psychological Counseling (Yujin Cho et al., 2023)

{{<citation>}}

Yujin Cho, Mingeon Kim, Seojin Kim, Oyun Kwon, Ryan Donghan Kwon, Yoonha Lee, Dohyun Lim. (2023)  
**Evaluating the Efficacy of Interactive Language Therapy Based on LLM for High-Functioning Autistic Adolescent Psychological Counseling**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.09243v1)  

---


**ABSTRACT**  
This study investigates the efficacy of Large Language Models (LLMs) in interactive language therapy for high-functioning autistic adolescents. With the rapid advancement of artificial intelligence, particularly in natural language processing, LLMs present a novel opportunity to augment traditional psychological counseling methods. This research primarily focuses on evaluating the LLM's ability to engage in empathetic, adaptable, and contextually appropriate interactions within a therapeutic setting. A comprehensive evaluation was conducted by a panel of clinical psychologists and psychiatrists using a specially developed scorecard. The assessment covered various aspects of the LLM's performance, including empathy, communication skills, adaptability, engagement, and the ability to establish a therapeutic alliance. The study avoided direct testing with patients, prioritizing privacy and ethical considerations, and instead relied on simulated scenarios to gauge the LLM's effectiveness. The results indicate that LLMs hold significant promise as supportive tools in therapy, demonstrating strengths in empathetic engagement and adaptability in conversation. However, challenges in achieving the depth of personalization and emotional understanding characteristic of human therapists were noted. The study also highlights the importance of ethical considerations in the application of AI in therapeutic contexts. This research provides valuable insights into the potential and limitations of using LLMs in psychological counseling for autistic adolescents. It lays the groundwork for future explorations into AI's role in mental health care, emphasizing the need for ongoing development to enhance the capabilities of these models in therapeutic settings.

{{</citation>}}


### (57/63) Conversational Data Exploration: A Game-Changer for Designing Data Science Pipelines (Genoveva Vargas-Solar et al., 2023)

{{<citation>}}

Genoveva Vargas-Solar, Tania Cerquitelli, Javier A. Espinosa-Oviedo, François Cheval, Anthelme Buchaille, Luca Polgar. (2023)  
**Conversational Data Exploration: A Game-Changer for Designing Data Science Pipelines**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-DB, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.06695v1)  

---


**ABSTRACT**  
This paper proposes a conversational approach implemented by the system Chatin for driving an intuitive data exploration experience. Our work aims to unlock the full potential of data analytics and artificial intelligence with a new generation of data science solutions. Chatin is a cutting-edge tool that democratises access to AI-driven solutions, empowering non-technical users from various disciplines to explore data and extract knowledge from it.

{{</citation>}}


## cs.IR (2)



### (58/63) Modeling User Viewing Flow using Large Language Models for Article Recommendation (Zhenghao Liu et al., 2023)

{{<citation>}}

Zhenghao Liu, Zulong Chen, Moufeng Zhang, Shaoyang Duan, Hong Wen, Liangyue Li, Nan Li, Yu Gu, Ge Yu. (2023)  
**Modeling User Viewing Flow using Large Language Models for Article Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.07619v1)  

---


**ABSTRACT**  
This paper proposes the User Viewing Flow Modeling (SINGLE) method for the article recommendation task, which models the user constant preference and instant interest from user-clicked articles. Specifically, we employ a user constant viewing flow modeling method to summarize the user's general interest to recommend articles. We utilize Large Language Models (LLMs) to capture constant user preferences from previously clicked articles, such as skills and positions. Then we design the user instant viewing flow modeling method to build interactions between user-clicked article history and candidate articles. It attentively reads the representations of user-clicked articles and aims to learn the user's different interest views to match the candidate article. Our experimental results on the Alibaba Technology Association (ATA) website show the advantage of SINGLE, which achieves 2.4% improvements over previous baseline models in the online A/B test. Our further analyses illustrate that SINGLE has the ability to build a more tailored recommendation system by mimicking different article viewing behaviors of users and recommending more appropriate and diverse articles to match user interests.

{{</citation>}}


### (59/63) CL-Flow:Strengthening the Normalizing Flows by Contrastive Learning for Better Anomaly Detection (Shunfeng Wang et al., 2023)

{{<citation>}}

Shunfeng Wang, Yueyang Li, Haichi Luo, Chenyang Bi. (2023)  
**CL-Flow:Strengthening the Normalizing Flows by Contrastive Learning for Better Anomaly Detection**  

---
Primary Category: cs.IR  
Categories: cs-CV, cs-IR, cs-LG, cs.IR  
Keywords: Anomaly Detection, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.06794v1)  

---


**ABSTRACT**  
In the anomaly detection field, the scarcity of anomalous samples has directed the current research emphasis towards unsupervised anomaly detection. While these unsupervised anomaly detection methods offer convenience, they also overlook the crucial prior information embedded within anomalous samples. Moreover, among numerous deep learning methods, supervised methods generally exhibit superior performance compared to unsupervised methods. Considering the reasons mentioned above, we propose a self-supervised anomaly detection approach that combines contrastive learning with 2D-Flow to achieve more precise detection outcomes and expedited inference processes. On one hand, we introduce a novel approach to anomaly synthesis, yielding anomalous samples in accordance with authentic industrial scenarios, alongside their surrogate annotations. On the other hand, having obtained a substantial number of anomalous samples, we enhance the 2D-Flow framework by incorporating contrastive learning, leveraging diverse proxy tasks to fine-tune the network. Our approach enables the network to learn more precise mapping relationships from self-generated labels while retaining the lightweight characteristics of the 2D-Flow. Compared to mainstream unsupervised approaches, our self-supervised method demonstrates superior detection accuracy, fewer additional model parameters, and faster inference speed. Furthermore, the entire training and inference process is end-to-end. Our approach showcases new state-of-the-art results, achieving a performance of 99.6\% in image-level AUROC on the MVTecAD dataset and 96.8\% in image-level AUROC on the BTAD dataset.

{{</citation>}}


## eess.IV (1)



### (60/63) Osteoporosis Prediction from Hand and Wrist X-rays using Image Segmentation and Self-Supervised Learning (Hyungeun Lee et al., 2023)

{{<citation>}}

Hyungeun Lee, Ung Hwang, Seungwon Yu, Chang-Hun Lee, Kijung Yoon. (2023)  
**Osteoporosis Prediction from Hand and Wrist X-rays using Image Segmentation and Self-Supervised Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.06834v1)  

---


**ABSTRACT**  
Osteoporosis is a widespread and chronic metabolic bone disease that often remains undiagnosed and untreated due to limited access to bone mineral density (BMD) tests like Dual-energy X-ray absorptiometry (DXA). In response to this challenge, current advancements are pivoting towards detecting osteoporosis by examining alternative indicators from peripheral bone areas, with the goal of increasing screening rates without added expenses or time. In this paper, we present a method to predict osteoporosis using hand and wrist X-ray images, which are both widely accessible and affordable, though their link to DXA-based data is not thoroughly explored. Initially, our method segments the ulnar, radius, and metacarpal bones using a foundational model for image segmentation. Then, we use a self-supervised learning approach to extract meaningful representations without the need for explicit labels, and move on to classify osteoporosis in a supervised manner. Our method is evaluated on a dataset with 192 individuals, cross-referencing their verified osteoporosis conditions against the standard DXA test. With a notable classification score (AUC=0.83), our model represents a pioneering effort in leveraging vision-based techniques for osteoporosis identification from the peripheral skeleton sites.

{{</citation>}}


## cs.NI (1)



### (61/63) MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning (Duo Wu et al., 2023)

{{<citation>}}

Duo Wu, Panlong Wu, Miao Zhang, Fangxin Wang. (2023)  
**MANSY: Generalizing Neural Adaptive Immersive Video Streaming With Ensemble and Representation Learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06812v1)  

---


**ABSTRACT**  
The popularity of immersive videos has prompted extensive research into neural adaptive tile-based streaming to optimize video transmission over networks with limited bandwidth. However, the diversity of users' viewing patterns and Quality of Experience (QoE) preferences has not been fully addressed yet by existing neural adaptive approaches for viewport prediction and bitrate selection. Their performance can significantly deteriorate when users' actual viewing patterns and QoE preferences differ considerably from those observed during the training phase, resulting in poor generalization. In this paper, we propose MANSY, a novel streaming system that embraces user diversity to improve generalization. Specifically, to accommodate users' diverse viewing patterns, we design a Transformer-based viewport prediction model with an efficient multi-viewport trajectory input output architecture based on implicit ensemble learning. Besides, we for the first time combine the advanced representation learning and deep reinforcement learning to train the bitrate selection model to maximize diverse QoE objectives, enabling the model to generalize across users with diverse preferences. Extensive experiments demonstrate that MANSY outperforms state-of-the-art approaches in viewport prediction accuracy and QoE improvement on both trained and unseen viewing patterns and QoE preferences, achieving better generalization.

{{</citation>}}


## cs.IT (1)



### (62/63) Meta-Reinforcement Learning for Timely and Energy-efficient Data Collection in Solar-powered UAV-assisted IoT Networks (Mengjie Yi et al., 2023)

{{<citation>}}

Mengjie Yi, Xijun Wang, Juan Liu, Yan Zhang, Ronghui Hou. (2023)  
**Meta-Reinforcement Learning for Timely and Energy-efficient Data Collection in Solar-powered UAV-assisted IoT Networks**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.06742v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) have the potential to greatly aid Internet of Things (IoT) networks in mission-critical data collection, thanks to their flexibility and cost-effectiveness. However, challenges arise due to the UAV's limited onboard energy and the unpredictable status updates from sensor nodes (SNs), which impact the freshness of collected data. In this paper, we investigate the energy-efficient and timely data collection in IoT networks through the use of a solar-powered UAV. Each SN generates status updates at stochastic intervals, while the UAV collects and subsequently transmits these status updates to a central data center. Furthermore, the UAV harnesses solar energy from the environment to maintain its energy level above a predetermined threshold. To minimize both the average age of information (AoI) for SNs and the energy consumption of the UAV, we jointly optimize the UAV trajectory, SN scheduling, and offloading strategy. Then, we formulate this problem as a Markov decision process (MDP) and propose a meta-reinforcement learning algorithm to enhance the generalization capability. Specifically, the compound-action deep reinforcement learning (CADRL) algorithm is proposed to handle the discrete decisions related to SN scheduling and the UAV's offloading policy, as well as the continuous control of UAV flight. Moreover, we incorporate meta-learning into CADRL to improve the adaptability of the learned policy to new tasks. To validate the effectiveness of our proposed algorithms, we conduct extensive simulations and demonstrate their superiority over other baseline algorithms.

{{</citation>}}


## physics.chem-ph (1)



### (63/63) ReactionT5: a large-scale pre-trained model towards application of limited reaction data (Tatsuya Sagawa et al., 2023)

{{<citation>}}

Tatsuya Sagawa, Ryosuke Kojima. (2023)  
**ReactionT5: a large-scale pre-trained model towards application of limited reaction data**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph  
Keywords: T5, Transformer  
[Paper Link](http://arxiv.org/abs/2311.06708v1)  

---


**ABSTRACT**  
Transformer-based deep neural networks have revolutionized the field of molecular-related prediction tasks by treating molecules as symbolic sequences. These models have been successfully applied in various organic chemical applications by pretraining them with extensive compound libraries and subsequently fine-tuning them with smaller in-house datasets for specific tasks. However, many conventional methods primarily focus on single molecules, with limited exploration of pretraining for reactions involving multiple molecules. In this paper, we propose ReactionT5, a novel model that leverages pretraining on the Open Reaction Database (ORD), a publicly available large-scale resource. We further fine-tune this model for yield prediction and product prediction tasks, demonstrating its impressive performance even with limited fine-tuning data compared to traditional models. The pre-trained ReactionT5 model is publicly accessible on the Hugging Face platform.

{{</citation>}}
