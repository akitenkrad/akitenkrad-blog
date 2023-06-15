---
draft: false
title: "Bandit Algorithm Basic"
date: 2023-06-15
author: "akitenkrad"
description: ""
tags: ["Algorithm"]
menu:
  sidebar:
    name: "Bandit Algorithm Basic"
    identifier: bandit_algorithm_basic
    parent: algorithms
    weight: 10
math: true
---

## $\epsilon$-greedy algorithm

> <u>**Parameters**</u>:  
$\quad \text{Arms}: \lbrace a_1, a_2, \ldots a_A, \rbrace$  
$\quad \text{Rewards}: \lbrace a_1: \lbrace r_{11}, r_{12}, \ldots, r_{1R} \rbrace, a_2: \lbrace r_{12}, r_{22}, \ldots, r_{2R} \rbrace, \ldots, a_A: \lbrace r_{A1}, r_{A2}, \ldots, r_{AR} \rbrace \rbrace$  
$\quad \text{States}: \lbrace a_1: \lbrace p_{11}, p_{12}, \ldots, p_{1R} \rbrace, a_2: \lbrace p_{12}, p_{22}, \ldots, p_{2R} \rbrace, \ldots, a_A: \lbrace p_{A1}, p_{A2}, \ldots, p_{AR} \rbrace \rbrace$  
$\quad \text{Strategies}: \lbrace s_1, s_2, \ldots, s_A \rbrace$  
$\quad \epsilon : \text{Const}$  
>  
> <u>**Functions**</u>:  
$\quad$ Bandit.environment(arm):  
$\qquad \lbrace p_1, p_2, \ldots, p_R \rbrace = \text{States} \lbrack\text{arm}\rbrack$  
$\qquad \lbrace r_1, r_2, \ldots, r_R \rbrace = \text{Rewards} \lbrack\text{arm}\rbrack$  
$\qquad r = \text{random}(\text{from}=\lbrace r_1, r_2, \ldots, r_R \rbrace, \text{prob}=\lbrace p_1, p_2, \ldots p_R \rbrace)$  
$\qquad \text{return} \quad r$  
>  
>$\quad$ Agent.update(arm, reward):  
$\qquad \text{Strategies} \lbrack \text{arm} \rbrack = \text{update}(\text{reward})$  
>  
>$\quad$ Agent.get_action():  
$\qquad \text{if} \quad \text{random()} < \epsilon:$  
$\qquad\quad \text{return} \quad \text{random}(\text{Strategies})$  
$\qquad \text{else}:$  
$\qquad\quad \text{return} \quad \text{argmax}(\text{Strategies})$  
>  
> <u>**Algorithm**</u>:  
> Loop for step in range(steps):  
> $\quad$ action = Agent.get_action()  
> $\quad$ reward = Bandit.environment(action)  
> $\quad$ Agent.update(action, reward)

