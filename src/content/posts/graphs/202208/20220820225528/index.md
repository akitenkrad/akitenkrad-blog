---
draft: false
title: "都道府県別国公立別学校数・学生数"
date: 2022-08-20 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-大学"]
menu:
  sidebar:
    name: 都道府県別国公立別学校数・学生数
    identifier: 20220820_graph
    parent: 202208_graphs
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out2.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>都道府県別国公立別学校数</figcaption>
</figure>

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>都道府県別国公立別学生数</figcaption>
</figure>

## Code
```python
!pip install -U kaleido
!pip install japanize-matplotlib

!apt -y install fonts-ipafont-gothic
import matplotlib
matplotlib.get_cachedir()
!ls /root/.cache/matplotlib
!rm /root/.cache/matplotlib/fontlist-v310.json

import pandas as pd
import json

import matplotlib.pyplot as plt
import japanize_matplotlib 
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def get_colorpalette(colorpalette, n_colors):
    """
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[int(x * 256) for x in _rgb]) for _rgb in palette]
    return rgb

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.20/hi0007.xlsx", header=[0,1,2], index_col=None)
students = data["学生数"].xs("計", level=1, axis=1)
students["区分"] = data["区分"]["区分"]["区分.1"]
students["国立_rate"] = students["国立"] / students["計"] * 100.0
students["公立_rate"] = students["公立"] / students["計"] * 100.0
students["私立_rate"] = students["私立"] / students["計"] * 100.0

schools = data.["学校数"]
schools["区分"] = data["区分"]["区分"]["区分.1"]
schools.columns = ["計", "国立", "公立", "私立", "区分"]
schools["国立_rate"] = schools["国立"] / schools["計"] * 100.0
schools["公立_rate"] = schools["公立"] / schools["計"] * 100.0
schools["私立_rate"] = schools["私立"] / schools["計"] * 100.0

colors = get_colorpalette("tab10", 6)
fig = make_subplots(specs=[[{"secondary_y": True}]])
for i, category in enumerate(["国立", "公立", "私立"]):
    fig.add_trace(
        go.Bar(
            name=category,
            x=students["区分"],
            y=students[category+"_rate"],
            marker_color=colors[i],
            opacity=0.3
        )
    )
    fig.add_trace(
        go.Scatter(
            name=category,
            x=students["区分"],
            y=students[category],
            marker_color=colors[i+3]
        ), secondary_y=True
    )

fig.update_layout(
    title="都道府県別国公立別学生数",
    barmode="stack"
)
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()

colors = get_colorpalette("tab10", 6)
fig = make_subplots(specs=[[{"secondary_y": True}]])
for i, category in enumerate(["国立", "公立", "私立"]):
    fig.add_trace(
        go.Bar(
            name=category,
            x=schools["区分"],
            y=schools[category+"_rate"],
            marker_color=colors[i],
            opacity=0.3
        )
    )
    fig.add_trace(
        go.Scatter(
            name=category,
            x=schools["区分"],
            y=schools[category],
            marker_color=colors[i+3]
        ), secondary_y=True
    )

fig.update_layout(
    title="都道府県別国公立別学校数",
    barmode="stack"
)
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

## Source
{{< citation >}}
Data From: [学校基本調査](https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00400001&tstat=000001011528)
{{< /citation >}}
