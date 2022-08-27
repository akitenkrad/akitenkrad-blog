---
draft: false
title: "家族類型の変遷"
date: 2022-08-27 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-家族類型", "Graph-国勢調査"]
menu:
  sidebar:
    name: 家族類型の変遷
    identifier: 20220827_graph
    parent: 202208_graphs
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1125pt" height="670pt" style="border:none"></iframe>
    <figcaption>家族類型の変遷</figcaption>
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
from pathlib import Path

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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.27/国勢調査_家族類型.xlsx", header=[0,1,2], index_col=[0,1])

colors = get_colorpalette("Set2", 3)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data.xs("00000_全国", axis=0, level=1).index.tolist(),
        y=data.xs("00000_全国", axis=0, level=1).xs("核家族世帯計", axis=1, level=2).values.squeeze().tolist(),
        name="核家族世帯",
        marker=dict(color=colors[0])
    )
)
fig.add_trace(
     go.Scatter(
        x=data.xs("00000_全国", axis=0, level=1).index.tolist(),
        y=(data.xs("00000_全国", axis=0, level=1).xs("核家族以外の世帯計", axis=1, level=2).values.squeeze() + 
           data.xs("00000_全国", axis=0, level=1).xs("非親族を含む世帯", axis=1, level=2).values.squeeze()).tolist(),
        name="核家族以外の複数世帯",
        marker=dict(color=colors[1])
    )
)
fig.add_trace(
    go.Scatter(
        x=data.xs("00000_全国", axis=0, level=1).index.tolist(),
        y=data.xs("00000_全国", axis=0, level=1).xs("単独世帯", axis=1, level=2).values.squeeze().tolist(),
        name="単独世帯",
        marker=dict(color=colors[2])
    )
)

fig.update_layout(
    width=1100, height=650,
    showlegend=True,
    template="seaborn"
)

fig.show()
```

## Source
{{< citation >}}
Data From: [国勢調査-世帯の家族類型](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200521&tstat=000001011777&cycle=0&tclass1=000001011805&tclass2val=0)
{{< /citation >}}
