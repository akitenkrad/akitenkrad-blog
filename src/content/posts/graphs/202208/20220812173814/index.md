---
draft: false
title: "金沢市の夏の月平均気温の遷移"
date: 2022-08-12
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-気象庁"]
menu:
  sidebar:
    name: 金沢市の夏の月平均気温の遷移
    identifier: 20220812_graph
    parent: 202208_graphs
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>金沢市の夏の月平均気温の遷移</figcaption>
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

import matplotlib.pyplot as plt
import japanize_matplotlib 
import seaborn as sns
import plotly.graph_objects as go

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def get_colorpalette(colorpalette, n_colors):
    """
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[int(x * 256) for x in _rgb]) for _rgb in palette]

# Load data
data = pd.read_html("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.12/kanazawa_temperature.html")[0]

# Visualize
colors = get_colorpalette("tab10", 4)
plot_data = [
    {"y": "6月", "name": "June"},
    {"y": "7月", "name": "July"},
    {"y": "8月", "name": "August"},
    {"y": "9月", "name": "September"},
]
fig = go.Figure()
for idx, p_data in enumerate(plot_data):
    fig.add_trace(go.Scatter(x=data["年"], y=data[p_data["y"]], name=p_data["name"], marker=dict(color=colors[idx])))

fig.update_yaxes(title="月平均気温")
fig.update_layout(title="夏の月平均気温 @金沢")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

## Source
{{< citation >}}
Data From: [気象庁](https://www.data.jma.go.jp/obd/stats/etrn/index.php?prec_no=56&block_no=47605&year=&month=&day=&view=p1)
{{< /citation >}}