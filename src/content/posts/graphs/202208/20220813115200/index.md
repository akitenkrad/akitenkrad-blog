---
draft: false
title: "コンビニエンスストアの店舗数の推移"
date: 2022-08-13 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-コンビニエンスストア"]
menu:
  sidebar:
    name: コンビニエンスストアの店舗数の推移
    identifier: 20220813_graph
    parent: 202208_graphs
    weight: 10
math: true
---

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>コンビニエンスストアの店舗数の推移</figcaption>
</figure>

```python
!pip install -U kaleido
!pip install japanize-matplotlib

!apt -y install fonts-ipafont-gothic
import matplotlib
matplotlib.get_cachedir()
!ls /root/.cache/matplotlib
!rm /root/.cache/matplotlib/fontlist-v310.json

from datetime import datetime
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
    return rgb

# Load data
data_1 = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.13/20220120111716.xlsx", header=[0,1], index_col=[0,1])
data_2 = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.13/20220120111818.xlsx", header=[0,1], index_col=[0,1])
data = pd.concat([data_1, data_2])

new_data = []
for i in range(1, 13):
    m = f"{i}月"
    records = data.xs(m, level=1)
    records.index = [datetime(year, i, 1) for year in range(2005, 2022)]
    new_data.append(records)
new_data = pd.concat(new_data, axis=0)
new_data = new_data.sort_index()

# Visualize
colors = get_colorpalette("tab10", 1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=new_data.index, y=new_data[("店舗数（店）", "全店")], name="全店舗数", marker=dict(color=colors[0])))

fig.update_yaxes(title="店舗数")
fig.update_layout(title="コンビニエンスストアの店舗数（全店）の推移")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

{{< citation >}}
Data From: [日本フランチャイズチェーン協会](https://www.jfa-fc.or.jp/particle/320.html)
{{< /citation >}}
