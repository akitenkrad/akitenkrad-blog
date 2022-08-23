---
draft: false
title: "1ヶ月当たりの実労働時間の推移"
date: 2022-08-22 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-労働時間", "Graph-時系列"]
menu:
  sidebar:
    name: 1ヶ月当たりの実労働時間の推移
    identifier: 20220822_graph
    parent: 202208_graphs
    weight: 10
math: true
---

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>1ヶ月当たりの実労働時間の推移</figcaption>
</figure>

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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.22/実労働時間推移.xlsx", header=[0,1,2], index_col=None)

years = data.loc[:, "年"].loc[:, "Unnamed: 0_level_1"].loc[:, "Unnamed: 0_level_2"].values.tolist()
morethan_5 = data.loc[:, "事業規模5人以上"].loc[:, "総実労働時間"].loc[:, "１か月当たり"].values.tolist()
morethan_30 = data.loc[:, "事業規模30人以上"].loc[:, "総実労働時間"].loc[:, "１か月当たり"].values.tolist()

colors = get_colorpalette("tab10", 2)
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=years,
        y=morethan_5,
        name="事業規模5人以上",
        marker=dict(color=colors[0])
    )
)
fig.add_trace(
    go.Scatter(
        x=years,
        y=morethan_30,
        name="事業規模30人以上",
        marker=dict(color=colors[1])
    )
)

fig.update_yaxes(title="労働時間 (h)")
fig.update_layout(title="1ヶ月当たりの実労働時間の推移")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

{{< citation >}}
Data From: [労働統計要覧](https://www.mhlw.go.jp/toukei/youran/indexyr_d.html)
{{< /citation >}}
