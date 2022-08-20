---
draft: false
title: "業種別企業特殊的人的資本 (2018)"
date: 2022-08-17
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-企業統計"]
menu:
  sidebar:
    name: 業種別企業特殊的人的資本 (2018)
    identifier: 20220817_graph
    parent: 202208_graphs
    weight: 10
math: true
---

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>業種別企業特殊的人的資本 (2018)</figcaption>
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

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def get_colorpalette(colorpalette, n_colors):
    """
    https://matplotlib.org/stable/tutorials/colors/colormaps.html
    """
    palette = sns.color_palette(colorpalette, n_colors)
    rgb = ['rgb({},{},{})'.format(*[int(x * 256) for x in _rgb]) for _rgb in palette]

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.17/jip2021_6.xlsx", sheet_name="名目_企業特殊的人的資本", header=0, index_col=0)
data["産業中分類"] = data["産業中分類"].apply(lambda x: f"{x}(中)")
sum_data = data.groupby("産業中分類").sum()
sum_data["産業大分類"] = ""
sum_data["産業中分類"] = ""
sum_data["部門分類"] = sum_data.index
data = pd.concat([data, sum_data], axis=0)
data = data.reset_index(drop=True)

fig = go.Figure()

fig.add_trace(
    go.Sunburst(
        labels=data["部門分類"],
        parents=data["産業中分類"],
        values=data[2018],
        branchvalues="total"
    )
)

fig.update_layout(title="業種別企業特殊的人的資本 (2018)")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

{{< citation >}}
Data From: [日本産業生産性（JIP）データベース](https://www.rieti.go.jp/jp/database/JIP2021/index.html#04)
{{< /citation >}}
