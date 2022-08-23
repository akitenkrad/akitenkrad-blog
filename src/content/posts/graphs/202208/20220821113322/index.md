---
draft: false
title: "地域別1世帯当たり1ヶ月間の支出額の推移"
date: 2022-08-21 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-支出額", "Graph-時系列"]
menu:
  sidebar:
    name: 地域別1世帯当たり1ヶ月間の支出額の推移
    identifier: 20220821_graph
    parent: 202208_graphs
    weight: 10
math: true
---

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>地域別1世帯当たり1ヶ月間の支出額の推移</figcaption>
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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.21/家計消費状況調査_H29.xlsx", header=[0,1], index_col=None, sheet_name="Data")

regions = ["北海道", "東北", "関東", "北陸", "東海", "近畿", "中国", "四国", "九州・沖縄"]
colors = get_colorpalette("tab10", len(regions))
target_idx = 41

fig = go.Figure()
for idx, region in enumerate(regions):
    
    _data = data.xs(region, axis=1, level=1).iloc[41, :]
    fig.add_trace(
        go.Scatter(
            x=_data.index,
            y=_data.values,
            name=region,
            marker=dict(color=colors[idx])
        )
    )

fig.update_yaxes(title="支出額（円）")
fig.update_layout(title="地域別1世帯当たり1ヶ月間の支出額の推移")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

{{< citation >}}
Data From: [家計消費状況調査](https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00200565)
{{< /citation >}}
