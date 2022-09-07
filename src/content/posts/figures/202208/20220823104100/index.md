---
draft: false
title: "睡眠時間の推移"
date: 2022-08-23 
author: "akitenkrad"
description: ""
tags: ["Figures", "Figure-睡眠時間", "Figure-時系列"]
menu:
  sidebar:
    name: 睡眠時間の推移
    identifier: 20220823_figure
    parent: 202208_figures
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>睡眠時間の推移</figcaption>
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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.23/社会生活基本調査_行動の種類別平均時間の推移.xlsx", header=[0,1], index_col=[0,1,2])

colors = get_colorpalette("tab10", 3)
fig = go.Figure()

for color, day in zip(colors, ["平日", "土曜日", "日曜日"]):
    worker = data.loc["有業者", day].loc["1次活動", :].loc["睡眠", :]
    in_freedom = data.loc["無業者", day].loc["1次活動", :].loc["睡眠", :]

    fig.add_trace(
        go.Scatter(
            x=worker.index.tolist(),
            y=worker.values.tolist(),
            mode="lines",
            name=f"有業者 ({day})",
            line=dict(color=color)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=in_freedom.index.tolist(),
            y=in_freedom.values.tolist(),
            mode="lines",
            name=f"無業者 ({day})",
            line=dict(color=color, dash="dash")
        )
    )

fig.update_yaxes(title="睡眠時間時間 (h)")
fig.update_layout(
    title="睡眠時間の推移",
    width=1100,
    height=600,
    showlegend=True,
    template="seaborn"
    )
fig.show()
```

## Source
{{< citation >}}
Data From: [社会生活基本調査](https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00200533&tstat=000001095335)
{{< /citation >}}
