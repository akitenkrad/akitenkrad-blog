---
draft: false
title: "総人口の推移"
date: 2022-08-26 
author: "akitenkrad"
description: ""
tags: ["Figures", "Figure-人口", "Figure-人口推計"]
menu:
  sidebar:
    name: 総人口の推移
    identifier: 20220826_figure
    parent: 202208_figures
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1120pt" height="1120pt" style="border:none"></iframe>
    <figcaption>総人口の推移</figcaption>
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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.26/総人口の推移（千人）.xlsx", header=0, index_col=[0,1])

from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
colors = get_colorpalette("tab10", 6)
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=["総人口の推移 (千人)", "総人口の推移 (割合)"],
    horizontal_spacing=0.15,
    vertical_spacing=0.15,
)

for idx, categ in enumerate(["15歳未満", "15〜64", "65歳以上"]):
    fig.add_trace(
        go.Bar(
            x=data.columns.tolist(),
            y=data.loc["実数値", :].loc[categ, :],
            name=categ + " (実数値)",
            marker_color=colors[2 * idx]
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=data.columns.tolist(),
            y=data.loc["割合", :].loc[categ, :],
            name=categ + " (割合)",
            marker_color=colors[2 * idx + 1],
        ),
        row=2, col=1
    )

fig.update_layout(
    barmode="stack",
    width=1100, height=1100,
    showlegend=True,
    template="seaborn"
)

fig.show()
```

## Source
{{< citation >}}
Data From: [人口推計](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200524&tstat=000000090001&cycle=0&tclass1=000000090004&tclass2=000001051180&tclass3val=0)
{{< /citation >}}
