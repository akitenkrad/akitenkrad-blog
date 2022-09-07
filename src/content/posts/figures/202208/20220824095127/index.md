---
draft: false
title: "有業者の社会生活における活動の変化"
date: 2022-08-24 
author: "akitenkrad"
description: ""
tags: ["Figures", "Figure-社会生活", "Figure-有業者", "Figure-時系列"]
menu:
  sidebar:
    name: 有業者の社会生活における活動の変化
    identifier: 20220824_figure
    parent: 202208_figures
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1130pt" height="1200pt" style="border:none"></iframe>
    <figcaption>有業者の社会生活における活動の変化</figcaption>
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

fig = make_subplots(
    rows=3,
    cols=3,
    specs=[[{"type": "pie"}]*3]*3,
    subplot_titles=[
        "平日 (1976)", "平日 (2001)", "平日 (2016)",
        "土曜日 (1976)", "土曜日 (2001)", "土曜日 (2016)",
        "日曜日 (1976)", "日曜日 (2001)", "日曜日 (2016)",
        ])
params = [
    {"day": "平日", "year": 1976, "row": 1, "col": 1},
    {"day": "平日", "year": 2001, "row": 1, "col": 2},
    {"day": "平日", "year": 2016, "row": 1, "col": 3},
    {"day": "土曜日", "year": 1976, "row": 2, "col": 1},
    {"day": "土曜日", "year": 2001, "row": 2, "col": 2},
    {"day": "土曜日", "year": 2016, "row": 2, "col": 3},
    {"day": "日曜日", "year": 1976, "row": 3, "col": 1},
    {"day": "日曜日", "year": 2001, "row": 3, "col": 2},
    {"day": "日曜日", "year": 2016, "row": 3, "col": 3},
]

for param in params:
    activities = data.loc["有業者", param["day"]].loc[:, param["year"]].reset_index()
    activities.columns = ["Lv0", "Lv1", "value"]
    first_activities = pd.DataFrame([{"Lv0": "", "Lv1": "1次活動", "value": data.loc["有業者", param["day"]].loc["1次活動", param["year"]].sum()}])
    second_activities = pd.DataFrame([{"Lv0": "", "Lv1": "2次活動", "value": data.loc["有業者", param["day"]].loc["2次活動", param["year"]].sum()}])
    third_activities = pd.DataFrame([{"Lv0": "", "Lv1": "3次活動", "value": data.loc["有業者", param["day"]].loc["3次活動", param["year"]].sum()}])
    activities = pd.concat([activities, first_activities, second_activities, third_activities], axis=0)

    fig.add_trace(
        go.Sunburst(
            labels=activities["Lv1"],
            parents=activities["Lv0"],
            values=activities["value"],
            branchvalues="total"
        ),
        row=param["row"],
        col=param["col"]
    )

fig.update_layout(title="有業者の社会活動の変化")
fig.update_layout(width=1100, height=1100)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

## Source
{{< citation >}}
Data From: [社会生活基本調査](https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00200533&tstat=000001095335)
{{< /citation >}}
