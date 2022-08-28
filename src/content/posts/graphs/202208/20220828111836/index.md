---
draft: false
title: "夫婦別子供の有無別生活時間 (2016)"
date: 2022-08-28 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-時間貧困", "Graph-社会生活基本調査", "Graph-生活時間"]
menu:
  sidebar:
    name: 夫婦別子供の有無別生活時間 (2016)
    identifier: 20220828_graph
    parent: 202208_graphs
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1120pt" height="850pt" style="border:none"></iframe>
    <figcaption>夫婦別子供の有無別生活時間 (2016)</figcaption>
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

import numpy as np
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

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.28/社会生活基本調査_行動の種類別総平均時間_2016.xlsx", header=0, index_col=None)
data = data.set_index(["曜日", "夫婦", "子供有無", "共働きか否か・夫と妻の雇用形態"])
data = data.xs("週全体", axis=0, level=0).xs("総数", axis=0, level=2)
data = data[[
    "01_睡眠", "02_身の回りの用事", "03_食事",
    "04_通勤・通学", "05_仕事", "06_学業", "07_家事", "08_介護・看護", "09_育児", "10_買い物",
     "11_移動(通勤・通学を除く)", "12_テレビ・ラジオ・新聞・雑誌", "13_休養・くつろぎ", "14_学習・自己啓発・訓練(学業以外)", "15_趣味・娯楽", "16_スポーツ", "17_ボランティア活動・社会参加活動", "18_交際・付き合い", "19_受診・療養", "20_その他"
]]

fig = make_subplots(
    rows=2,
    cols=3,
    specs=[[{"type": "pie"}]*3]*2,
    subplot_titles=[
        "夫（子供なし）", "夫（子育て期）", "夫（子供と同居）",
        "妻（子供なし）", "妻（子育て期）", "妻（子供と同居）",
        ],
    horizontal_spacing=0.15,
    vertical_spacing=0.1
    )
params = [
    ["夫", "子供なし", 1, 1], ["夫", "子育て期", 1, 2], ["夫", "子供と同居", 1, 3],
    ["妻", "子供なし", 2, 1], ["妻", "子育て期", 2, 2], ["妻", "子供と同居", 2, 3],
]
category_map = {
    '01_睡眠'                               : "1次活動",
    '02_身の回りの用事'                     : "1次活動", 
    '03_食事'                               : "1次活動", 
    '04_通勤・通学'                         : "2次活動",
    '05_仕事'                               : "2次活動",
    '06_学業'                               : "2次活動",
    '07_家事'                               : "2次活動",
    '08_介護・看護'                         : "2次活動",
    '09_育児'                               : "2次活動",
    '10_買い物'                             : "2次活動",
    '11_移動(通勤・通学を除く)'             : "3次活動",
    '12_テレビ・ラジオ・新聞・雑誌'         : "3次活動",
    '13_休養・くつろぎ'                     : "3次活動",
    '14_学習・自己啓発・訓練(学業以外)'     : "3次活動",
    '15_趣味・娯楽'                         : "3次活動",
    '16_スポーツ'                           : "3次活動",
    '17_ボランティア活動・社会参加活動'     : "3次活動",
    '18_交際・付き合い'                     : "3次活動",
    '19_受診・療養'                         : "3次活動",
    '20_その他'                             : "3次活動",
    '1次活動'                               : "",
    '2次活動'                               : "",
    '3次活動'                               : "",
}

for param in params:
    activities = data.loc[param[0], :].loc[param[1], :].reset_index()
    activities.columns = ["index", "value"]
    first_act = sum([row["value"] for idx, row in activities.iterrows() if category_map[row["index"]]=="1次活動"])
    second_act = sum([row["value"] for idx, row in activities.iterrows() if category_map[row["index"]]=="2次活動"])
    third_act = sum([row["value"] for idx, row in activities.iterrows() if category_map[row["index"]]=="3次活動"])
    activities = pd.concat([
        activities, 
        pd.DataFrame([{"index": "1次活動", "value": first_act}]),
        pd.DataFrame([{"index": "2次活動", "value": second_act}]),
        pd.DataFrame([{"index": "3次活動", "value": third_act}]),
        ])
    fig.add_trace(
        go.Sunburst(
            labels=activities["index"].values.tolist(),
            parents=[category_map[col] for col in activities["index"].values.tolist()],
            values=activities["value"].values.tolist(),
            branchvalues="total"
        ),
        row=param[2],
        col=param[3]
    )

fig.update_layout(title="夫婦別子供の有無別生活時間 (2016)")
fig.update_layout(width=1100, height=800)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

## Source
{{< citation >}}
Data From: [社会生活基本調査 (2016)](https://www.e-stat.go.jp/stat-search?page=1&toukei=00200533&survey=%E7%A4%BE%E4%BC%9A%E7%94%9F%E6%B4%BB%E5%9F%BA%E6%9C%AC%E8%AA%BF%E6%9F%BB)
{{< /citation >}}
