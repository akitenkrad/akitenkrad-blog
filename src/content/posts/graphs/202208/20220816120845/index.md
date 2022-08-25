---
draft: false
title: "都道府県別消費者物価指数/教育（全国平均=100）"
date: 2022-08-16 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-物価"]
menu:
  sidebar:
    name: 都道府県別消費者物価指数/教育（全国平均=100）
    identifier: 20220816_graph
    parent: 202208_graphs
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>都道府県別消費者物価指数/教育（全国平均=100）</figcaption>
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

!git clone https://github.com/akitenkrad/land

import pandas as pd
import json

import matplotlib.pyplot as plt
import japanize_matplotlib 
import seaborn as sns
import plotly.graph_objects as go

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.16/2020.xlsx", header=0, index_col=None)

geo_data = json.load(open("/content/land/japan.geojson"))
for geometry in geo_data["features"]:
    geometry["id"] = geometry["properties"]["id"] * 1000

fig = go.Figure(
    go.Choroplethmapbox(
        geojson=geo_data,
        locations=data["地域コード"],
        z=data["教育"],
        colorscale="YlOrRd",
        marker_opacity=0.5,
        marker_line_width=0,
        text=data["地域"]
    )
)
fig.update_layout(
    mapbox_style="stamen-toner",
    mapbox_zoom=4,
    mapbox_center = {"lat": 36.814279, "lon": 138.849672}
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
```

## Source
{{< citation >}}
Data From: [小売物価統計調査（構造編）](https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200571&tstat=000001067253&cycle=7&year=20200&month=0)
{{< /citation >}}
