---
draft: false
title: "東京23区の乗用車保有台数"
date: 2022-08-14 
author: "akitenkrad"
description: ""
tags: ["Figures", "Figure-車両データ"]
menu:
  sidebar:
    name: 東京23区の乗用車保有台数
    identifier: 20220814_figure
    parent: 202208_figures
    weight: 10
math: true
---

## Graph
<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>東京23区の乗用車保有台数</figcaption>
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

import json
import pandas as pd

import matplotlib.pyplot as plt
import japanize_matplotlib 
import seaborn as sns
import plotly.graph_objects as go

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# Load data
data = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.14/000264266.xlsx", header=0, index_col=[0, 1])
gov_code = {row["団体名"]: str(row["団体コード"]) for _, row in pd.read_html("https://www.j-lis.go.jp/spd/code-address/kantou/cms_13414181.html")[0].iterrows()}

cars = data.xs("自家用", level=1)
cars["区分"] = cars.index
cars["行政コード"] = [gov_code.get(idx, "") for idx in cars.index]
cars = cars[cars["行政コード"] != ""]

# Load geojson
tokyo = json.load(open(f"/content/land/tokyo.geojson"))
for geometry in tokyo["features"]:
    geometry["id"] = str(geometry["properties"]["code"])

# Visualize
fig = go.Figure(
    go.Choroplethmapbox(
        geojson=tokyo,
        locations=cars["行政コード"],
        z=cars["乗用計"],
        colorscale="YlOrRd",
        marker_opacity=0.5,
        marker_line_width=0,
        text=cars["区分"]
    )
)
fig.update_layout(
    mapbox_style="stamen-toner",
    mapbox_zoom=9.5,
    mapbox_center = {"lat": 35.700700, "lon": 139.458245}
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
```

## Source
{{< citation >}}
Data From: [関東運輸局](https://wwwtb.mlit.go.jp/kanto/jidou_gian/toukei/tiiki_betu.html)
{{< /citation >}}
