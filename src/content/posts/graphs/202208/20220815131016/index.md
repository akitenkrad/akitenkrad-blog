---
draft: false
title: "年齢別大学院入学者数 (2021)"
date: 2022-08-15 
author: "akitenkrad"
description: ""
tags: ["Graphs", "Graph-大学"]
menu:
  sidebar:
    name: 年齢別大学院入学者数 (2021)
    identifier: 20220815_graph
    parent: 202208_graphs
    weight: 10
math: true
---

<figure style="width:100%; display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <iframe src="out.html" width="1110pt" height="650pt" style="border:none"></iframe>
    <figcaption>年齢別大学院入学者数 (2021)</figcaption>
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

data_m = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.15/2021.xlsx",sheet_name="修士", header=[0, 1], index_col=None)
data_d = pd.read_excel("/content/drive/MyDrive/DailyGraphs/2022.08/2022.08.15/2021.xlsx",sheet_name="博士", header=[0, 1], index_col=None)

master = data_m.loc[:, [("区分", "区分"), ("計", "計")]]
master.columns = ["age", "count"]
master["category"] = "修士課程"
master["age"] = master["age"].apply(lambda x: "M:" + x)
master = master.drop(master[master["age"]=="M:社会人"].index, axis=0)
master = master.drop(master[master["age"]=="M:留学生"].index, axis=0)

doctor = data_d.loc[:, [("区分", "区分"), ("計", "計")]]
doctor.columns = ["age", "count"]
doctor["category"] = "博士課程"
doctor["age"] = doctor["age"].apply(lambda x: "D:" + x)
doctor = doctor.drop(doctor[doctor["age"]=="D:社会人"].index, axis=0)
doctor = doctor.drop(doctor[doctor["age"]=="D:留学生"].index, axis=0)

total = pd.DataFrame([
    {"age": "修士課程", "count": master["count"].sum(), "category": ""},
    {"age": "博士課程", "count": doctor["count"].sum(), "category": ""},
    ])
data = pd.concat([total, master, doctor]).reset_index(drop=True)

fig = go.Figure()

fig.add_trace(
    go.Sunburst(
        labels=data["age"],
        parents=data["category"],
        values=data["count"],
        branchvalues="total"
    )
)

fig.update_layout(title="年齢別大学院入学者数 (2021)")
fig.update_layout(width=1100, height=600)
fig.update_layout(showlegend=True)
fig.update_layout(template="seaborn")
fig.show()
```

{{< citation >}}
Data From: [学校基本調査 from e-Stat](https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00400001&tstat=000001011528)
{{< /citation >}}
