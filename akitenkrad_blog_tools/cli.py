from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import click
import pandas as pd
from dateutil.parser import parse as parse_date
from nltk import FreqDist
from plotly import graph_objects as go
from tqdm import tqdm

from akitenkrad_blog_tools.arxiv_utils import get_arxiv_posts
from akitenkrad_blog_tools.ss_utils import add_references
from akitenkrad_blog_tools.utils import Paper


@click.group()
def cli():
    pass


@cli.command()
@click.option("--title", type=str, help="title of the paper", required=True)
@click.option("--without-citations", is_flag=True, help="create template without citations")
def new_paper(title: str, without_citations: bool):
    """create new post for papers"""
    assert len(title) > 0

    text = open(Path(__file__).parent / "templates" / "post_template.txt").read()
    date = datetime.now()
    text = text.format(
        TITLE=title,
        DATE=date.strftime("%Y-%m-%d"),
        NAME=datetime.now().strftime("%Y.%m.%d"),
        IDENTIFIER=datetime.now().strftime("%Y%m%d"),
        PARENT=date.strftime("%Y%m"),
    )
    new_post_path = Path(
        f"src/content/posts/papers/{date.strftime('%Y%m')}/{datetime.now().strftime('%Y%m%d%H%M%S')}/index.md"
    )

    if not new_post_path.parent.parent.exists():
        new_post_path.parent.parent.mkdir(parents=True)
        with open(new_post_path.parent.parent / "_index.md", mode="w", encoding="utf-8") as wf:
            wf.write(
                f"""---
title: {date.strftime("%Y.%m")}
menu:
    sidebar:
        name: {date.strftime("%Y.%m")}
        identifier: {date.strftime("%Y%m")}
        parent: papers
        weight: 10
---
            """
            )

    new_post_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_post_path, mode="wt", encoding="utf-8") as wf:
        wf.write(text)

    # add references
    if not without_citations:
        add_references(title, new_post_path)

    # copy hero.jpg
    shutil.copy("src/resources/assets/images/hero.jpg", str(new_post_path.parent / "hero.jpg"))

    print(f"New Post -> {str(new_post_path.absolute())}")


@cli.command()
@click.option("--title", type=str, help="title of the paper", required=True)
def new_graph(title: str):
    """create new post for graphs"""
    assert len(title) > 0

    text = open(Path(__file__).parent / "templates" / "graph_template.txt").read()
    date = datetime.now()
    text = text.format(
        TITLE=title,
        DATE=date.strftime("%Y-%m-%d"),
        IDENTIFIER=f"{datetime.now().strftime('%Y%m%d')}_graph",
        PARENT=f"{date.strftime('%Y%m')}_graphs",
    )
    new_graph_path = Path(
        f"src/content/posts/graphs/{date.strftime('%Y%m')}/{datetime.now().strftime('%Y%m%d%H%M%S')}/index.md"
    )

    if not new_graph_path.parent.parent.exists():
        new_graph_path.parent.parent.mkdir(parents=True)
        with open(new_graph_path.parent.parent / "_index.md", mode="w", encoding="utf-8") as wf:
            wf.write(
                f"""---
title: {date.strftime("%Y.%m")}
menu:
    sidebar:
        name: {date.strftime("%Y.%m")}
        identifier: {date.strftime("%Y%m")}_graphs
        parent: graphs
        weight: 10
---
            """
            )

    new_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_graph_path, mode="wt", encoding="utf-8") as wf:
        wf.write(text)

    print(f"New Graph -> {str(new_graph_path.absolute())}")


@cli.command()
@click.option(
    "--date", type=str, help="date to collect arxiv papers", required=True, default=datetime.now() - timedelta(days=5)
)
def update_arxiv(date):
    def __update_arxiv(target_date):
        post_all: list[Paper] = [
            post for post in get_arxiv_posts(target_date - timedelta(days=2)) if len(post.keywords) > 0
        ]

        if len(post_all) == 0:
            return

        post_dict: dict[str, list[Paper]] = {}
        for post in post_all:
            if post.primary_category not in post_dict:
                post_dict[post.primary_category] = []
            post_dict[post.primary_category].append(post)

        new_path = Path(
            f"src/content/posts/arxiv/{target_date.strftime('%Y%m')}/{target_date.strftime('%Y%m%d%H%M%S')}/index.md"
        )

        if not new_path.parent.parent.exists():
            new_path.parent.parent.mkdir(parents=True)
            with open(new_path.parent.parent / "_index.md", mode="w", encoding="utf-8") as wf:
                wf.write(
                    f"""---
title: {target_date.strftime("%Y.%m")}
menu:
    sidebar:
        name: {target_date.strftime("%Y.%m")}
        identifier: {target_date.strftime("%Y%m")}_arxiv
        parent: arxiv
        weight: 10
---
            """
                )

        new_path.parent.mkdir(parents=True, exist_ok=True)

        # prepare content
        text = f"""---
draft: false
title: "arXiv @ {target_date.strftime("%Y.%m.%d")}"
date: {target_date.strftime("%Y-%m-%d")}
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:{target_date.year}"]
menu:
  sidebar:
    name: "arXiv @ {target_date.strftime("%Y.%m.%d")}"
    identifier: arxiv_{target_date.strftime("%Y%m%d")}
    parent: {target_date.strftime("%Y%m")}_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


"""

        text += "## Primary Categories\n\n"
        text += "\n".join(
            [
                f"- [{categ} ({len(posts)})](#{categ.lower().replace('.', '')}-{len(posts)})"
                for categ, posts in sorted([(k, v) for k, v in post_dict.items()], key=lambda x: x[0])
            ]
        )
        text += "\n\n"

        kw_fd = FreqDist()
        for categ, posts in post_dict.items():
            if len(posts) < 10:
                continue
            for post in posts:
                for kw in post.keywords:
                    kw_fd[(categ, kw.keyword)] += 1
        kw_df = pd.DataFrame(kw_fd.items(), columns=["org", "count"])
        kw_df["category"] = kw_df["org"].apply(lambda x: x[0])
        kw_df["keyword"] = kw_df["org"].apply(lambda x: x[1])
        kw_df = kw_df[["category", "keyword", "count"]].sort_values(by=["category", "count"], ascending=[True, False])
        kw_df = pd.pivot_table(kw_df, index="keyword", columns="category", values="count", fill_value="", aggfunc="sum")

        text += f"""## Keywords

{kw_df.to_markdown()}

<script>
$(function() {{
    $("table").addClass("keyword-table")
    $("table thead").addClass("sticky-top")
}})
</script>
"""
        paper_count = 1
        for primary_category, posts in post_dict.items():
            text += f"\n\n## {primary_category} ({len(posts)})\n\n"
            for post in posts:
                title, content = post.generate_citation_text(paper_count)
                text += f"""\n
### ({paper_count}/{len(post_all)}) {title}

{{{{<citation>}}}}
{content}
{{{{</citation>}}}}
"""
                paper_count += 1

        with open(new_path, mode="wt", encoding="utf-8") as f:
            f.write(text)

        fig = go.Figure()
        labels = list(post_dict.keys())
        values = [len(posts) for posts in post_dict.values()]
        fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4))
        fig.update_traces(hoverinfo="label+percent", textinfo="label+value", textposition="inside", textfont_size=20)
        fig.update_layout(width=800, height=600, margin=dict(t=1, b=1, l=1, r=1))
        fig.write_html(str(new_path.parent / "pie.html"))

        shutil.copy("src/resources/assets/images/arxiv.png", str(new_path.parent / "hero.png"))

    target_date = parse_date(date)
    for d in tqdm(range(5, 0, -1), total=5):
        __update_arxiv(target_date - timedelta(days=d))


if __name__ == "__main__":
    cli()
