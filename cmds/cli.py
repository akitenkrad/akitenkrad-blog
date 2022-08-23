import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import click
from tqdm import tqdm

from cmds.utils import Paper, SemanticScholar, add_references


@click.group()
def cli():
    pass


@cli.command()
@click.option("--title", type=str, help="title of the paper", required=True)
def new_paper(title: str):
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


if __name__ == "__main__":
    cli()
