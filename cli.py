import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import click
from tqdm import tqdm

from utils import Paper, SemanticScholar


@click.group()
def cli():
    pass


@cli.command()
def new_post():
    text = f"""---
draft: false
title: "TITLE"
date: {datetime.now().strftime('%Y-%m-%d')}
author: "akitenkrad"
description: ""
tags: ["Round-1", "20XX"]
menu:
  sidebar:
    name: {datetime.now().strftime('%Y.%m.%d')}
    identifier: {datetime.now().strftime('%Y%m%d')}
    parent: papers
    weight: 10
math: true
---

- [x] Round-1: Overview
- [ ] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

## Abstract

## What's New

## Dataset

## Model Description

### Training Settings

## Results

## References


"""

    new_post_path = Path(f"src/content/posts/papers/{datetime.now().strftime('%Y%m%d%H%M%S')}/index.md")
    new_post_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_post_path, mode="wt", encoding="utf-8") as wf:
        wf.write(text)

    shutil.copy("src/resources/assets/images/hero.jpg", str(new_post_path.parent / "hero.jpg"))

    print(f"New Post -> {str(new_post_path.absolute())}")


@cli.command()
@click.option("--path", type=str, help="path for new note")
def new_note(path: str):
    title = Path(path).name
    parents = str(Path(path.replace(title, "")))
    text = f"""---
draft: false
title: "TITLE"
date: {datetime.now().strftime('%Y-%m-%d')}
author: "akitenkrad"
description: ""
tags: ["Round-1"]
menu:
  sidebar:
    name: {title}
    identifier: {datetime.now().strftime('%Y%m%d')}
    parent: notes{parents}
    weight: 10
math: true
---

- [x] Round-1: Overview
- [ ] Round-2: Model Implementation Details
- [ ] Round-3: Experiments

## Citation

## Abstract

## What's New

## Dataset

## Model Description

### Training Settings

## Results

## References


"""

    new_note_path = Path(f"src/content/notes/{path}/index.md")
    if new_note_path.exists():
        raise RuntimeError(f"file already exists: {new_note_path}")

    new_note_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_note_path, mode="wt", encoding="utf-8") as wf:
        wf.write(text)

    shutil.copy("src/resources/assets/images/hero.jpg", str(new_note_path.parent / "hero.jpg"))
    print(f"New Note -> {str(new_note_path.absolute())}")


@cli.command()
@click.option("--title", type=str, help="title of the paper")
@click.option("--to", type=click.Path(exists=True), help="output file path")
def get_references(title: str, to: str):
    ss = SemanticScholar()
    paper_id = ss.get_paper_id(title=title)
    paper = ss.get_paper_detail(paper_id=paper_id)

    # Load Cache
    cache_path = Path("__cache__/papers.json")
    if cache_path.exists():
        cache = json.load(open(cache_path))
    else:
        cache = {}
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Read References
    if paper is not None:
        cache[paper.paper_id] = paper.to_dict()

        with open(to, mode="at", encoding="utf-8") as wf:
            for ref in tqdm(paper.references, desc="Reading references..."):
                ref_paper = ss.get_paper_detail(ref.paper_id)
                if ref_paper is not None:
                    ref_paper.print_citation(wf)

                    if ref_paper.paper_id not in cache:
                        cache[ref_paper.paper_id] = ref_paper.to_dict()
    else:
        print("No such a paper:", title)
    json.dump(cache, open(cache_path, mode="wt", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    cli()
