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
@click.option("--title", type=str, help="title of the paper")
def new_post(title: str):
    text = open(Path(__file__).parent / "templates" / "post_template.txt").read()
    text = text.format(
        TITLE=title, DATE=datetime.now().strftime("%Y-%m-%d"), NAME=datetime.now().strftime("%Y.%m.%d"), IDENTIFIER=datetime.now().strftime("%Y%m%d")
    )
    new_post_path = Path(f"src/content/posts/papers/{datetime.now().strftime('%Y%m%d%H%M%S')}/index.md")
    new_post_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_post_path, mode="wt", encoding="utf-8") as wf:
        wf.write(text)

    # add references
    add_references(title, new_post_path)

    # copy hero.jpg
    shutil.copy("src/resources/assets/images/hero.jpg", str(new_post_path.parent / "hero.jpg"))

    print(f"New Post -> {str(new_post_path.absolute())}")


@cli.command()
@click.option("--title", type=str, help="title of the paper")
@click.option("--out-file", type=click.Path(exists=True), help="output file path")
def get_references(title: str, out_file: str):
    add_references(title, out_file)


if __name__ == "__main__":
    cli()
