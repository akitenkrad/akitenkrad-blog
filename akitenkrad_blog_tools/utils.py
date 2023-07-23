import io
import json
import os
import re
import socket
import string
import time
import urllib.parse
import urllib.request
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from io import TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from urllib.error import HTTPError, URLError

import numpy as np
import requests
from attrdict import AttrDict
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm


@dataclass
class Author(object):
    ss_author_id: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {"author_id": self.ss_author_id, "name": self.name}

    @staticmethod
    def from_dict(dict_data: dict[str, Any]):
        return Author(**dict_data)


@dataclass
class RefPaper(object):
    ss_paper_id: str
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.ss_paper_id, "title": self.title}

    @staticmethod
    def from_dict(dict_data: dict[str, Any]):
        return RefPaper(**dict_data)


@dataclass
class Paper(object):
    title: str
    abstract: str = ""
    venue: str = ""
    year: int = 2999
    article: str = ""
    paper_id: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""
    primary_category: str = ""
    authors: list[Author] = field(default_factory=list)
    citations: list[RefPaper] = field(default_factory=list)
    references: list[RefPaper] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    reference_count: int = 0
    citation_count: int = 0
    influential_citation_count: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    introduction_summary: str = ""
    at: float = field(default_factory=lambda: datetime.now(timezone(timedelta(hours=9), "JST")).timestamp())

    @property
    def has_abstract(self) -> bool:
        return len(self.abstract.strip()) > 0

    def __str__(self):
        return f"<Paper title:{self.title[:15]}...>"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "article": self.article,
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "title": self.title,
            "abstract": self.abstract,
            "venue": self.venue,
            "year": self.year,
            "primary_category": self.primary_category,
            "categories": self.categories,
            "keywords": self.keywords,
            "authors": [author.to_dict() for author in self.authors],
            "citations": [ref_paper.to_dict() for ref_paper in self.citations],
            "references": [ref_paper.to_dict() for ref_paper in self.references],
            "reference_count": self.reference_count,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "fields_of_study": self.fields_of_study,
            "introduction_summary:": self.introduction_summary,
            "at": self.at,
        }

    @staticmethod
    def from_dict(dict_data: dict[str, Any]):
        if "authors" in dict_data:
            dict_data["authors"] = [Author.from_dict(item) for item in dict_data["authors"]]
        if "citations" in dict_data:
            dict_data["citations"] = [RefPaper.from_dict(item) for item in dict_data["citations"]]
        if "references" in dict_data:
            dict_data["references"] = [RefPaper.from_dict(item) for item in dict_data["references"]]

        return Paper(**dict_data)

    def to_short_text(self):
        author_text = ""
        if len(self.authors) == 1:
            author_text = self.authors[0].name.replace('"', "'")
        elif len(self.authors) > 1:
            author_text = self.authors[0].name.replace('"', "'") + " et al"
        return f"{author_text} {self.year}".replace(" ", "_")

    def generate_citation_text(self):
        author_text = ""
        if len(self.authors) == 1:
            author_text = self.authors[0].name.replace('"', "'")
        elif len(self.authors) > 1:
            author_text = self.authors[0].name.replace('"', "'") + " et al."
        title_text = self.title.replace('"', "'")
        title = f"{title_text} ({author_text}, {self.year})"

        content = f"""
{", ".join([author.name for author in self.authors]) + f". ({self.year})  "}
**{title_text}**{"  "}

---
Primary Category: {self.primary_category}{"  "}
Categories: {", ".join(sorted(self.categories))}{"  "}
Keywords: {", ".join(sorted(self.keywords))}{"  "}
[Paper Link]({self.url}){"  "}

---


{"**ABSTRACT**  " if self.has_abstract else ""}
{self.abstract.replace(os.linesep, " ").strip() if self.has_abstract else ""}
"""

        return title, content

    def print_citation(self, f: TextIOWrapper):
        author_text = ""
        if len(self.authors) == 1:
            author_text = self.authors[0].name.replace('"', "'")
        elif len(self.authors) > 1:
            author_text = self.authors[0].name.replace('"', "'") + " et al."
        title = self.title.replace('"', "'")

        citation = [
            f'{{{{< ci-details summary="{title} ({author_text}, {self.year})">}}}}',
            "",
            ", ".join([author.name for author in self.authors]) + f". ({self.year})  ",
            f"**{title}**  ",
            self.venue + "  ",
            f"[Paper Link]({self.url})" + "  ",
            f"Influential Citation Count ({self.influential_citation_count}), SS-ID ({self.paper_id})  ",
            "",
            "**ABSTRACT**  " if self.has_abstract else "",
            "" + self.abstract.replace(os.linesep, " ").strip() if self.has_abstract else "",
            "",
            "{{< /ci-details >}}",
            "",
            os.linesep,
        ]
        f.write(re.sub(r"\n\n+", "\n", os.linesep.join(citation)))


def extract_keywords(keywords: list[str], text) -> list[str]:
    matched_keywords = []
    for kw in keywords:
        if kw in text:
            matched_keywords.append(kw)
    return matched_keywords


def get_pdf_text(pdf_url: str) -> str:
    content = io.BytesIO(requests.get(pdf_url).content)
    reader = PdfReader(content)
    texts = ""
    header = True
    for p in range(len(reader.pages)):
        text = reader.pages[p].extract_text()
        for line in text.split("\n"):
            if "introduction" in line.lower():
                header = False
            if not header:
                texts += line + os.linesep
    return texts


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__iter__"):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.strftime("%Y%m%d %H:%M:%S.%f")
        elif isinstance(obj, date):
            return datetime(obj.year, obj.month, obj.day, 0, 0, 0).strftime("%Y%m%d %H:%M:%S.%f")
        else:
            return super().default(obj)


def generate_text_image(text: str, fontsize: int, out_file: str):
    image = Image.new("RGB", (1600, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path(__file__).parent / "fonts/CodeM-Regular.ttf"), fontsize)

    x = 400
    y = 50

    draw.text((x, y), text, fill=(80, 80, 80), font=font)

    image.save(out_file)
