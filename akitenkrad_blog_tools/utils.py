import io
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import requests
from keywords.keywords import Keyword
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


@dataclass
class Author(object):
    """
    Represents an author of a paper.

    Attributes:
        ss_author_id (str): The ID of the author.
        name (str): The name of the author.
    """

    ss_author_id: str
    name: str

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the Author object to a dictionary.

        Returns:
            dict[str, Any]: The dictionary representation of the Author object.
        """
        return {"author_id": self.ss_author_id, "name": self.name}

    @staticmethod
    def from_dict(dict_data: dict[str, Any]):
        """
        Creates an Author object from a dictionary.

        Args:
            dict_data (dict[str, Any]): The dictionary containing the Author data.

        Returns:
            Author: The Author object created from the dictionary.
        """
        return Author(**dict_data)


@dataclass
class RefPaper(object):
    """
    Represents a reference paper.

    Attributes:
        ss_paper_id (str): The ID of the reference paper.
        title (str): The title of the reference paper.
    """

    ss_paper_id: str
    title: str

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the RefPaper object to a dictionary.

        Returns:
            dict[str, Any]: The dictionary representation of the RefPaper object.
        """
        return {"paper_id": self.ss_paper_id, "title": self.title}

    @staticmethod
    def from_dict(dict_data: dict[str, Any]):
        """
        Creates a RefPaper object from a dictionary.

        Args:
            dict_data (dict[str, Any]): The dictionary containing the RefPaper data.

        Returns:
            RefPaper: The RefPaper object created from the dictionary.
        """
        return RefPaper(**dict_data)


@dataclass
class Paper(object):
    """
    Represents a paper.

    Attributes:
        title (str): The title of the paper.
        abstract (str): The abstract of the paper.
        venue (str): The venue of the paper.
        year (int): The year of the paper.
        article (str): The article of the paper.
        paper_id (str): The ID of the paper.
        arxiv_id (str): The arXiv ID of the paper.
        url (str): The URL of the paper.
        pdf_url (str): The PDF URL of the paper.
        primary_category (str): The primary category of the paper.
        authors (list[Author]): The authors of the paper.
        citations (list[RefPaper]): The citations of the paper.
        references (list[RefPaper]): The references of the paper.
        categories (list[str]): The categories of the paper.
        keywords (list[Keyword]): The keywords of the paper.
        reference_count (int): The reference count of the paper.
        citation_count (int): The citation count of the paper.
        influential_citation_count (int): The influential citation count of the paper.
        fields_of_study (list[str]): The fields of study of the paper.
        introduction_summary (str): The introduction summary of the paper.
        at (float): The timestamp of when the paper was created.
    """

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
    keywords: list[Keyword] = field(default_factory=list)
    reference_count: int = 0
    citation_count: int = 0
    influential_citation_count: int = 0
    fields_of_study: list[str] = field(default_factory=list)
    introduction_summary: str = ""
    at: float = field(default_factory=lambda: datetime.now(timezone(timedelta(hours=9), "JST")).timestamp())

    @property
    def has_abstract(self) -> bool:
        """
        Checks if the paper has an abstract.

        Returns:
            bool: True if the paper has an abstract, False otherwise.
        """
        return len(self.abstract.strip()) > 0

    @property
    def keyword_score(self) -> int:
        return sum([kw.score for kw in self.keywords])

    def __str__(self):
        return f"<Paper title:{self.title[:15]}...>"

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        """
        Converts the Paper object to a dictionary.

        Returns:
            dict: The dictionary representation of the Paper object.
        """
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
        """
        Creates a Paper object from a dictionary.

        Args:
            dict_data (dict[str, Any]): The dictionary containing the Paper data.

        Returns:
            Paper: The Paper object created from the dictionary.
        """
        if "authors" in dict_data:
            dict_data["authors"] = [Author.from_dict(item) for item in dict_data["authors"]]
        if "citations" in dict_data:
            dict_data["citations"] = [RefPaper.from_dict(item) for item in dict_data["citations"]]
        if "references" in dict_data:
            dict_data["references"] = [RefPaper.from_dict(item) for item in dict_data["references"]]

        return Paper(**dict_data)

    def to_short_text(self):
        """
        Generates a short text representation of the paper.

        Returns:
            str: The short text representation of the paper.
        """
        author_text = ""
        if len(self.authors) == 1:
            author_text = self.authors[0].name.replace('"', "'")
        elif len(self.authors) > 1:
            author_text = self.authors[0].name.replace('"', "'") + " et al"
        return f"{author_text} {self.year}".replace(" ", "_")

    def markup_with_keywords(self, text: str):
        """
        Marks up the text with the keywords of the paper.

        Args:
            text (str): The text to be marked up.

        Returns:
            str: The marked up text.
        """
        tokens = text.split()
        markup_flags = [False] * len(tokens)
        for kw in self.keywords:
            kw_tokens = [k.lower().strip() for k in kw.word.split()]
            for i in range(len(tokens) - len(kw_tokens) + 1):
                temp_tokens = [t.lower().strip() for t in tokens[i : i + len(kw_tokens)]]
                ptn = Keyword.get_ptn(kw_tokens[0])
                if ptn.match(temp_tokens[0]):
                    for j in range(i, i + len(kw_tokens)):
                        markup_flags[j] = True

        markuped_tokens = []
        for token, flag in zip(tokens, markup_flags):
            if flag:
                markuped_tokens.append(f"<b>{token}</b>")
            else:
                markuped_tokens.append(token)

        return " ".join(markuped_tokens)

    def generate_citation_text(self, index: int):
        """
        Generates the citation text for the paper.

        Args:
            index (int): The index of the citation.
        """
        author_text = ""
        if len(self.authors) == 1:
            author_text = self.authors[0].name.replace('"', "'")
        elif len(self.authors) > 1:
            author_text = self.authors[0].name.replace('"', "'") + " et al."
        title_text = self.title.replace('"', "'")
        title = f"{title_text} ({author_text}, {self.year})"

        pdf_url = self.pdf_url if self.pdf_url.endswith(".pdf") else self.pdf_url + ".pdf"
        if "http://" in pdf_url:
            pdf_url = pdf_url.replace("http://", "https://")

        content = f"""
{", ".join([author.name for author in self.authors]) + f". ({self.year})  "}
**{title_text}**
<br/>
<button class="copy-to-clipboard" title="{title_text}" index={index}>
  <span class="copy-to-clipboard-item">Copy Title<span>
</button>
<div class="toast toast-copied toast-index-{index} align-items-center text-bg-secondary border-0 position-absolute top-0 end-0" role="alert" aria-live="assertive" aria-atomic="true">
  <div class="d-flex">
    <div class="toast-body">
      Copied!
    </div>
  </div>
</div>

---
Primary Category: {self.primary_category}{"  "}
Categories: {", ".join(sorted(self.categories))}{"  "}
Keyword Score: {self.keyword_score}{"  "}
Keywords: {", ".join([kw.keyword for kw in self.keywords])}{"  "}
<a type="button" class="btn btn-outline-primary" href="{self.url}" target="_blank" >Paper Link</a>
<button type="button" class="btn btn-outline-primary download-pdf" url="{pdf_url}" filename="{Path(pdf_url).name}">Download PDF</button>

---


{"**ABSTRACT**  " if self.has_abstract else ""}
{self.markup_with_keywords(self.abstract.replace(os.linesep, " ").strip()) if self.has_abstract else ""}
"""

        return title, content

    def print_citation(self, f: TextIOWrapper):
        """
        Prints the citation of the paper to a file.

        Args:
            f (TextIOWrapper): The file to write the citation to.
        """
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


def get_pdf_text(pdf_url: str) -> str:
    """
    Retrieves the text content of a PDF file from a given URL.

    Args:
        pdf_url (str): The URL of the PDF file.

    Returns:
        str: The text content of the PDF file.
    """
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
    """
    Custom JSON encoder that handles additional data types.

    Overrides the default method of the JSONEncoder class to handle additional data types
    such as numpy types, datetime, and date.

    Usage:
        json.dumps(data, cls=JsonEncoder)
    """

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
    """
    Generates an image with the given text.

    Args:
        text (str): The text to be displayed in the image.
        fontsize (int): The font size of the text.
        out_file (str): The output file path for the generated image.
    """
    image = Image.new("RGB", (1600, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path(__file__).parent / "fonts/CodeM-Regular.ttf"), fontsize)

    x = 400
    y = 50

    draw.text((x, y), text, fill=(80, 80, 80), font=font)

    image.save(out_file)
