import json
import os
import re
import socket
import string
import time
import urllib.parse
import urllib.request
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

from attrdict import AttrDict
from sumeval.metrics.rouge import RougeCalculator

Author = namedtuple("Author", ("author_id", "name"))
RefPaper = namedtuple("RefPaper", ("paper_id", "title"))


class Paper(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, f"__{key}", value)

        self.__at = datetime.now(timezone(timedelta(hours=9), "JST")).timestamp()

    def __get(self, key: str, default: Any) -> Any:
        value = getattr(self, key) if hasattr(self, key) else default
        if value is None:
            return default
        else:
            return value

    def __filter_none(self, value: Any, default: Any = ""):
        if value is None:
            return default
        else:
            return value

    def add_fields(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, f"__{key}", value)

    @property
    def paper_id(self) -> str:
        """paper id from SemanticScholar"""
        return self.__get("__paperId", default="")

    @property
    def url(self) -> str:
        """url from SemanticScholar"""
        return self.__get("__url", default="")

    @property
    def title(self) -> str:
        """title from SemanticScholar"""
        return self.__get("__title", default="")

    @property
    def abstract(self) -> str:
        """abstract from SemanticScholar"""
        return self.__get("__abstract", default="")

    @property
    def venue(self) -> str:
        """venue from SemanticScholar"""
        return self.__get("__venue", default="")

    @property
    def year(self) -> int:
        """year from SemanticScholar"""
        return int(self.__get("__year", default=-1))

    @property
    def reference_count(self) -> int:
        """reference count from SemanticScholar"""
        return int(self.__get("__referenceCount", default=0))

    @property
    def citation_count(self) -> int:
        """citation count from SemanticScholar"""
        return int(self.__get("__citationCount", default=0))

    @property
    def influential_citation_count(self) -> int:
        """influential citation count from SemanticScholar"""
        return int(self.__get("__influentialCitationCount", default=0))

    @property
    def is_open_access(self) -> bool:
        """is open access from SemanticScholar"""
        return self.__get("__isOpenAccess", default=False)

    @property
    def fields_of_study(self) -> List[str]:
        """fields of study from SemanticScholar"""
        return self.__get("__fieldsOfStudy", default=[])

    @property
    def authors(self) -> List[Author]:
        """authors from SemanticScholar"""
        author_list = self.__get("__authors", default=[])
        return [Author(self.__filter_none(a["authorId"]), self.__filter_none(a["name"])) for a in author_list]

    @property
    def citations(self) -> List[RefPaper]:
        """citations from SemanticScholar"""
        citation_list = self.__get("__citations", default=[])
        return [RefPaper(p["paperId"], p["title"]) for p in citation_list]

    @property
    def references(self) -> List[RefPaper]:
        """references from SemanticScholar"""
        reference_list = self.__get("__references", default=[])
        return [RefPaper(p["paperId"], p["title"]) for p in reference_list]

    @property
    def at(self) -> datetime:
        return datetime.fromtimestamp(self.__at)

    def __str__(self):
        return f'<Paper id:{self.paper_id} title:{self.title[:15]}... @{self.at.strftime("%Y.%m.%d-%H:%M:%S")}>'

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        return {
            "abstract": self.abstract,
            "authors": [{"author_id": a.author_id, "name": a.name} for a in self.authors],
            "citation_count": self.citation_count,
            "citations": [{"paper_id": r.paper_id, "title": r.title} for r in self.citations if r.paper_id is not None],
            "fields_of_study": self.fields_of_study,
            "influential_citation_count": self.influential_citation_count,
            "is_open_access": self.is_open_access,
            "paper_id": self.paper_id,
            "reference_count": self.reference_count,
            "references": [{"paper_id": r.paper_id, "title": r.title} for r in self.references if r.paper_id is not None],
            "title": self.title,
            "url": self.url,
            "venue": self.venue,
            "year": self.year,
            "at": self.__get("__at", default=0),
        }

    @staticmethod
    def from_dict(paper_data: dict):
        kwargs = {
            "paperId": paper_data["paper_id"],
            "url": paper_data["url"],
            "title": paper_data["title"],
            "abstract": paper_data["abstract"],
            "venue": paper_data["venue"],
            "year": paper_data["year"],
            "referenceCount": paper_data["reference_count"],
            "citationCount": paper_data["citation_count"],
            "influentialCitationCount": paper_data["influential_citation_count"],
            "isOpenAccess": paper_data["is_open_access"],
            "fieldsOfStudy": paper_data["fields_of_study"],
            "authors": [{"authorId": a["author_id"], "name": a["name"]} for a in paper_data["authors"]],
            "citations": [{"paperId": r["paper_id"], "title": r["title"]} for r in paper_data["citations"]],
            "references": [{"paperId": r["paper_id"], "title": r["title"]} for r in paper_data["references"]],
            "at": paper_data["at"],
        }
        return Paper(**kwargs)

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
            "**ABSTRACT**  ",
            "" + self.abstract.replace(os.linesep, " "),
            "",
            "{{< /ci-details >}}",
            os.linesep,
        ]
        f.write(os.linesep.join(citation))


class SemanticScholar(object):
    API: Dict[str, str] = {
        "search_by_title": "https://api.semanticscholar.org/graph/v1/paper/search?{QUERY}",
        "search_by_id": "https://api.semanticscholar.org/graph/v1/paper/{PAPER_ID}?{PARAMS}",
    }
    CACHE_PATH: Path = Path("__cache__/papers.pickle")

    def __init__(self, threshold: float = 0.95):
        self.__api = AttrDict(self.API)
        self.__rouge = RougeCalculator(stopwords=True, stemming=False, word_limit=-1, length_limit=-1, lang="en")
        self.__threshold = threshold

    @property
    def threshold(self) -> float:
        return self.__threshold

    def __retry_and_wait(self, msg: str, ex: Exception, retry: int) -> int:
        retry += 1
        if 5 < retry:
            pass
        if retry == 1:
            msg = "\n" + msg

        if isinstance(ex, socket.timeout) and ex.errno == -3:
            time.sleep(300.0)
        else:
            time.sleep(5.0)
        return retry

    def get_paper_id(self, title: str) -> str:

        # remove punctuation
        title = title
        for punc in string.punctuation:
            title = title.replace(punc, " ")
        title = re.sub(r"\s\s+", " ", title, count=1000)

        retry = 0
        while retry < 5:
            try:
                params = {
                    "query": title,
                    "fields": "title",
                    "offset": 0,
                    "limit": 100,
                }
                response = urllib.request.urlopen(self.__api.search_by_title.format(QUERY=urllib.parse.urlencode(params)), timeout=5.0)
                content = json.loads(response.read().decode("utf-8"))
                time.sleep(3.5)
                break

            except HTTPError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except URLError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except socket.timeout as ex:
                retry = self.__retry_and_wait(f"API Timeout -> Retry: {retry}", ex, retry)
            except Exception as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)

            if 5 <= retry:
                return ""

        for item in content["data"]:
            # remove punctuation
            ref_str = item["title"].lower()
            for punc in string.punctuation:
                ref_str = ref_str.replace(punc, " ")
            ref_str = re.sub(r"\s\s+", " ", ref_str, count=1000)

            score = self.__rouge.rouge_l(summary=title.lower(), references=ref_str)
            if score > self.threshold:
                return item["paperId"].strip()
        return ""

    def get_paper_detail(self, paper_id: str) -> Optional[Paper]:

        retry = 0
        while retry < 5:
            try:
                fields = [
                    "paperId",
                    "url",
                    "title",
                    "abstract",
                    "venue",
                    "year",
                    "referenceCount",
                    "citationCount",
                    "influentialCitationCount",
                    "isOpenAccess",
                    "fieldsOfStudy",
                    "authors",
                    "citations",
                    "references",
                    "embedding",
                ]
                params = f'fields={",".join(fields)}'
                response = urllib.request.urlopen(self.__api.search_by_id.format(PAPER_ID=paper_id, PARAMS=params), timeout=5.0)
                time.sleep(3.5)
                break

            except HTTPError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except URLError as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)
            except socket.timeout as ex:
                retry = self.__retry_and_wait(f"API Timeout -> Retry: {retry}", ex, retry)
            except Exception as ex:
                retry = self.__retry_and_wait(f"{str(ex)} -> Retry: {retry}", ex, retry)

            if 5 <= retry:
                return None

        content = json.loads(response.read().decode("utf-8"))
        return Paper(**content)
