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
from enum import Enum
from io import TextIOWrapper
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO
from urllib.error import HTTPError, URLError

from attrdict import AttrDict
from sumeval.metrics.rouge import RougeCalculator
from tqdm import tqdm

from akitenkrad_blog_tools.utils import Paper


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

        print(msg)

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
                response = urllib.request.urlopen(
                    self.__api.search_by_title.format(QUERY=urllib.parse.urlencode(params)), timeout=5.0
                )
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
                response = urllib.request.urlopen(
                    self.__api.search_by_id.format(PAPER_ID=paper_id, PARAMS=params), timeout=5.0
                )
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


def add_references(title: str, out_file: PathLike):
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

        with open(out_file, mode="at", encoding="utf-8") as wf:
            for ref in tqdm(paper.references, desc="Reading references..."):
                ref_paper = ss.get_paper_detail(ref.paper_id)
                if ref_paper is not None:
                    ref_paper.print_citation(wf)

                    if ref_paper.paper_id not in cache:
                        cache[ref_paper.paper_id] = ref_paper.to_dict()
    else:
        print("No such a paper:", title)
    json.dump(cache, open(cache_path, mode="wt", encoding="utf-8"), ensure_ascii=False, indent=2)
