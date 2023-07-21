import json
from datetime import datetime, timedelta
from pathlib import Path

import arxiv
import pytz

from cmds.utils import Paper, extract_keywords


def get_arxiv_posts(day: datetime) -> list[Paper]:
    start = day.strftime("%Y%m%d") + "000000"
    end = day.strftime("%Y%m%d") + "235959"
    cv_papers = arxiv.Search(
        query="cat:cs.* AND submittedDate:[{} TO {}]".format(start, end), sort_by=arxiv.SortCriterion.SubmittedDate
    )

    with open(Path(__file__).parent / "keywords.json", mode="rt", encoding="utf-8") as f:
        keywords = json.load(f)

    posts = []
    for idx, paper in enumerate(cv_papers.results()):
        summary = paper.summary.replace("\n", " ").replace(". ", ".\n")
        categories = [paper.primary_category]
        categories += [f"{c.replace('.', '-')}" for c in paper.categories]

        title_keywords = extract_keywords(keywords, paper.title)
        summary_keywords = extract_keywords(keywords, summary)
        extracted_keywords = sorted(list(set(title_keywords + summary_keywords)))

        posts.append(
            Paper(
                **{
                    "article": "arXiv",
                    "title": paper.title,
                    "abstract": summary,
                    "authors": paper.authors,
                    "primary_category": categories[0] if len(categories) > 0 else "",
                    "categories": categories,
                    "keywords": extracted_keywords,
                    "url": paper.entry_id,
                    "arxiv_id": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                    "year": day.year,
                }
            )
        )

    return posts
