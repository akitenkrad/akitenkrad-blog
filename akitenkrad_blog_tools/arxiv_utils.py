import json
from datetime import datetime
from pathlib import Path

import arxiv
from keywords.keywords import Keyword, extract_keywords

from akitenkrad_blog_tools.utils import Paper


def get_arxiv_posts(day: datetime) -> list[Paper]:
    keywords = Keyword.load_keywords()
    start = day.strftime("%Y%m%d") + "000000"
    end = day.strftime("%Y%m%d") + "235959"
    cv_papers = arxiv.Search(
        query="cat:cs.* AND submittedDate:[{} TO {}]".format(start, end), sort_by=arxiv.SortCriterion.SubmittedDate
    )

    posts = []
    for paper in cv_papers.results():
        summary = paper.summary.replace("\n", " ").replace(". ", ".\n")
        categories = [paper.primary_category]
        categories += [f"{c.replace('.', '-')}" for c in paper.categories]

        title_keywords = extract_keywords(paper.title, keywords, remove_stopwords=True)
        summary_keywords = extract_keywords(summary, keywords, remove_stopwords=True)
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
        posts = sorted(posts, key=lambda p: p.keyword_score, reverse=True)

    return posts
