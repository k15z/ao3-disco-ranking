from typing import Dict, List

from bs4 import BeautifulSoup
from unidecode import unidecode

from .utils import get


class WorkDoesNotExistError(Exception):
    """Typically means the user submitted garbage."""


class WorkIsPrivateError(Exception):
    """Typically happens if the author made the work private."""


class Work:
    work_id: str
    title: str
    authors: List[str]
    statistics: Dict[str, str]
    tags: Dict[str, List[str]]
    summary: str
    content: str

    @classmethod
    def load(cls, work_id: str) -> "Work":
        url = f"http://archiveofourown.org/works/{work_id}?view_adult=true"
        work = Work()
        try:
            soup = BeautifulSoup(get(url), "html.parser")
        except KeyError:
            raise WorkDoesNotExistError(url)

        work.work_id = work_id
        work.title = cls._get_title(soup)
        work.authors = cls._get_authors(soup)
        work.statistics = cls._get_statistics(soup)
        work.tags = cls._get_tags(soup)
        work.summary = cls._get_summary(soup)
        work.content = cls._get_content(soup)

        return work

    @classmethod
    def _get_title(cls, soup: BeautifulSoup) -> str:
        title = soup.find("h2", class_="title heading")
        if not title:
            raise WorkIsPrivateError("Unable to find title. The work may be private.")
        return unidecode(title.text).strip()

    @classmethod
    def _get_summary(cls, soup: BeautifulSoup) -> str:
        content = soup.find("div", class_="summary module")
        if not content:
            return ""
        summary = content.find("blockquote")
        return unidecode(summary.get_text(" ", strip=True))

    @classmethod
    def _get_content(cls, soup: BeautifulSoup) -> str:
        content = soup.find("div", id="chapters")
        chapters = content.select("p")
        return "\n\n".join(
            [unidecode(chapter.get_text(" ", strip=True)) for chapter in chapters]
        ).strip()

    @classmethod
    def _get_authors(cls, soup: BeautifulSoup) -> List[str]:
        authors = []

        for tag in soup.find("h3", class_="byline heading").contents:
            if tag.name == "a":
                user_id = tag.get("href").split("/")[2]
                authors.append(user_id)

        return authors

    @classmethod
    def _get_statistics(cls, soup: BeautifulSoup) -> Dict[str, str]:
        categories = [
            "language",
            "published",
            "status",
            "words",
            "chapters",
            "comments",
            "kudos",
            "bookmarks",
            "hits",
        ]
        meta = soup.find("dl", class_="work meta group")

        statistics = {}
        for category in categories:
            stat = meta.find("dd", class_=category)
            if stat:
                statistics[category] = unidecode(stat.text).strip()
        return statistics

    @classmethod
    def _get_tags(cls, soup: BeautifulSoup) -> Dict[str, List[str]]:
        meta = soup.find("dl", class_="work meta group")

        def get_tag_info(category, meta):
            try:
                tag_list = meta.find("dd", class_=str(category) + " tags").find_all(class_="tag")
            except AttributeError:
                return []
            return [unidecode(result.text) for result in tag_list]

        tag_types = [
            "rating",
            "category",
            "fandom",
            "relationship",
            "character",
            "freeform",
        ]
        return {tag_type: get_tag_info(tag_type, meta) for tag_type in tag_types}


if __name__ == "__main__":
    work = Work.load("29210880")
    print(work.summary)
