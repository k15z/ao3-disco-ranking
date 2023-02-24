import json
import logging
import os
from typing import List

import pymysql

from ao3_disco_ranking.types import WorkID

from .work import Work

logger = logging.getLogger()


def connect():
    return pymysql.connect(
        host=os.environ["AO3_DISCO_MYSQL_HOST"],
        user=os.environ["AO3_DISCO_MYSQL_USER"],
        passwd=os.environ["AO3_DISCO_MYSQL_PASS"],
        db="prod",
        port=25060,
    )


def get_work_jsons(work_ids: List[WorkID], include_content: bool = False):
    """Batch request work_jsons.
    This submits a single SQL query for fetching all the provided work IDs. For those
    which are not in the database, it calls `get_work_json` to fetch it instead.
    """
    if len(work_ids) == 0:
        return []
    work_id_to_json = {}
    with connect() as conn:
        with conn.cursor() as cursor:
            placeholder = ",".join(["%s"] * len(work_ids))
            query = f"""
                SELECT work_id, json, content 
                FROM work 
                WHERE work_id IN ({placeholder})
            """
            cursor.execute(query, work_ids)
        for work_id, data, content in cursor.fetchall():
            result = json.loads(data)
            if include_content:
                result["content"] = content
            work_id_to_json[work_id] = result

    for work_id in work_ids:
        if work_id in work_id_to_json:
            yield work_id_to_json[work_id]
        else:
            try:
                yield get_work_json(work_id, include_content=include_content)
            except Exception as e:
                yield {"error": type(e).__name__}


def get_work_json(work_id: str, include_content: bool = False):
    logging.info(f"Scraping work {work_id}...")
    work = Work.load(work_id).__dict__
    content = work["content"]
    del work["content"]
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO work (work_id, updated_at, json, content) VALUES (%s, NOW(), %s, %s)",
                (work_id, json.dumps(work), content),
            )

            tags_to_insert = []
            for tag_type in work["tags"]:
                for tag_value in work["tags"][tag_type]:
                    tags_to_insert.append((work_id, tag_type, tag_value))
            cursor.executemany(
                "INSERT INTO work_tag (work_id, tag_type, tag_value) VALUES (%s, %s, %s)",
                tags_to_insert,
            )
            conn.commit()
    if include_content:
        work["content"] = content
    return work
