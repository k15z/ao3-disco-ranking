import logging

logging.basicConfig(level=logging.DEBUG)


from bottle import post, request, route, run, static_file

from ao3_disco_ranking.ao3 import get_work_jsons
from ao3_disco_ranking.query import QueryHandler

qh = QueryHandler.load("handler.pkl")


@route("/app/<work_id>")
def app(work_id):
    return static_file("app.html", root="ao3_disco_ranking/scripts/")


@post("/tag")
def tag():
    query = request.json["q"]  # prefix of the tags
    required_tags = [tuple(x) for x in request.json.get("required_tags", [])]
    excluded_tags = [tuple(x) for x in request.json.get("excluded_tags", [])]
    one_or_more_tags = [tuple(x) for x in request.json.get("one_or_more_tags", [])]
    return qh.tags_filter.suggest_tags(query, required_tags, excluded_tags, one_or_more_tags)


@post("/work")
def work():
    work_ids = request.json["work_ids"]
    return {work_id: obj for work_id, obj in zip(work_ids, get_work_jsons(work_ids))}


@post("/query")
def query():
    work_ids = request.json["work_ids"]
    blocklist = request.json.get("blocklist", [])
    required_tags = [tuple(x) for x in request.json.get("required_tags", [])]
    excluded_tags = [tuple(x) for x in request.json.get("excluded_tags", [])]

    work_score = qh.multi_query(
        work_ids=work_ids,
        blocklist=blocklist,
        required_tags=required_tags,
        excluded_tags=excluded_tags,
        num_results=10,
    )
    return {"results": [{"id": work_id, "score": score} for work_id, score in work_score]}


run(host="localhost", port=8080)
