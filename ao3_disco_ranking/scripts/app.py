from bottle import post, request, route, run, static_file

from ao3_disco_ranking.ao3 import get_work_jsons
from ao3_disco_ranking.query import QueryHandler

qh = QueryHandler.load("handler.pkl")


@route("/app/<work_id>")
def app(work_id):
    return static_file("app.html", root="ao3_disco_ranking/scripts/")


@post("/work")
def work():
    work_ids = request.json["work_ids"]
    return {work_id: obj for work_id, obj in zip(work_ids, get_work_jsons(work_ids))}


@post("/query")
def query():
    work_ids = request.json["work_ids"]
    blocklist = request.json.get("blocklist", [])
    included_tags = request.json.get("included_tags", [])
    excluded_tags = request.json.get("excluded_tags", [])
    work_score = qh.multi_query(
        work_ids=work_ids,
        blocklist=blocklist,
        included_tags=included_tags,
        excluded_tags=excluded_tags,
        num_results=20,
    )
    return {"results": [{"id": work_id, "score": score} for work_id, score in work_score]}


run(host="localhost", port=8080)
