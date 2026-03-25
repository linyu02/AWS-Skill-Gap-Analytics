import os
from decimal import Decimal

import boto3
from botocore.config import Config

CFG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={"max_attempts": 2, "mode": "standard"},
)

ddb = boto3.client("dynamodb", config=CFG)

def lambda_handler(event, context):
    table = os.environ["WEEKLY_TABLE"]

    week = event.get("week")
    if not week:
        raise ValueError("Missing required field: 'week' (e.g. 2026-W03)")

    limit = int(event.get("limit", 10))
    category_filter = event.get("category")  # optional: "technical" or "soft"

    # Query by Week (partition key)
    items = []
    last_evaluated_key = None

    expr_attr_names = {"#W": "Week"}
    expr_attr_values = {":w": {"S": week}}

    # Optional category filter
    filter_expression = None
    if category_filter:
        expr_attr_names["#C"] = "category"
        expr_attr_values[":c"] = {"S": category_filter}
        filter_expression = "#C = :c"

    while True:
        kwargs = {
            "TableName": table,
            "KeyConditionExpression": "#W = :w",
            "ExpressionAttributeNames": expr_attr_names,
            "ExpressionAttributeValues": expr_attr_values,
        }
        if filter_expression:
            kwargs["FilterExpression"] = filter_expression

        if last_evaluated_key:
            kwargs["ExclusiveStartKey"] = last_evaluated_key

        resp = ddb.query(**kwargs)
        items.extend(resp.get("Items", []))
        last_evaluated_key = resp.get("LastEvaluatedKey")

        if not last_evaluated_key:
            break

    # Convert and sort
    parsed = []
    for it in items:
        skill = it.get("Skill", {}).get("S")
        cnt_s = it.get("job_count", {}).get("N", "0")
        category = it.get("category", {}).get("S")
        if skill:
            parsed.append({
                "skill": skill,
                "job_count": int(Decimal(cnt_s)),
                "category": category
            })

    parsed.sort(key=lambda x: x["job_count"], reverse=True)
    top = parsed[:limit]

    return {
        "status": "ok",
        "week": week,
        "category": category_filter,
        "returned": len(top),
        "top_skills": top,
        "total_skills_in_week": len(parsed),
    }
