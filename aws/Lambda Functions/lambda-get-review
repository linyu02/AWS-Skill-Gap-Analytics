import os
import json
from decimal import Decimal

import boto3
from botocore.config import Config

CFG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={"max_attempts": 2, "mode": "standard"}
)

dynamodb = boto3.resource("dynamodb", config=CFG)
table = dynamodb.Table(os.environ["REVIEW_TABLE"].strip())


def to_json_safe(value):
    if isinstance(value, Decimal):
        # convert whole numbers to int, others to float
        if value % 1 == 0:
            return int(value)
        return float(value)
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    return value


def response(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(to_json_safe(body))
    }


def lambda_handler(event, context):
    path_params = event.get("pathParameters") or {}
    review_id = path_params.get("review_id")

    if not review_id:
        return response(400, {"message": "Missing path parameter: review_id"})

    resp = table.get_item(Key={"review_id": review_id})
    item = resp.get("Item")

    if not item:
        return response(404, {"message": "Review not found"})

    return response(200, {
        "review_id": item["review_id"],
        "status": item.get("status"),
        "payload": item.get("payload", {}),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at")
    })
