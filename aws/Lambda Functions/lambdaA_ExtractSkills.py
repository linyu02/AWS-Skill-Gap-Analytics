import os, json
import math
from datetime import datetime, timezone
from collections import Counter

import boto3
from botocore.config import Config

# -------------------------------
# JSON Schema for Structured Output
# -------------------------------
SKILL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["technical_skills", "soft_skills"],
    "properties": {
        "technical_skills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill", "evidence"],
                "properties": {
                    "skill": {"type": "string"},
                    "evidence": {"type": "string"}
                }
            }
        },
        "soft_skills": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["skill", "evidence"],
                "properties": {
                    "skill": {"type": "string"},
                    "evidence": {"type": "string"}
                }
            }
        }
    }
}
# --- clients with sane timeouts ---
CFG = Config(
    connect_timeout=5,
    read_timeout=25,
    retries={"max_attempts": 2, "mode": "standard"},
)

s3 = boto3.client("s3", config=CFG)
ddb = boto3.client("dynamodb", config=CFG)

# Region: if your Lambda runs in one region but Bedrock is in another, set BEDROCK_REGION
brt = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("BEDROCK_REGION"),
    config=CFG
)

# ---------- S3 helpers ----------
def list_txt_keys(bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            k = item["Key"]
            if k.lower().endswith(".txt") and not k.endswith("/"):
                keys.append(k)
    return keys

def load_s3_text(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8", errors="replace")

# ---------- Skill normalization ----------
CANON = {
    "amazon web services": "aws",
    "ml": "machine learning",
    "time series": "time-series",
    "time-series forecasting": "time series forecasting",
    "data analysis": "data analytics",
    "python programming":"python"

}

def canonicalize(skill: str) -> str:
    s = (skill or "").strip().lower()
    return CANON.get(s, s)

# ---------- Bedrock extraction ----------
def bedrock_extract_with_evidence(job_text: str, top_n: int) -> dict:
    model_id = os.environ["MODEL_ID"]

    system = (
        "You are an information extraction system. "
        "Return JSON only. Do not add any commentary."
    )

    prompt = f"""
Extract REQUIRED skills from this job description.

Return JSON ONLY in this exact schema:
{{
  "technical_skills": [
    {{"skill": "technical skill", "evidence": "exact quote from the job description"}}
  ],
  "soft_skills": [
    {{"skill": "soft skill", "evidence": "exact quote from the job description"}}
  ]
}}

STRICT RULES:
- ONLY include skills explicitly supported by the job description.
- Do NOT infer or guess skills.
- Each "evidence" MUST be an exact short quote (2–12 words) copied verbatim from the job description.
- Normalize skill strings:
  - lowercase
  - remove fluff words like "strong", "experience with", "familiarity with"
  - prefer canonical names (e.g., "aws" not "amazon web services"; "machine learning" not "ml")
- Exclude degrees, years of experience, and responsibilities (only skills).
- Return up to {top_n} technical skills and up to {top_n} soft skills (fewer is ok).
- JSON only. No commentary.

JOB DESCRIPTION:
<<<
{job_text}
>>>
""".strip()

    body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 3000,
    "temperature": 0.0,
    "system": system,
    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],

    # ✅ Structured output config
    "output_config": {
        "format": {
            "type": "json_schema",
            "schema": SKILL_SCHEMA
        }
    }
    }

    resp = brt.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body).encode("utf-8"),
    )

    raw = resp["body"].read().decode("utf-8", errors="replace")
    payload = json.loads(raw)

    # Anthropic responses are content blocks
    text_out = payload["content"][0]["text"]
    parsed = json.loads(text_out)
    # ---- debug ---- 
    #print("===== RAW text_out START =====")
    #print(text_out)
    #print("===== RAW text_out END =====")
    #parsed = json.loads(text_out)


    def validate(items):
        cleaned = []
        if not isinstance(items, list):
            return cleaned
        for it in items:
            if not isinstance(it, dict):
                continue
            skill = (it.get("skill") or "").strip()
            evidence = (it.get("evidence") or "").strip()

            # strict evidence check
            if not skill or not evidence:
                continue
            if evidence not in job_text:
                continue

            cleaned.append({"skill": skill.lower(), "evidence": evidence})

        # de-dupe by skill
        seen = set()
        out = []
        for it in cleaned:
            if it["skill"] not in seen:
                out.append(it)
                seen.add(it["skill"])
        return out

    return {
        "technical_skills": validate(parsed.get("technical_skills"))[:top_n],
        "soft_skills": validate(parsed.get("soft_skills"))[:top_n],
    }

# -----------embed layer ----------------
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v1")

def bedrock_embed(texts: list[str]) -> list[list[float]]:
    vecs = []
    for t in texts:
        body = {"inputText": t}
        resp = brt.invoke_model(
            modelId=EMBED_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body).encode("utf-8"),
        )
        raw = resp["body"].read().decode("utf-8", errors="replace")
        payload = json.loads(raw)

        emb = payload.get("embedding")
        if not emb:
            raise ValueError(f"Embedding missing for text={t!r}; keys={list(payload.keys())}")
        vecs.append(emb)
    return vecs


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x*x for x in vec)) + 1e-12
    return [x / norm for x in vec]

def dot(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))
def merge_similar_skills(counts: Counter, threshold: float = 0.7):
    skills = list(counts.keys())
    if len(skills) <= 1:
        return counts, {s: s for s in skills}

    E = [l2_normalize(v) for v in bedrock_embed(skills)]
    n = len(skills)
    adj = [[] for _ in range(n)]

    # O(n^2) graph using cosine similarity (dot of normalized vectors)
    for i in range(n):
        for j in range(i + 1, n):
            if dot(E[i], E[j]) >= threshold:
                adj[i].append(j)
                adj[j].append(i)

    # connected components
    seen = [False] * n
    clusters = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        clusters.append(comp)

    # choose canonical: most frequent, tie -> shortest
    mapping = {}
    for comp in clusters:
        group = [skills[k] for k in comp]
        canon = sorted(group, key=lambda s: (-counts[s], len(s), s))[0]
        for s in group:
            mapping[s] = canon

    new_counts = Counter()
    for s, c in counts.items():
        new_counts[mapping[s]] += c

    return new_counts, mapping

# ---------- DynamoDB writers ----------
def ddb_set_weekly(table: str, week: str, skill: str, category: str, n: int, run_utc: str):
    ddb.update_item(
        TableName=table,
        Key={"Week": {"S": week}, "Skill": {"S": skill}},
        UpdateExpression="SET job_count = :n, category = :cat, last_updated_utc = :t",
        ExpressionAttributeValues={
            ":n": {"N": str(n)},
            ":cat": {"S": category},
            ":t": {"S": run_utc},
        },
    )

def ddb_upsert_registry(table: str, skill: str, week: str, run_utc: str):
    ddb.update_item(
        TableName=table,
        Key={"Skill": {"S": skill}},
        UpdateExpression=(
            "SET "
            "first_seen_week = if_not_exists(first_seen_week, :w), "
            "first_seen_utc  = if_not_exists(first_seen_utc,  :t), "
            "last_seen_week  = :w, "
            "last_seen_utc   = :t"
        ),
        ExpressionAttributeValues={":w": {"S": week}, ":t": {"S": run_utc}},
    )

# ---------- Lambda handler ----------
def lambda_handler(event, context):
    bucket = event.get("bucket") or os.environ["BUCKET_NAME"]
    prefix = event["prefix"]
    week = event["week"]
    top_n = int(event.get("top_n") or os.environ.get("DEFAULT_TOP_N", "15"))

    run_utc = datetime.now(timezone.utc).isoformat()
    weekly_table = os.environ["WEEKLY_TABLE"]
    registry_table = os.environ["REGISTRY_TABLE"]

    keys = list_txt_keys(bucket, prefix)
    if not keys:
        return {"status": "ok", "week": week, "jobs_processed": 0, "unique_skills": 0}

    counts_tech = Counter()
    counts_soft = Counter()
    registry_skills = set()

    processed = 0
    for key in keys:
        print(f"[DEBUG] processing key={key}")
        job_text = load_s3_text(bucket, key)
        result = bedrock_extract_with_evidence(job_text, top_n=top_n)

        tech = {canonicalize(x["skill"]) for x in result["technical_skills"]}
        soft = {canonicalize(x["skill"]) for x in result["soft_skills"]}

        for sk in tech:
            counts_tech[sk] += 1
            registry_skills.add(sk)

        for sk in soft:
            counts_soft[sk] += 1
            registry_skills.add(sk)

        processed += 1
    # --- merge similar skills across all jobs this run ---
    threshold = float(os.environ.get("SKILL_SIM_THRESHOLD", "0.86"))
    counts_tech, tech_map = merge_similar_skills(counts_tech, threshold=threshold)
    counts_soft, soft_map = merge_similar_skills(counts_soft, threshold=threshold)
    # rebuild registry_skills AFTER merging
    registry_skills = set(counts_tech.keys()) | set(counts_soft.keys())


    # write once per skill
    for sk, n in counts_tech.items():
        ddb_set_weekly(weekly_table, week, sk, "technical", n, run_utc)

    for sk, n in counts_soft.items():
        ddb_set_weekly(weekly_table, week, sk, "soft", n, run_utc)

    for sk in registry_skills:
        ddb_upsert_registry(registry_table, sk, week, run_utc)

    top10 = (counts_tech + counts_soft).most_common(10)

    return {
        "status": "ok",
        "week": week,
        "jobs_processed": processed,
        "unique_skills": len(registry_skills),
        "top10_this_run": [{"skill": s, "job_count": c} for s, c in top10],
    }

