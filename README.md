# Skill Gap Analytics (AWS + LLM)

Turn messy job postings into clear learning priorities.

![Demo](./data%20analysis/weekly_skill_demand.gif)

---

## Why I Built This

Job descriptions often made me feel like I needed to learn *everything* while market demand kept changing.

So I reframed the problem:

> Instead of asking “what should I learn?”,
> I asked **“what actually matters most?”**

👉 [Project story](YOUR_NOTION_LINK)

---

## Key Takeaways

![Top Skills](./assets/top_skills.png)

* A small set of skills appeared consistently across many roles
* About 30% of the top skills were soft skills
* Communication showed up in about 90% of job descriptions

---

## How It Works

S3 → Lambda → Step Functions → DynamoDB → S3

1. Upload weekly job descriptions and my resume to S3
2. Prevent duplicate weekly runs
3. Extract and normalize skills with Bedrock
4. Retrieve top weekly skills
5. Compare market demand against my resume
6. Request human review
7. Write final recommendations

![Architecture](./assets/architecture.png)

👉 [System design](YOUR_NOTION_LINK)

---

## Repo Guide

* `lambda/`
  Core AWS Lambda functions for:

  * skill extraction
  * top skill retrieval
  * resume comparison
  * human review request
  * final recommendation writing

* `step_functions/`
  Workflow definition for pipeline orchestration

* `data analysis/`
  Local analysis and visualization files, including charts and GIFs

* `assets/`
  README visuals and architecture diagram

* `sample_output/`
  Example outputs such as matched skills, gaps, and recommendations

---

## Stack

Python · AWS (S3, Lambda, Step Functions, DynamoDB, SNS, IAM) · Bedrock · Pandas
