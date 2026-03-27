# Skill Gap Analytics (AWS + LLM)

I thought I needed to learn everything—so I built a system to find what actually matters.

Built with AWS serverless + LLMs, this pipeline extracts, aggregates, and analyzes job market demand to guide learning decisions.

![Demo](./data%20analysis/weekly_skill_demand.gif)
---

## 🔗 Explore More

- 📖 **Project Story** → [How a Math Student Turned Job Market Noise into Learning Decisions](https://www.notion.so/How-a-Math-Student-Turned-Job-Market-Noise-into-Learning-Decisions-3277b91442b380a69023e2a73bf1f489)
- ⚙️ **Technical Appendix** → [AWS Architecture for the Job Market Skill Tracker](https://www.notion.so/AWS-Architecture-for-the-Job-Market-Skill-Tracker-32c7b91442b380008e73ddcfab0ed73c)

---

## 💡 Why I Built This

Job descriptions made it feel like:
> “I need to learn everything.”

So I treated the job market like a dataset and asked:

> **Which skills actually matter across the roles I care about?**

---

## 📊 Key Findings

- The market is **more structured than it feels**  
- A small set of skills shows up **consistently across roles**  
- ~30% of top skills are **soft skills**; **communication appears in ~90%**  
- My resume already covered **~85% of demand**  

👉 The problem wasn’t “learn everything” —  
it was **identify high-leverage gaps + communicate existing skills better**

---

## What This System Does 

- Extracts skills from job descriptions using an LLM with evidence  
- Tracks **weekly skill demand trends**  
- Compares market demand against a resume  
- Outputs **targeted skill gaps and next steps**

## Reference Architecture
![Architecture](./AWS_SkillGapAnalytisc_Architecture.drawio.png)
---

## What’s in This Repo

```bash
lambda/              # extraction + resume comparison  
step_functions/      # orchestration  
data_analysis/       # code for visualization + analysis  
sample_outputs/      # structured outputs  
