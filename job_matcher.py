"""Hybrid resume search and explainable candidate ranking."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from resume_rag import ResumeIndex, make_embedder


def extract_requirements(job_description: str) -> tuple[list[str], float]:
    """Infer must-have skills and a minimum experience requirement from JD text."""
    skill_matches = re.findall(
        r"\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|SQL|R|Go|Rust|Scala|AWS|Azure|GCP|"
        r"Docker|Kubernetes|Terraform|React|Node\.js|Django|Flask|FastAPI|Spring|"
        r"Machine Learning|Deep Learning|NLP|LLM|Generative AI|PyTorch|TensorFlow|"
        r"scikit-learn|Pandas|NumPy|Spark|Hadoop|Airflow|ETL|MLOps|"
        r"PostgreSQL|MongoDB|Redis|Snowflake|Tableau|Power BI|Excel)\b",
        job_description,
        re.IGNORECASE,
    )
    unique_skills: list[str] = []
    for skill in skill_matches:
        normalized = skill.lower()
        if normalized not in {item.lower() for item in unique_skills}:
            unique_skills.append(skill)
    years = [
        float(match.group(1))
        for match in re.finditer(
            r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:relevant\s+)?experience\b",
            job_description,
            re.IGNORECASE,
        )
    ]
    return unique_skills, max(years, default=0.0)


def match_resumes(
    job_description: str,
    index: ResumeIndex,
    top_k: int = 10,
    must_have_skills: list[str] | None = None,
    min_experience_years: float | None = None,
) -> dict[str, Any]:
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")
    start = time.perf_counter()
    inferred_skills, inferred_years = extract_requirements(job_description)
    required_skills = must_have_skills if must_have_skills is not None else inferred_skills
    min_years = inferred_years if min_experience_years is None else min_experience_years
    rows = index.search(job_description, top_k=max(top_k, 1))
    candidates: dict[str, dict[str, Any]] = {}

    for row in rows:
        metadata = row["metadata"]
        resume_path = metadata["resume_path"]
        candidate = candidates.setdefault(
            resume_path,
            {
                "candidate_name": metadata["candidate_name"],
                "resume_path": resume_path,
                "skills": set(json.loads(metadata.get("skills", "[]"))),
                "experience_years": float(metadata.get("experience_years", 0)),
                "education": metadata.get("education", ""),
                "semantic": 0.0,
                "excerpts": [],
                "sections": set(),
                "keyword_hits": set(),
            },
        )
        semantic = max(0.0, min(1.0, 1.0 - float(row["distance"])))
        candidate["semantic"] = max(candidate["semantic"], semantic)
        candidate["excerpts"].append((semantic, row["text"]))
        candidate["sections"].add(metadata.get("section", "General"))
        doc_text = row["text"].lower()
        for skill in required_skills:
            if skill.lower() in doc_text:
                candidate["keyword_hits"].add(skill)

    ranked: list[dict[str, Any]] = []
    for candidate in candidates.values():
        matched_skills = sorted(
            skill for skill in candidate["skills"]
            if any(skill.lower() == required.lower() for required in required_skills)
        )
        if required_skills and any(
            not any(skill.lower() == found.lower() for found in candidate["skills"])
            for skill in required_skills
        ):
            continue
        if candidate["experience_years"] < min_years:
            continue

        skill_coverage = (
            len(matched_skills) / len(required_skills) if required_skills else 0.0
        )
        lexical_coverage = (
            len(candidate["keyword_hits"]) / len(required_skills) if required_skills else 0.0
        )
        if required_skills:
            score = 100 * (
                0.65 * candidate["semantic"]
                + 0.25 * skill_coverage
                + 0.10 * lexical_coverage
            )
        else:
            score = 100 * candidate["semantic"]
        excerpts = [
            text for _, text in sorted(candidate["excerpts"], reverse=True)[:3]
        ]
        section_names = sorted(candidate["sections"])
        reasoning = (
            f"Matched {len(matched_skills)} of {len(required_skills)} required skills; "
            f"{candidate['experience_years']:g} years of experience. "
            f"Relevant resume sections: {', '.join(section_names)}."
        )
        ranked.append(
            {
                "candidate_name": candidate["candidate_name"],
                "resume_path": candidate["resume_path"],
                "match_score": round(max(0, min(100, score))),
                "matched_skills": matched_skills,
                "relevant_excerpts": excerpts,
                "reasoning": reasoning,
            }
        )

    ranked.sort(key=lambda item: (-item["match_score"], item["candidate_name"]))
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "job_description": job_description,
        "top_matches": ranked[:top_k],
        "metrics": {"latency_ms": elapsed_ms},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="Text file containing a job description")
    parser.add_argument("--resumes", type=Path, default=Path("data/sample_resumes"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--provider", choices=("huggingface", "openai"), default="huggingface")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection", default="resumes")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--must-have", nargs="*", default=None)
    parser.add_argument("--min-years", type=float, default=None)
    args = parser.parse_args()

    embedder = make_embedder(args.provider, args.model)
    index = ResumeIndex(args.db, args.collection, embedder)
    if index.collection.count() == 0:
        index.ingest_directory(args.resumes)
    result = match_resumes(
        args.job.read_text(encoding="utf-8"),
        index,
        top_k=args.top_k,
        must_have_skills=args.must_have,
        min_experience_years=args.min_years,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
