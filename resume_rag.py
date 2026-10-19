"""Resume ingestion, section-aware chunking, metadata extraction, and vector storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fs_tools import list_files, read_file

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
SECTION_NAMES = {
    "summary": "Summary",
    "profile": "Summary",
    "professional summary": "Summary",
    "experience": "Experience",
    "work experience": "Experience",
    "employment history": "Experience",
    "education": "Education",
    "skills": "Skills",
    "technical skills": "Skills",
    "projects": "Projects",
    "certifications": "Certifications",
}
SKILL_TERMS = (
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "SQL", "R",
    "Go", "Rust", "Scala", "AWS", "Azure", "GCP", "Docker", "Kubernetes",
    "Terraform", "Linux", "React", "Node.js", "Django", "Flask", "FastAPI",
    "Spring", "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
    "LLM", "Generative AI", "PyTorch", "TensorFlow", "scikit-learn", "Pandas",
    "NumPy", "Spark", "Hadoop", "Airflow", "ETL", "Data Engineering",
    "Data Analysis", "Statistics", "Computer Vision", "MLOps", "CI/CD",
    "Git", "GraphQL", "REST", "PostgreSQL", "MongoDB", "Redis", "Snowflake",
    "Tableau", "Power BI", "Figma", "Product Management", "Agile", "Salesforce",
    "Cybersecurity", "Networking", "HTML", "CSS", "Excel",
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install requirements.txt to use the default Hugging Face embedder."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


class OpenAIEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use OpenAI embeddings.") from exc
        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


def make_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model_name)
    if provider == "huggingface":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")


@dataclass
class ResumeDocument:
    text: str
    path: str
    candidate_name: str
    skills: list[str]
    experience_years: float
    education: str


@dataclass
class DocumentChunk:
    text: str
    section: str
    chunk_index: int


def load_resume_text(path: Path) -> str:
    if path.suffix.lower() == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    result = read_file(str(path))
    if not result["success"]:
        raise RuntimeError(f"Unable to load resume '{path}': {result['error']}")
    return result["content"]


def _section_for_heading(line: str) -> str | None:
    normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return SECTION_NAMES.get(normalized)


def chunk_document(text: str, max_chars: int = 1200, overlap: int = 150) -> list[DocumentChunk]:
    """Split by recognized resume sections first, then by paragraph-sized windows."""
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > overlap >= 0.")
    sections: list[tuple[str, list[str]]] = []
    current_name = "General"
    current_lines: list[str] = []
    for line in text.splitlines():
        heading = _section_for_heading(line)
        if heading:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name, current_lines = heading, []
        elif line.strip():
            current_lines.append(line.strip())
    if current_lines:
        sections.append((current_name, current_lines))

    chunks: list[DocumentChunk] = []
    for section_name, lines in sections:
        section_text = "\n".join(lines)
        paragraphs = re.split(r"\n+", section_text)
        buffer = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if buffer:
                    chunks.append(DocumentChunk(buffer, section_name, len(chunks)))
                    buffer = ""
                step = max_chars - overlap
                for start in range(0, len(paragraph), step):
                    piece = paragraph[start : start + max_chars]
                    if piece:
                        chunks.append(DocumentChunk(piece, section_name, len(chunks)))
                continue
            candidate = f"{buffer}\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) > max_chars and buffer:
                chunks.append(DocumentChunk(buffer, section_name, len(chunks)))
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            chunks.append(DocumentChunk(buffer, section_name, len(chunks)))
    return chunks


def extract_metadata(text: str, filename: str) -> tuple[str, list[str], float, str]:
    first_lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate_name = next(
        (line for line in first_lines[:5] if len(line) <= 70 and not _section_for_heading(line)),
        Path(filename).stem.replace("_", " ").replace("-", " ").title(),
    )
    found_skills = [
        skill for skill in SKILL_TERMS
        if re.search(rf"(?<![\w+#.]){re.escape(skill)}(?![\w+#.])", text, re.IGNORECASE)
    ]
    year_values = [
        float(match.group(1))
        for match in re.finditer(
            r"(?<!\d)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)",
            text,
            re.IGNORECASE,
        )
    ]
    date_ranges = re.findall(
        r"\b(19\d{2}|20\d{2})\s*[-–]\s*(Present|Current|19\d{2}|20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    dated_years = sum(
        max(0, (2026 if end.lower() in {"present", "current"} else int(end)) - int(start))
        for start, end in date_ranges
    )
    experience_years = max(year_values + [float(dated_years), 0.0])
    education_lines: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _section_for_heading(line) == "Education":
            for following_line in lines[index + 1 :]:
                if _section_for_heading(following_line):
                    break
                if following_line.strip():
                    education_lines.append(following_line.strip())
            break
    if not education_lines:
        education_match = re.search(
            r"(?im)^.*(?:\bB\.?\s?S\.?|\bM\.?\s?S\.?|\bB\.?\s?A\.?|\bM\.?\s?A\.?|"
            r"\bPh\.?\s?D\.?|\bBachelor\b|\bMaster\b|\bDoctorate\b|\bAssociate\b|\bDiploma\b).*$",
            text,
        )
        if education_match:
            education_lines = [education_match.group(0).strip()]
    education = " ".join(education_lines)[:300]
    return candidate_name, found_skills, experience_years, education


def build_document(path: Path, base_dir: Path) -> ResumeDocument:
    text = load_resume_text(path)
    name, skills, years, education = extract_metadata(text, path.name)
    return ResumeDocument(
        text=text,
        path=path.relative_to(base_dir).as_posix(),
        candidate_name=name,
        skills=skills,
        experience_years=years,
        education=education,
    )


class ResumeIndex:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to use ChromaDB.") from exc
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

    def ingest_directory(self, resume_dir: Path) -> int:
        base_dir = resume_dir.resolve()
        directories = [base_dir] + sorted(
            path for path in base_dir.rglob("*") if path.is_dir()
        )
        paths: list[Path] = []
        for directory in directories:
            entries = list_files(str(directory))
            errors = [entry["error"] for entry in entries if "error" in entry]
            if errors:
                raise OSError("; ".join(errors))
            paths.extend(
                Path(entry["filepath"])
                for entry in entries
                if Path(entry["filepath"]).suffix.lower() in SUPPORTED_EXTENSIONS
            )
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No supported resumes found under {resume_dir}")
        count = 0
        for path in paths:
            document = build_document(path, base_dir)
            chunks = chunk_document(document.text)
            if not chunks:
                continue
            vectors = self.embedder.embed([chunk.text for chunk in chunks])
            file_key = hashlib.sha256(document.path.encode("utf-8")).hexdigest()[:16]
            ids = [f"{file_key}-{chunk.chunk_index}" for chunk in chunks]
            self.collection.delete(where={"resume_path": document.path})
            self.collection.add(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=vectors,
                metadatas=[
                    {
                        "resume_path": document.path,
                        "candidate_name": document.candidate_name,
                        "skills": json.dumps(document.skills),
                        "experience_years": document.experience_years,
                        "education": document.education,
                        "section": chunk.section,
                    }
                    for chunk in chunks
                ],
            )
            count += 1
        return count

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        size = self.collection.count()
        if not size:
            raise RuntimeError("The resume index is empty. Ingest resumes first.")
        results = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=size,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": doc,
                "metadata": metadata,
                "distance": distance,
            }
            for doc, metadata, distance in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumes", type=Path, default=Path("data/sample_resumes"))
    parser.add_argument("--db", type=Path, default=Path("chroma_db"))
    parser.add_argument("--provider", choices=("huggingface", "openai"), default="huggingface")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--collection", default="resumes")
    args = parser.parse_args()

    index = ResumeIndex(args.db, args.collection, make_embedder(args.provider, args.model))
    print(f"Ingested {index.ingest_directory(args.resumes)} resumes into {args.db}.")


if __name__ == "__main__":
    main()
