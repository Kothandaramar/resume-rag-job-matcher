# Resume RAG and Job Matching

A local-first resume retrieval project: load PDF/DOCX/TXT/Markdown resumes, keep
chunks grouped by recognizable resume sections, create sentence embeddings,
persist them in ChromaDB, and rank candidates using semantic similarity plus
must-have skill and experience filters.

> **Sample data notice:** the 30 profiles and five job descriptions in this
> repository are fictional synthetic examples for coursework/testing. Do not use
> this prototype as the sole basis for hiring decisions.

## Quick start

Python 3.10+ is recommended. Create and activate a virtual environment, then:

```powershell
python -m pip install -r requirements.txt
python resume_rag.py
python job_matcher.py --job data\job_descriptions\01_ml_engineer.txt
```

The first run downloads the default Hugging Face model
`sentence-transformers/all-MiniLM-L6-v2`; subsequent embeddings run locally.
The local persistent Chroma collection is written to `chroma_db\` (ignored by
Git). Re-running ingestion upserts each resume's chunks.

### Search options

```powershell
python job_matcher.py --job data\job_descriptions\02_data_engineer.txt --top-k 10
python job_matcher.py --job data\job_descriptions\01_ml_engineer.txt --must-have Python PyTorch --min-years 5
```

`--must-have` replaces inferred must-have skills. `--min-years` overrides the
experience threshold inferred from phrases such as “5+ years of experience”.
Only candidates satisfying every required skill and the minimum experience are
returned. Output is JSON with the requested candidate fields and measured
latency.

### Optional OpenAI embeddings

Set `OPENAI_API_KEY` in the environment, then pass `--provider openai
--model text-embedding-3-small` to **both** ingestion and search commands. Do not
mix providers/models within a Chroma collection. Hugging Face is the default
and does not require a hosted API key.

## Project map

- `fs_tools.py`: reusable Milestone 1-style file listing, extraction, writing,
  and keyword search utilities.
- `resume_rag.py`: file loading through `fs_tools.py` (TXT/Markdown/PDF/DOCX), section-aware chunking,
  name/skills/experience/education extraction, embeddings, and ChromaDB index.
- `job_matcher.py`: JD requirement inference, hybrid scoring, filtering, and
  explainable JSON output.
- `data/sample_resumes/`: 30 fictional resumes spanning engineering, data,
  product, design, security, and operations.
- `data/job_descriptions/`: five fictional test job descriptions.
- `notebooks/experimentation.ipynb`: runnable retrieval experiments and metrics.
- `demo_video.mp4` and `demo_video.vtt`: 3.5-minute narrated visual walkthrough
  with an optional WebVTT transcript.
- `tests/`: dependency-free unit tests for parsing, chunking, filtering, and
  ranking.

## Chunking and scoring

Chunks remain labeled with their resume section (Experience, Education, Skills,
Projects, etc.); long sections are windowed with overlap. Resume metadata is
stored beside each vector. Search combines cosine similarity (65%), required
skill coverage (25%), and critical-skill lexical evidence (10%). With no
required skills, semantic similarity determines the score. The score is a
heuristic ranking signal, not a calibrated probability.

## Evaluation

Open `notebooks/experimentation.ipynb` after installing dependencies and run all
cells. It reports top-1 retrieval accuracy against the fictional benchmark,
mean reciprocal rank, and per-query latency. These small synthetic-data
measurements are for pipeline validation, not a claim about real-world hiring
performance.

## Demo video

The [3.5-minute narrated walkthrough](./demo_video.mp4) covers the project
architecture, resume ingestion, semantic retrieval, hybrid scoring, must-have
filters, and notebook evaluation. Captions/transcript are available in
[WebVTT format](./demo_video.vtt). The video is a visual walkthrough, not a
screen recording of a live embedding run; the displayed candidate result is
explicitly illustrative. Run the commands above to generate real results.

## Limitations

Metadata extraction uses transparent regular expressions and can miss unusual
resume layouts. PDF extraction depends on selectable text; OCR is not included.
The prototype does not infer protected characteristics, and should not be used
to automate or make employment decisions without appropriate human oversight,
validation, and legal review.
