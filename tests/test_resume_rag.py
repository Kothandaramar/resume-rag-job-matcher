import unittest

from resume_rag import chunk_document, extract_metadata
from job_matcher import extract_requirements, match_resumes


class FakeIndex:
    def __init__(self, rows):
        self.rows = rows

    def search(self, query, top_k=10):
        return self.rows


class ResumeProcessingTests(unittest.TestCase):
    def test_chunking_preserves_section_labels_and_window_limits(self):
        text = "Education\nBSc Computer Science\nExperience\n" + ("Python engineer. " * 100)
        chunks = chunk_document(text, max_chars=100, overlap=10)
        self.assertEqual(chunks[0].section, "Education")
        self.assertTrue(any(chunk.section == "Experience" for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))

    def test_metadata_extraction(self):
        text = (
            "Alex Morgan\nSkills\nPython, PyTorch, SQL\n"
            "Experience\n5+ years of experience building ML systems\n"
            "Education\nM.S. Computer Science"
        )
        name, skills, years, education = extract_metadata(text, "resume.txt")
        self.assertEqual(name, "Alex Morgan")
        self.assertIn("Python", skills)
        self.assertGreaterEqual(years, 5)
        self.assertIn("Computer Science", education)

    def test_jd_requirement_extraction(self):
        skills, years = extract_requirements(
            "Need 5+ years of experience with Python, PyTorch, and SQL."
        )
        self.assertEqual({skill.lower() for skill in skills}, {"python", "pytorch", "sql"})
        self.assertEqual(years, 5)

    def test_matcher_filters_must_haves_and_returns_json_shape(self):
        index = FakeIndex([
            {
                "metadata": {
                    "resume_path": "good.txt",
                    "candidate_name": "A Candidate",
                    "skills": '["Python", "PyTorch"]',
                    "experience_years": 6,
                    "education": "MSc",
                    "section": "Experience",
                },
                "text": "Built Python and PyTorch machine learning systems.",
                "distance": 0.1,
            },
            {
                "metadata": {
                    "resume_path": "bad.txt",
                    "candidate_name": "B Candidate",
                    "skills": '["Python"]',
                    "experience_years": 8,
                    "education": "",
                    "section": "Skills",
                },
                "text": "Python developer.",
                "distance": 0.01,
            },
        ])
        result = match_resumes(
            "5+ years of experience with Python and PyTorch.",
            index,
            top_k=10,
        )
        self.assertEqual(len(result["top_matches"]), 1)
        self.assertEqual(result["top_matches"][0]["candidate_name"], "A Candidate")
        self.assertEqual(result["top_matches"][0]["match_score"], 94)
        self.assertIn("latency_ms", result["metrics"])


if __name__ == "__main__":
    unittest.main()
