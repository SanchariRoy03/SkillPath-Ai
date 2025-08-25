# roadmap_engine.py
from typing import List, Dict, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

CAREERS = {'Data Scientist': ['Python', 'Statistics', 'Probability', 'Data Wrangling', 'Exploratory Data Analysis', 'Machine Learning', 'Model Evaluation', 'SQL', 'Feature Engineering', 'Visualization', 'Experiment Design'], 'Machine Learning Engineer': ['Python', 'Data Structures', 'Algorithms', 'Machine Learning', 'Deep Learning', 'Model Serving', 'MLOps', 'Docker', 'Cloud (AWS/GCP/Azure)', 'CI/CD', 'Monitoring'], 'Frontend Developer': ['HTML', 'CSS', 'JavaScript', 'TypeScript', 'React', 'State Management', 'Testing', 'Performance Optimization', 'Accessibility', 'REST APIs', 'Tooling (Webpack/Vite)']}

PREREQS = {'Statistics': ['Probability'], 'Exploratory Data Analysis': ['Data Wrangling'], 'Machine Learning': ['Python', 'Statistics', 'Algorithms'], 'Model Evaluation': ['Machine Learning'], 'Feature Engineering': ['Machine Learning'], 'Deep Learning': ['Machine Learning', 'Linear Algebra'], 'Model Serving': ['Machine Learning', 'Docker'], 'MLOps': ['Docker', 'CI/CD', 'Cloud (AWS/GCP/Azure)'], 'React': ['JavaScript'], 'State Management': ['React'], 'Testing': ['JavaScript'], 'Performance Optimization': ['JavaScript'], 'Accessibility': ['HTML', 'CSS']}

RESOURCE_BANK = {'Python': ['Course: Intro to Python', 'Book: Automate the Boring Stuff'], 'Statistics': ['Course: Stats with Python', 'Book: Practical Statistics'], 'Probability': ['Course: Probability Essentials'], 'Data Wrangling': ['Course: Data Cleaning in Pandas'], 'Exploratory Data Analysis': ['Course: EDA with Pandas & Matplotlib'], 'Machine Learning': ['Course: ML Specialization', 'Book: Hands-On ML'], 'Model Evaluation': ['Article: Metrics (Precision/Recall/AUC)'], 'SQL': ['Course: SQL for Data Analysis'], 'Feature Engineering': ['Course: Feature Engineering'], 'Visualization': ['Course: Effective Data Visualization'], 'Experiment Design': ['Article: A/B Testing Fundamentals'], 'Data Structures': ['Course: DS in Python'], 'Algorithms': ['Course: Algorithms (Greedy/DP)'], 'Deep Learning': ['Course: Deep Learning', 'Book: Dive into DL'], 'Model Serving': ['Article: FastAPI for ML Serving'], 'MLOps': ['Course: MLOps Basics'], 'Docker': ['Course: Docker for Devs'], 'Cloud (AWS/GCP/Azure)': ['Course: Cloud for ML Engineers'], 'CI/CD': ['Guide: CI/CD Pipelines'], 'Monitoring': ['Article: ML Monitoring 101'], 'HTML': ['Course: HTML5 Basics'], 'CSS': ['Course: CSS Flex/Grid'], 'JavaScript': ['Course: Modern JavaScript'], 'TypeScript': ['Course: TypeScript Fundamentals'], 'React': ['Course: React Complete Guide'], 'State Management': ['Course: Redux/Zustand/Recoil'], 'Testing': ['Course: Jest/RTL'], 'Performance Optimization': ['Guide: Web Performance'], 'Accessibility': ['Guide: WCAG Basics'], 'REST APIs': ['Course: REST API Design'], 'Tooling (Webpack/Vite)': ['Guide: Vite/Webpack'], 'Linear Algebra': ['Course: Linear Algebra for ML']}

ALL_SKILLS = sorted({s for skills in CAREERS.values() for s in skills} | set(PREREQS.keys()))

class SkillExtractor:
    def __init__(self, skill_bank: List[str]):
        corpus = skill_bank + [s.lower() for s in skill_bank]
        self.skill_bank = skill_bank
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3)).fit(corpus)
        self.skill_matrix = self.vectorizer.transform(skill_bank)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("&", "and")
        text = re.sub(r"[^a-zA-Z0-9\-\/\+\s\(\)]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract(self, text: str, top_k_per_span: int = 3, threshold: float = 0.35) -> List[Tuple[str, float]]:
        text = self._normalize_text(text)
        tokens = text.split()
        spans = []
        window = 8
        for i in range(0, len(tokens), max(1, window // 2)):
            span = " ".join(tokens[i:i + window])
            if span:
                spans.append(span)

        detected = {}
        for span in spans:
            vec = self.vectorizer.transform([span])
            sims = cosine_similarity(vec, self.skill_matrix).flatten()
            top_ids = sims.argsort()[::-1][:top_k_per_span]
            for idx in top_ids:
                skill = self.skill_bank[idx]
                score = float(sims[idx])
                if score >= threshold:
                    detected[skill] = max(detected.get(skill, 0.0), score)

        return sorted(detected.items(), key=lambda x: x[1], reverse=True)


class LearningPathPlanner:
    def __init__(self, prereqs: Dict[str, List[str]]):
        self.G = nx.DiGraph()
        for skill, pres in prereqs.items():
            for p in pres:
                self.G.add_edge(p, skill)
        for s in ALL_SKILLS:
            if s not in self.G.nodes:
                self.G.add_node(s)

    def order_missing_skills(self, missing: List[str]) -> List[str]:
        needed = set()
        for m in missing:
            if m in self.G:
                needed |= nx.ancestors(self.G, m)
            needed.add(m)
        SG = self.G.subgraph(needed).copy()
        try:
            order = list(nx.topological_sort(SG))
        except nx.NetworkXUnfeasible:
            order = sorted(SG.nodes(), key=lambda n: SG.in_degree(n))
        return [s for s in order if s in set(missing)]


class ResourceRecommender:
    def __init__(self, bank: Dict[str, List[str]]):
        self.bank = bank
    def recommend(self, skill: str, k: int = 2) -> List[str]:
        return self.bank.get(skill, [])[:k]


class CareerPathPredictor:
    def __init__(self):
        self.roles = ["Intern", "Junior Data Scientist", "Data Scientist", "Senior Data Scientist",
                      "ML Engineer", "Senior ML Engineer", "Data Science Manager"]
        self.P = np.array([
            [0.05, 0.55, 0.25, 0.0, 0.10, 0.0, 0.05],
            [0.0, 0.10, 0.65, 0.10, 0.10, 0.03, 0.02],
            [0.0, 0.05, 0.10, 0.60, 0.15, 0.05, 0.05],
            [0.0, 0.0, 0.05, 0.10, 0.45, 0.25, 0.15],
            [0.0, 0.0, 0.05, 0.20, 0.20, 0.35, 0.20],
            [0.0, 0.0, 0.0, 0.10, 0.10, 0.30, 0.50],
            [0.0, 0.0, 0.0, 0.05, 0.05, 0.20, 0.70],
        ])
        self.P = self.P / self.P.sum(axis=1, keepdims=True)

    def next_k_steps(self, start_role: str, k: int = 3):
        if start_role not in self.roles:
            raise ValueError(f"Unknown role: {start_role}")
        idx = self.roles.index(start_role)
        state = np.zeros(len(self.roles))
        state[idx] = 1.0
        Pk = np.linalg.matrix_power(self.P, k)
        probs = state @ Pk
        ranked = sorted([(r, float(probs[i])) for i, r in enumerate(self.roles)],
                        key=lambda x: x[1], reverse=True)
        return ranked[:5]


class RoadmapEngine:
    def __init__(self, careers, prereqs, resource_bank):
        self.careers = careers
        self.extractor = SkillExtractor(ALL_SKILLS)
        self.planner = LearningPathPlanner(prereqs)
        self.resources = ResourceRecommender(resource_bank)
        self.path_predictor = CareerPathPredictor()

    def generate(self, user_text: str, target_career: str, already_known: List[str] = None, weekly_hours: int = 8):
        if target_career not in self.careers:
            raise ValueError(f"Unknown target career: {target_career}")
        already_known = set(already_known or [])
        extracted = self.extractor.extract(user_text)
        extracted_skills = {s for s, _ in extracted}
        known = set(extracted_skills) | set(already_known)
        required = set(self.careers[target_career])
        missing = sorted(list(required - known))
        ordered_missing = self.planner.order_missing_skills(missing)
        pace = 2 if weekly_hours >= 10 else 1
        timeline = []
        month = 1
        for i in range(0, len(ordered_missing), pace):
            batch = ordered_missing[i:i + pace]
            block = {
                "Month": month,
                "Focus Skills": batch,
                "Resources": {sk: self.resources.recommend(sk) for sk in batch},
                "Projects": self._suggest_projects(batch, target_career)
            }
            timeline.append(block)
            month += 1
        start_role = "Junior Data Scientist" if "Data" in target_career else "Intern"
        next_roles = self.path_predictor.next_k_steps(start_role, k=3)
        return {
            "target_career": target_career,
            "known_skills": sorted(list(known)),
            "missing_skills": ordered_missing,
            "timeline": timeline,
            "career_outlook": next_roles,
            "extracted_skills_scored": extracted
        }

    @staticmethod
    def _suggest_projects(skills_batch: List[str], career: str) -> List[str]:
        ideas = []
        if "Machine Learning" in skills_batch:
            ideas.append("Build an end-to-end ML pipeline on a public dataset")
        if "Model Serving" in skills_batch:
            ideas.append("Serve a trained model via FastAPI/Docker")
        if "MLOps" in skills_batch:
            ideas.append("Set up CI/CD to auto-deploy model updates")
        if "React" in skills_batch:
            ideas.append("Create a React dashboard for model predictions")
        if not ideas:
            ideas.append(f"Mini project tying {', '.join(skills_batch)} to {career}")
        return ideas


if __name__ == "__main__":
    user_resume_text = """
    B.Tech (CSE-DS). Strong in Python, Statistics, EDA, Visualization.
    Experience with SQL and Pandas. Built a small CNN in PyTorch.
    Containerization basics with Docker. Learning AWS. Interested in MLOps.
    """
    engine = RoadmapEngine(CAREERS, PREREQS, RESOURCE_BANK)
    result = engine.generate(user_text=user_resume_text,
                             target_career="Machine Learning Engineer",
                             already_known=["Linear Algebra"],
                             weekly_hours=12)
    rows = []
    for block in result["timeline"]:
        for sk in block["Focus Skills"]:
            rows.append({
                "Month": block["Month"],
                "Skill": sk,
                "Top Resources": " | ".join(block["Resources"].get(sk, [])),
                "Project Suggestion(s)": " | ".join(block["Projects"])
            })
    df = pd.DataFrame(rows)
    print("Known skills:", result["known_skills"])
    print("Missing skills (ordered):", result["missing_skills"])
    print("Career outlook (3 steps):", result["career_outlook"])
    print("\\nTimeline:\\n", df.to_string(index=False))
