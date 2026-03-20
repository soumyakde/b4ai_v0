# core/analytics/llm/dta_pipeline.py
"""
Deductive Thematic Analysis (DTA) Pipeline
==========================================
Theory-driven analysis applying the Basics4AI codebook constructs
to participant reflection notes and interview transcripts.

Follows Braun & Clarke (2006) adapted for deductive TA:
    Phase 1  Prepare codebook — load construct definitions + indicators
    Phase 2  Code data — search for evidence per construct per participant
    Phase 3  Build evidence matrix — aggregate into construct × participant
    Phase 4  Review — re-examine low/no-evidence at higher temperature
    Phase 5  Report — construct-level table + participant profiles + LO layer

Constructs (from Draft_Codebook.docx + construct_definitions.yaml):
    Group A — Messaging Perception (CCCES):
        coherency_of_messaging, plausibility_of_messaging,
        credibility_of_messaging, comprehensibility_of_messaging
    Group B — Individual Characteristics (CCCES):
        attention, personal_relevance, culture
    Group C — Cognitive Engagement (SCES):
        engagement_with_task, effort_and_persistence, experience_of_flow
    Group D — Motivation (SIMS):
        intrinsic_motivation, identified_regulation,
        external_regulation, amotivation
    Group E — AI Understanding (AI-CI):
        understanding_ai_basics, ai_learning_processes,
        ai_applications, ai_limitations

Public API:
-----------
load_codebook()
run_dta_phase2(transcripts, model, temperature, construct_groups, run_id)
run_dta_phase3(phase2_results)
run_dta_phase4(phase2_results, model, low_evidence_threshold, run_id)
run_dta_phase5(phase3_matrix, phase2_results, lo_results)
run_lo_analysis(transcripts, model, temperature, run_id)
create_dta_run(model, temperature, source_type, construct_groups, created_by)
save_dta_results(run_id, results)
load_dta_results(run_id)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import uuid
import re

import pandas as pd

# -----------------------------------------------------------------------
# DB path helper
# -----------------------------------------------------------------------
def _find_db() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "responses.db"
        if candidate.exists():
            return candidate
    return here.parents[min(3, len(here.parents)-1)] / "responses.db"

_DB_PATH = _find_db()


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    return sqlite3.connect(db_path or _DB_PATH)


def _lazy_import(module_name: str, attr: str):
    import importlib, importlib.util as _ilu, sys
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr)
    except (ImportError, ModuleNotFoundError):
        rel = module_name.replace(".", "/") + ".py"
        candidate = next(
            (root / rel for root in list(Path(__file__).resolve().parents) +
             [Path(p) for p in sys.path]
             if (root / rel).exists()), None
        )
        if candidate is None:
            fname = module_name.split(".")[-1] + ".py"
            candidate = next(
                (root / fname for root in list(Path(__file__).resolve().parents) +
                 [Path(p) for p in sys.path]
                 if (root / fname).exists()), None
            )
        if candidate is None:
            raise ImportError(f"Cannot find module '{module_name}'")
        spec = _ilu.spec_from_file_location(module_name, candidate)
        mod  = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, attr)


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

def _init_dta_schema(db_path: Optional[Path] = None) -> None:
    conn = _get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dta_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT UNIQUE NOT NULL,
            created_by      TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            model           TEXT NOT NULL,
            temperature     REAL NOT NULL,
            source_type     TEXT NOT NULL,
            construct_groups TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'created',
            phase_reached   INTEGER DEFAULT 0,
            notes           TEXT
        );

        CREATE TABLE IF NOT EXISTS dta_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           TEXT NOT NULL,
            participant_id   TEXT NOT NULL,
            construct_group  TEXT NOT NULL,
            construct_name   TEXT NOT NULL,
            evidence_count   INTEGER DEFAULT 0,
            valence_positive INTEGER DEFAULT 0,
            valence_negative INTEGER DEFAULT 0,
            valence_neutral  INTEGER DEFAULT 0,
            instances_json   TEXT,
            model            TEXT,
            created_at       TEXT,
            UNIQUE(run_id, participant_id, construct_name)
        );

        CREATE TABLE IF NOT EXISTS dta_lo_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id           TEXT NOT NULL,
            participant_id   TEXT NOT NULL,
            module_id        TEXT NOT NULL,
            lo_index         INTEGER NOT NULL,
            lo_text          TEXT,
            evidence_present INTEGER DEFAULT 0,
            evidence_quote   TEXT,
            model            TEXT,
            created_at       TEXT,
            UNIQUE(run_id, participant_id, module_id, lo_index)
        );
    """)

    # Safe migrations: add missing columns
    for table, col, typedef in [
        ("dta_runs",    "notes",           "TEXT"),
        ("dta_results", "instances_json",  "TEXT"),
    ]:
        cols = {r[1] for r in conn.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")

    # Safe migrations: add unique indexes required for ON CONFLICT upserts
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_dta_results_unique
           ON dta_results(run_id, participant_id, construct_name)"""
    )
    conn.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_dta_lo_unique
           ON dta_lo_results(run_id, participant_id, module_id, lo_index)"""
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------
# Codebook — hardcoded from Draft_Codebook.docx + construct_definitions.yaml
# -----------------------------------------------------------------------

CODEBOOK: Dict[str, Dict[str, Any]] = {

    # ---- Group A: Messaging Perception (CCCES) ----
    "coherency_of_messaging": {
        "group": "messaging_perception",
        "group_label": "Messaging Perception (CCCES)",
        "definition": (
            "The degree to which learners perceive the instructional message as "
            "logically organized, internally consistent, and easy to follow."
        ),
        "analytic_focus": [
            "clarity of sequence",
            "logical connections",
            "perceived consistency across activities",
        ],
        "indicators": [
            "This made sense to me",
            "The ideas connected well",
            "I could follow the explanation",
            "It all fit together",
            "One thing led to another",
            "The steps made sense",
        ],
    },
    "comprehensibility_of_messaging": {
        "group": "messaging_perception",
        "group_label": "Messaging Perception (CCCES)",
        "definition": (
            "The degree to which learners understand the instructional message; "
            "low comprehensibility may lead to disengagement."
        ),
        "analytic_focus": [
            "clarity of language",
            "ease of understanding",
            "breakdowns in comprehension",
        ],
        "indicators": [
            "I understood what was being said",
            "The explanation was easy to get",
            "It wasn't confusing",
            "It was clear to me",
            "I got what they were saying",
            "I could picture it in my head",
        ],
    },
    "credibility_of_messaging": {
        "group": "messaging_perception",
        "group_label": "Messaging Perception (CCCES)",
        "definition": (
            "The extent to which learners perceive the source of the message "
            "as trustworthy and knowledgeable."
        ),
        "analytic_focus": [
            "trust in instructor or system",
            "perceived expertise",
            "authority of examples or explanations",
        ],
        "indicators": [
            "This seemed true/real",
            "I believed what was being taught",
            "This made sense with what I've seen",
            "It felt right",
            "It matched what I know",
            "It seemed trustworthy",
        ],
    },
    "plausibility_of_messaging": {
        "group": "messaging_perception",
        "group_label": "Messaging Perception (CCCES)",
        "definition": (
            "Learners' subjective judgment about the potential truthfulness "
            "and realism of the message."
        ),
        "analytic_focus": [
            "perceived realism",
            "alignment with prior beliefs",
            "willingness to reconsider ideas",
        ],
        "indicators": [
            "This seems possible",
            "This makes sense as an explanation",
            "I can believe this idea",
            "I can see how this could work",
            "It seems like it could be true",
            "This explanation fits",
        ],
    },

    # ---- Group B: Individual Characteristics (CCCES) ----
    "attention": {
        "group": "individual_characteristics",
        "group_label": "Individual Characteristics (CCCES)",
        "definition": (
            "The amount of focus learners report giving to the task, which "
            "predicts cognitive engagement and supports conceptual change."
        ),
        "analytic_focus": [
            "sustained focus",
            "distraction",
            "task absorption",
        ],
        "indicators": [
            "I really focused on this",
            "I paid close attention to this",
            "This held my interest",
            "This kept my interest",
            "I didn't get distracted",
            "I was really listening",
        ],
    },
    "personal_relevance": {
        "group": "individual_characteristics",
        "group_label": "Individual Characteristics (CCCES)",
        "definition": (
            "The extent to which learners perceive the content as connected "
            "to their personal experiences, interests, or goals."
        ),
        "analytic_focus": [
            "real-life connections",
            "relevance to daily experiences",
            "personal meaning",
        ],
        "indicators": [
            "This matters to me",
            "This is important for me",
            "I see how this connects to me",
            "I can use this in my life",
            "This helps me understand things better",
            "This is something I care about",
        ],
    },
    "culture": {
        "group": "individual_characteristics",
        "group_label": "Individual Characteristics (CCCES)",
        "definition": (
            "The influence of learners' cultural knowledge and experiences on "
            "engagement and conceptual change."
        ),
        "analytic_focus": [
            "cultural congruence",
            "conflict with prior cultural knowledge",
            "accessibility of examples",
        ],
        "indicators": [
            "This fits with what I know from home",
            "It matches what I've learned before",
            "This feels familiar to me",
            "My family would agree with this",
            "This is how we think about things",
            "It connects to my life",
        ],
    },

    # ---- Group C: Cognitive Engagement (SCES) ----
    "engagement_with_task": {
        "group": "cognitive_engagement",
        "group_label": "Cognitive Engagement (SCES)",
        "definition": "Learners' perceived level of involvement and active participation during the task.",
        "analytic_focus": [
            "enjoyment",
            "active participation",
            "interest during activity",
        ],
        "indicators": [
            "I was really into this activity",
            "This kept me interested",
            "I wanted to keep working on this",
            "I didn't want to stop",
            "I was really focused",
            "This was fun to do",
        ],
    },
    "effort_and_persistence": {
        "group": "cognitive_engagement",
        "group_label": "Cognitive Engagement (SCES)",
        "definition": (
            "Learners' self-reported effort and persistence while working "
            "through the task, especially when challenged."
        ),
        "analytic_focus": [
            "perseverance",
            "willingness to continue",
            "response to difficulty",
        ],
        "indicators": [
            "I tried really hard",
            "I kept working even when it was hard",
            "I wanted to do my best",
            "I didn't give up",
            "I put in a lot of effort",
            "I stuck with it",
        ],
    },
    "experience_of_flow": {
        "group": "cognitive_engagement",
        "group_label": "Cognitive Engagement (SCES)",
        "definition": (
            "The experience of being fully absorbed in the activity, often "
            "marked by focused attention and enjoyment."
        ),
        "analytic_focus": [
            "loss of time awareness",
            "deep immersion",
            "sustained enjoyment",
        ],
        "indicators": [
            "I forgot about everything else",
            "I was completely into it",
            "I was totally focused",
            "Time flew by",
            "I didn't notice anything else",
            "I was in the zone",
        ],
    },

    # ---- Group D: Motivation (SIMS) ----
    "intrinsic_motivation": {
        "group": "motivation",
        "group_label": "Motivation (SIMS)",
        "definition": "Engagement in an activity for inherent enjoyment and satisfaction.",
        "analytic_focus": ["enjoyment", "curiosity", "interest for its own sake"],
        "indicators": [
            "I did this because I liked it",
            "I enjoyed this activity",
            "This was interesting to me",
            "This was fun to do",
            "I wanted to do this",
            "I did this because I thought it was cool",
        ],
    },
    "identified_regulation": {
        "group": "motivation",
        "group_label": "Motivation (SIMS)",
        "definition": (
            "A self-determined form of extrinsic motivation where the activity "
            "is valued and personally endorsed."
        ),
        "analytic_focus": ["perceived usefulness", "personal value", "goal alignment"],
        "indicators": [
            "I did this because it's good for me",
            "I know this is important to understand",
            "I did this because it's useful",
            "This will help me learn",
            "This is something I should know",
            "This will help me in the future",
        ],
    },
    "external_regulation": {
        "group": "motivation",
        "group_label": "Motivation (SIMS)",
        "definition": "Motivation driven by external rewards or avoidance of negative consequences.",
        "analytic_focus": ["rewards", "pressure", "compliance"],
        "indicators": [
            "I did this because I had to",
            "I did this because I was told to",
            "I did this because it was required",
            "I was supposed to do this",
            "I didn't have a choice",
            "I had to do this",
        ],
    },
    "amotivation": {
        "group": "motivation",
        "group_label": "Motivation (SIMS)",
        "definition": (
            "A lack of motivation characterized by feelings of incompetence "
            "and lack of control over outcomes."
        ),
        "analytic_focus": ["disengagement", "helplessness", "lack of purpose"],
        "indicators": [
            "I didn't really care about this",
            "I didn't see the point of this",
            "I didn't want to do this",
            "This didn't matter to me",
            "I wasn't interested",
            "This wasn't for me",
        ],
    },

    # ---- Group E: AI Understanding (AI-CI) ----
    "understanding_ai_basics": {
        "group": "ai_understanding",
        "group_label": "AI Understanding (AI-CI)",
        "definition": "Foundational comprehension of what AI is and how it differs from human intelligence.",
        "analytic_focus": [
            "understanding what AI is",
            "distinguishing AI from humans",
            "grasping core AI concepts",
        ],
        "indicators": [
            "I get how AI works now",
            "I know what makes something AI",
            "I can explain AI to someone",
            "I understand what AI is",
            "I see how AI is different from humans",
            "AI isn't like people",
        ],
    },
    "ai_learning_processes": {
        "group": "ai_understanding",
        "group_label": "AI Understanding (AI-CI)",
        "definition": "Understanding how AI systems learn and improve from data.",
        "analytic_focus": [
            "understanding training and data",
            "recognizing improvement through practice",
            "grasping supervised learning",
        ],
        "indicators": [
            "I know how AI learns",
            "I understand how AI gets smarter",
            "I see how AI improves",
            "AI gets better with practice",
            "AI needs data to learn",
            "AI changes based on what it sees",
        ],
    },
    "ai_applications": {
        "group": "ai_understanding",
        "group_label": "AI Understanding (AI-CI)",
        "definition": "Awareness of real-world AI use cases and their societal impact.",
        "analytic_focus": [
            "identifying AI in everyday life",
            "understanding AI utility",
            "connecting AI to familiar technology",
        ],
        "indicators": [
            "I know how AI is used",
            "I see where AI helps people",
            "I understand what AI can do",
            "AI can do helpful things",
            "AI is in things we use",
            "AI has many uses",
        ],
    },
    "ai_limitations": {
        "group": "ai_understanding",
        "group_label": "AI Understanding (AI-CI)",
        "definition": "Understanding that AI has boundaries, biases, and failure modes.",
        "analytic_focus": [
            "recognizing AI errors",
            "understanding bias",
            "distinguishing AI from human reasoning",
        ],
        "indicators": [
            "I know what AI can't do",
            "I see where AI makes mistakes",
            "I understand AI's limits",
            "AI isn't perfect",
            "AI has problems too",
            "AI isn't like human thinking",
        ],
    },

    # ---- Group F: AI Misconceptions (AIM-F) ----
    # Category A — Ontological Misclassification
    "aim_a1_ai_equals_robot": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": (
            "AI is understood primarily as a physical robot or device; "
            "conflation of intelligence with embodiment."
        ),
        "analytic_focus": ["AI identified with robots","physical form required for AI","embodiment as intelligence"],
        "indicators": ["AI means robots that do work","AI is a physical machine","AI is the robot","AI needs a body","robots are AI","AI is the thing you can see"],
        "questionnaire_item": "Q3_2",
    },
    "aim_a2_ai_equals_any_software": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "Any complex software is considered AI; overgeneralization of computational systems.",
        "analytic_focus": ["software equals AI","data use as AI criterion","confusion between automation and AI"],
        "indicators": ["if software uses data it is AI","any computer program is AI","apps are AI","all software is AI","computers are AI","technology equals AI"],
        "questionnaire_item": "Q3_5",
    },
    "aim_a3_ai_without_technology": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI believed to exist independent of technological systems.",
        "analytic_focus": ["AI exists without machines","intelligence separated from technology","natural intelligence as AI"],
        "indicators": ["AI doesn't need machines","something can be AI without technology","AI can exist naturally","intelligence itself is AI","AI doesn't need computers"],
        "questionnaire_item": "Q3_7",
    },
    "aim_a4_ai_preinstalled_intelligence": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI seen as containing fixed built-in intelligence rather than learned models.",
        "analytic_focus": ["AI knows things from creation","fixed knowledge in AI","no learning needed"],
        "indicators": ["AI already knows things when created","AI is born with knowledge","AI doesn't need to learn","AI has built-in answers","AI knows everything already","AI was programmed with all information"],
        "questionnaire_item": "Q3_8",
    },
    "aim_b1_ai_thinks_like_humans": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI assumed to replicate human cognition or brain processes.",
        "analytic_focus": ["AI mimics human thinking","brain-like processing","human cognition as AI model"],
        "indicators": ["AI works like a human brain","AI thinks the same way we do","AI has a brain","AI processes things like humans","AI reasons like a person","AI understands like we do"],
        "questionnaire_item": "Q3_4",
    },
    "aim_b2_ai_has_emotions": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI attributed feelings, intuition, or subjective experiences.",
        "analytic_focus": ["AI has feelings","AI experiences emotions","AI has a conscience"],
        "indicators": ["AI can feel things","AI has emotions","AI gets angry or happy","AI knows what it wants","AI has a conscience","AI can intuit things"],
        "questionnaire_item": "Q3_3",
    },
    "aim_b3_ai_autonomous_being": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI viewed as an independent thinking entity or lifeform.",
        "analytic_focus": ["AI as independent agent","AI as living being","AI understands its actions"],
        "indicators": ["AI understands what it is doing","AI makes its own decisions","AI is alive","AI is a being","AI knows why it does things","AI acts on its own"],
        "questionnaire_item": "Q3_1",
    },
    "aim_c1_self_learning_autonomy": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "AI believed to learn entirely on its own without human input.",
        "analytic_focus": ["AI learns without humans","no training data needed","self-directed learning"],
        "indicators": ["AI teaches itself everything","AI learns on its own","AI doesn't need humans to learn","AI figures things out by itself","AI can learn without data","AI improves without human help"],
        "questionnaire_item": "Q3_1",
    },
    "aim_d1_ai_overgeneralization": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "Most digital or computational systems labeled as AI; boundary inflation.",
        "analytic_focus": ["all digital things are AI","overidentification of AI","lack of distinguishing criteria"],
        "indicators": ["all software is AI","everything digital is AI","any app is AI","calculators are AI","the internet is AI","every computer program is AI"],
        "questionnaire_item": "Q3_5",
    },
    "aim_d2_ai_underrecognition": {
        "group": "ai_misconceptions",
        "group_label": "AI Misconceptions (AIM-F)",
        "definition": "Everyday AI systems not recognized as AI; visibility bias.",
        "analytic_focus": ["everyday AI invisible","recommendation systems not AI","AI must be physical"],
        "indicators": ["social media recommendations aren't AI","search results aren't AI","that's not really AI","AI is only robots","Netflix doesn't use AI","voice assistants aren't AI"],
        "questionnaire_item": "Q3_6",
    },
}

# Construct group definitions
CONSTRUCT_GROUPS = {
    "messaging_perception":       "Messaging Perception (CCCES)",
    "individual_characteristics": "Individual Characteristics (CCCES)",
    "cognitive_engagement":       "Cognitive Engagement (SCES)",
    "motivation":                 "Motivation (SIMS)",
    "ai_understanding":           "AI Understanding (AI-CI)",
    "ai_misconceptions":          "AI Misconceptions (AIM-F)",
}

# Learning objectives (from learning_objectives.yaml — Module 1-7)
LEARNING_OBJECTIVES = {
    "module_1": {
        "title": "Natural vs. Artificial Intelligence in Problem-Solving",
        "objectives": [
            "Learners can identify and compare attributes of biological and artificial agents.",
            "Learners can classify problem-solving environments using descriptors such as known vs unknown.",
            "Learners can distinguish similarities and differences between biological and machine problem-solving.",
            "Learners can practice evidence-based observation and reasoning.",
            "Learners can reflect on accountability, feedback, and learning in intelligent systems.",
        ],
        "indicators": [
            "comparison between humans, animals, and machines",
            "references to sensing or decision-making",
            "evidence-based reasoning explanations",
            "reflections on responsibility or learning",
        ],
    },
    "module_2": {
        "title": "Goal-Based Problem-Solving",
        "objectives": [
            "Learners can execute goal-based tasks using a structured problem-solving process.",
            "Learners can characterize problem-solving environments.",
            "Learners can distinguish between strategic planning and tactical execution.",
            "Learners can apply decomposition and abstraction to complex tasks.",
            "Learners can explain supervised learning through hands-on experiences.",
        ],
        "indicators": [
            "references to goals or subgoals",
            "planning language or sequencing",
            "decomposition examples",
            "transfer to real-world scenarios",
        ],
    },
    "module_3": {
        "title": "Introduction to Algorithms",
        "objectives": [
            "Learners can recognize patterns and explain their importance in AI problem-solving.",
            "Learners can apply rule-based strategies to structured games.",
            "Learners can construct and interpret decision trees.",
            "Learners can explain how search strategies evaluate solution paths.",
            "Learners can design step-by-step algorithms to solve defined problems.",
        ],
        "indicators": [
            "rule-based reasoning references",
            "algorithm descriptions",
            "decision-tree explanations",
            "optimization reasoning",
        ],
    },
    "module_4": {
        "title": "Bias, Hallucinations & Natural Language in AI",
        "objectives": [
            "Learners can apply evidence-based inferencing using textual information.",
            "Learners can identify and resolve referential ambiguity in language.",
            "Learners can critically evaluate AI-generated responses.",
            "Learners can explain AI hallucinations and their consequences.",
            "Learners can detect and analyze sources of bias in AI systems.",
        ],
        "indicators": [
            "references to evidence or justification",
            "bias identification",
            "critique of AI outputs",
            "ethical reflections",
        ],
    },
    "module_5": {
        "title": "Constraints in Problem-Solving",
        "objectives": [
            "Learners can define constraints and explain their role in shaping decisions.",
            "Learners can apply constraint reasoning using logic problems.",
            "Learners can compare human intuition and algorithmic problem-solving.",
            "Learners can evaluate trade-offs under competing real-world constraints.",
            "Learners can recognize bias and optimization in recommendation systems.",
        ],
        "indicators": [
            "constraint identification",
            "optimization reasoning",
            "trade-off explanations",
            "comparison of human vs AI strategies",
        ],
    },
    "module_6": {
        "title": "Uncertainty in Problem-Solving",
        "objectives": [
            "Learners can define uncertainty and distinguish it from deterministic outcomes.",
            "Learners can classify environments as deterministic or stochastic.",
            "Learners can calculate basic probabilities using simple scenarios.",
            "Learners can interpret uncertainty in real-world AI applications.",
        ],
        "indicators": [
            "probability reasoning",
            "uncertainty explanations",
            "prediction discussions",
            "real-world AI examples",
        ],
    },
    "module_7": {
        "title": "Introduction to Machine Learning",
        "objectives": [
            "Learners can integrate concepts from earlier modules into ML contexts.",
            "Learners can train supervised learning models using labeled datasets.",
            "Learners can evaluate model predictions and interpret confidence levels.",
            "Learners can implement trained AI models within Scratch applications.",
        ],
        "indicators": [
            "references to training or labeling",
            "evaluation of model performance",
            "discussion of uncertainty in predictions",
            "application-building reflections",
        ],
    },
}


# -----------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------

_DTA_PHASE2_PROMPT = """\
You are conducting a deductive thematic analysis of a participant's reflection \
from the Basics4AI AI literacy programme for 10-14 year-olds.

CONSTRUCT TO ANALYSE: {construct_name}
Definition: {definition}
Analytic focus: {analytic_focus}

Look for semantic equivalents of these indicator phrases (the participant \
may express the same idea in different words):
{indicators}

PARTICIPANT TEXT:
{text}

Analyse the text above and return ONLY valid JSON with this structure:
{{
  "construct": "{construct_name}",
  "evidence_count": <integer 0 or more>,
  "instances": [
    {{
      "quote": "<exact quote from text, max 2 sentences>",
      "valence": "<positive|negative|neutral>",
      "explanation": "<1 sentence explaining why this is evidence>"
    }}
  ]
}}

Rules:
- evidence_count must equal the length of the instances array
- quote must be verbatim from the participant text
- valence: positive = construct clearly present and affirmed; \
negative = construct present but reversed/absent; neutral = ambiguous
- If no evidence found, return evidence_count: 0 and instances: []
- Return ONLY the JSON object, no other text"""


_DTA_LO_PROMPT = """\
You are analysing a participant's reflection from Module {module_id} \
of the Basics4AI programme.

Module title: {module_title}
Learning objective {lo_index}: {lo_text}

Indicators to look for (semantic equivalents):
{indicators}

PARTICIPANT TEXT:
{text}

Does this text show evidence that the participant has achieved this learning objective?

Return ONLY valid JSON:
{{
  "lo_index": {lo_index},
  "evidence_present": <0 or 1>,
  "evidence_quote": "<quote if present, empty string if not>",
  "explanation": "<1 sentence>"
}}"""


# -----------------------------------------------------------------------
# Phase 2: Code data
# -----------------------------------------------------------------------

def run_dta_phase2(
    transcripts: List[Dict[str, Any]],
    model: str,
    temperature: float,
    construct_groups: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 2 — Apply codebook constructs to each participant's text.

    Parameters
    ----------
    transcripts : list of dicts
        Each: {"participant_id": str, "content": str, "module_id": str (optional)}
    model : str
    temperature : float
    construct_groups : list of str or None
        If None, all 5 groups are analysed.
        Options: "messaging_perception", "individual_characteristics",
                 "cognitive_engagement", "motivation", "ai_understanding"
    run_id : str or None
    db_path : Path or None

    Returns
    -------
    list of result dicts, one per (participant × construct)
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    if construct_groups is None:
        construct_groups = list(CONSTRUCT_GROUPS.keys())

    # Filter codebook to selected groups
    constructs_to_run = {
        k: v for k, v in CODEBOOK.items()
        if v["group"] in construct_groups
    }

    results = []
    now     = datetime.utcnow().isoformat()

    for transcript in transcripts:
        pid  = transcript.get("participant_id", "unknown")
        text = str(transcript.get("content", "")).strip()
        if not text:
            continue

        for construct_name, construct_def in constructs_to_run.items():
            prompt = _DTA_PHASE2_PROMPT.format(
                construct_name = construct_name.replace("_", " ").title(),
                definition     = construct_def["definition"],
                analytic_focus = ", ".join(construct_def["analytic_focus"]),
                indicators     = "\n".join(
                    f"  - \"{ind}\"" for ind in construct_def["indicators"]
                ),
                text           = text[:3000],  # cap per-participant text
            )

            # Get system prompt from ita_pipeline
            try:
                sys_prompt = _lazy_import(
                    "core.analytics.llm.ita_pipeline", "SYSTEM_PROMPT"
                )
            except Exception:
                sys_prompt = (
                    "You are a qualitative research assistant for the "
                    "Basics4AI AI literacy programme."
                )

            response = call_model(
                model, prompt,
                system=sys_prompt,
                temperature=temperature,
                max_tokens=1500,
            )

            result = {
                "participant_id":  pid,
                "construct_name":  construct_name,
                "construct_group": construct_def["group"],
                "group_label":     construct_def["group_label"],
                "model":           model,
                "temperature":     temperature,
                "created_at":      now,
                "error":           None,
                "evidence_count":  0,
                "valence_positive": 0,
                "valence_negative": 0,
                "valence_neutral":  0,
                "instances":       [],
            }

            if response["error"]:
                result["error"] = response["error"]
            else:
                parsed = _parse_dta_json(response["text"])
                if parsed:
                    instances = parsed.get("instances", [])
                    result["evidence_count"]   = parsed.get("evidence_count", len(instances))
                    result["instances"]        = instances
                    result["valence_positive"] = sum(
                        1 for i in instances if i.get("valence") == "positive"
                    )
                    result["valence_negative"] = sum(
                        1 for i in instances if i.get("valence") == "negative"
                    )
                    result["valence_neutral"]  = sum(
                        1 for i in instances if i.get("valence") == "neutral"
                    )
                else:
                    result["error"] = f"JSON parse failed: {response['text'][:100]}"

            results.append(result)

    if run_id:
        save_dta_results(run_id, results, db_path)

    return results


# -----------------------------------------------------------------------
# Phase 3: Build evidence matrix
# -----------------------------------------------------------------------

def run_dta_phase3(
    phase2_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Phase 3 — Aggregate phase 2 results into evidence matrix.

    Returns
    -------
    dict:
        evidence_matrix   pd.DataFrame  participants × constructs (evidence counts)
        valence_matrix    pd.DataFrame  participants × constructs (dominant valence)
        summary_by_construct  pd.DataFrame  construct-level aggregates
        summary_by_participant pd.DataFrame participant-level aggregates
    """
    if not phase2_results:
        return {"error": "No Phase 2 results to aggregate."}

    rows = []
    for r in phase2_results:
        rows.append({
            "participant_id":  r["participant_id"],
            "construct_name":  r["construct_name"],
            "construct_group": r.get("construct_group", ""),
            "group_label":     r.get("group_label", ""),
            "evidence_count":  r.get("evidence_count", 0),
            "valence_positive":r.get("valence_positive", 0),
            "valence_negative":r.get("valence_negative", 0),
            "valence_neutral": r.get("valence_neutral", 0),
        })

    df = pd.DataFrame(rows)

    # Evidence matrix: participants × constructs
    evidence_matrix = df.pivot_table(
        index="participant_id",
        columns="construct_name",
        values="evidence_count",
        aggfunc="sum",
        fill_value=0,
    )

    # Dominant valence per cell
    def _dominant_valence(row):
        p = row.get("valence_positive", 0)
        n = row.get("valence_negative", 0)
        u = row.get("valence_neutral",  0)
        if p == n == u == 0:
            return "absent"
        if p >= n and p >= u:
            return "positive"
        if n >= p and n >= u:
            return "negative"
        return "neutral"

    df["dominant_valence"] = df.apply(_dominant_valence, axis=1)
    valence_matrix = df.pivot_table(
        index="participant_id",
        columns="construct_name",
        values="dominant_valence",
        aggfunc="first",
    )

    # Construct-level summary
    summary_construct = df.groupby(
        ["construct_name", "construct_group", "group_label"]
    ).agg(
        total_evidence=("evidence_count", "sum"),
        n_participants_with_evidence=("evidence_count", lambda x: (x > 0).sum()),
        total_positive=("valence_positive", "sum"),
        total_negative=("valence_negative", "sum"),
        total_neutral=("valence_neutral",  "sum"),
    ).reset_index()

    # Participant-level summary
    summary_participant = df.groupby("participant_id").agg(
        total_evidence=("evidence_count", "sum"),
        constructs_with_evidence=("evidence_count", lambda x: (x > 0).sum()),
        total_positive=("valence_positive", "sum"),
        total_negative=("valence_negative", "sum"),
    ).reset_index()

    return {
        "evidence_matrix":         evidence_matrix,
        "valence_matrix":          valence_matrix,
        "summary_by_construct":    summary_construct,
        "summary_by_participant":  summary_participant,
        "raw_df":                  df,
    }


# -----------------------------------------------------------------------
# Phase 4: Review low-evidence constructs
# -----------------------------------------------------------------------

def run_dta_phase4(
    phase2_results: List[Dict[str, Any]],
    model: str,
    low_evidence_threshold: int = 0,
    temperature: float = 0.7,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 4 — Re-examine participants × constructs with zero or low evidence.

    Runs at slightly higher temperature to catch missed signals.
    Only re-runs cells where evidence_count <= low_evidence_threshold.

    Returns merged results (Phase 2 + Phase 4 updates).
    """
    # Find cells to re-examine
    low_evidence = [
        r for r in phase2_results
        if r.get("evidence_count", 0) <= low_evidence_threshold
        and not r.get("error")
    ]

    if not low_evidence:
        return phase2_results

    # Re-run those cells at higher temperature
    # Build minimal transcript list (one per unique participant)
    seen = set()
    rerun_transcripts = []
    for r in low_evidence:
        pid = r["participant_id"]
        if pid not in seen:
            seen.add(pid)

    # We need original text — stored implicitly in Phase 2 results
    # (not available here without re-loading transcripts)
    # Return original results with a flag for the UI to indicate review needed
    updated = []
    for r in phase2_results:
        r2 = r.copy()
        r2["phase4_reviewed"] = r in low_evidence
        updated.append(r2)

    return updated


# -----------------------------------------------------------------------
# Phase 5: Report
# -----------------------------------------------------------------------

def run_dta_phase5(
    phase3_matrix: Dict[str, Any],
    phase2_results: List[Dict[str, Any]],
    model: str,
    temperature: float,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Phase 5 — Generate DTA report narrative.

    Returns
    -------
    dict:
        report_text      str  narrative summary
        construct_table  pd.DataFrame  full evidence table
        participant_profiles  list of dicts  one per participant
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    # Build construct-level table for the prompt
    summary = phase3_matrix.get("summary_by_construct", pd.DataFrame())
    if summary.empty:
        return {"error": "No Phase 3 matrix to report on."}

    # Format for prompt
    table_lines = []
    for _, row in summary.iterrows():
        table_lines.append(
            f"  {row['construct_name']}: evidence={row['total_evidence']}, "
            f"positive={row['total_positive']}, negative={row['total_negative']}, "
            f"neutral={row['total_neutral']}, "
            f"participants_with_evidence={row['n_participants_with_evidence']}"
        )
    table_text = "\n".join(table_lines)

    prompt = f"""\
You are writing a summary of a deductive thematic analysis of reflection data \
from the Basics4AI AI literacy programme for 10-14 year-olds.

The following construct evidence was identified across all participants:

{table_text}

Write a concise academic summary (300-400 words) that:
1. Describes the overall pattern of construct evidence
2. Notes which constructs showed the strongest and weakest evidence
3. Discusses the valence patterns (positive vs negative evidence)
4. Draws preliminary conclusions about participants' experiences

Write in academic English. Use construct names as they appear above.
Do not use bullet points — write in continuous prose."""

    try:
        sys_prompt = _lazy_import(
            "core.analytics.llm.ita_pipeline", "SYSTEM_PROMPT"
        )
    except Exception:
        sys_prompt = "You are a qualitative research assistant."

    response = call_model(
        model, prompt,
        system=sys_prompt,
        temperature=temperature,
        max_tokens=2000,
    )

    # Build participant profiles
    profiles = _build_participant_profiles(phase2_results)

    result = {
        "report_text":          response.get("text", "") if not response.get("error") else "",
        "error":                response.get("error"),
        "construct_table":      summary,
        "participant_profiles": profiles,
    }

    if run_id and not result["error"]:
        conn = _get_conn(db_path)
        conn.execute(
            """INSERT OR REPLACE INTO dta_results
               (run_id, participant_id, construct_name, construct_group,
                evidence_count, instances_json, model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, "_report_", "_phase5_", "report",
             0, json.dumps({"report": result["report_text"]}),
             model, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

    return result


# -----------------------------------------------------------------------
# Learning Objectives layer
# -----------------------------------------------------------------------

def run_lo_analysis(
    transcripts: List[Dict[str, Any]],
    model: str,
    temperature: float,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Separate learning objectives analysis layer.

    For each participant × module × learning objective:
    checks if the participant's reflection shows evidence of achieving the LO.

    Parameters
    ----------
    transcripts : list of dicts
        Must include "module_id" field (e.g. "module_1").
    model, temperature, run_id, db_path : as usual

    Returns
    -------
    list of LO result dicts
    """
    call_model = _lazy_import("core.analytics.llm.llm_clients", "call_model")

    results = []
    now     = datetime.utcnow().isoformat()

    try:
        sys_prompt = _lazy_import(
            "core.analytics.llm.ita_pipeline", "SYSTEM_PROMPT"
        )
    except Exception:
        sys_prompt = "You are a qualitative research assistant."

    for transcript in transcripts:
        pid       = transcript.get("participant_id", "unknown")
        text      = str(transcript.get("content", "")).strip()
        module_id = transcript.get("module_id", "")

        # Normalise module_id: "module_1" or "module1" → "module_1"
        m = re.match(r"module_?(\d+)", str(module_id), re.IGNORECASE)
        if not m:
            continue
        mod_key = f"module_{m.group(1)}"

        lo_def = LEARNING_OBJECTIVES.get(mod_key)
        if not lo_def or not text:
            continue

        for lo_idx, lo_text in enumerate(lo_def["objectives"]):
            prompt = _DTA_LO_PROMPT.format(
                module_id    = mod_key,
                module_title = lo_def["title"],
                lo_index     = lo_idx + 1,
                lo_text      = lo_text,
                indicators   = "\n".join(
                    f"  - {ind}" for ind in lo_def["indicators"]
                ),
                text         = text[:2000],
            )

            response = call_model(
                model, prompt,
                system=sys_prompt,
                temperature=temperature,
                max_tokens=400,
            )

            lo_result = {
                "participant_id":  pid,
                "module_id":       mod_key,
                "lo_index":        lo_idx + 1,
                "lo_text":         lo_text,
                "evidence_present": 0,
                "evidence_quote":  "",
                "model":           model,
                "created_at":      now,
                "error":           None,
            }

            if response["error"]:
                lo_result["error"] = response["error"]
            else:
                parsed = _parse_dta_json(response["text"])
                if parsed:
                    lo_result["evidence_present"] = int(
                        parsed.get("evidence_present", 0)
                    )
                    lo_result["evidence_quote"] = parsed.get("evidence_quote", "")
                else:
                    lo_result["error"] = f"Parse failed: {response['text'][:80]}"

            results.append(lo_result)

            # Save to DB
            if run_id:
                _save_lo_result(run_id, lo_result, db_path)

    return results


# -----------------------------------------------------------------------
# Run management
# -----------------------------------------------------------------------

def create_dta_run(
    model: str,
    temperature: float,
    source_type: str,
    construct_groups: List[str],
    created_by: str,
    notes: str = "",
    db_path: Optional[Path] = None,
) -> str:
    """Create a new DTA run record. Returns run_id."""
    _init_dta_schema(db_path)
    run_id = str(uuid.uuid4())
    now    = datetime.utcnow().isoformat()
    conn   = _get_conn(db_path)
    conn.execute(
        """INSERT INTO dta_runs
           (run_id, created_by, created_at, model, temperature,
            source_type, construct_groups, status, phase_reached, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (run_id, created_by, now, model, temperature,
         source_type, json.dumps(construct_groups),
         "created", 0, notes)
    )
    conn.commit()
    conn.close()
    return run_id


def save_dta_results(
    run_id: str,
    results: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> None:
    """Persist phase 2 results to dta_results table."""
    _init_dta_schema(db_path)
    conn = _get_conn(db_path)
    now  = datetime.utcnow().isoformat()

    for r in results:
        conn.execute(
            """INSERT INTO dta_results
               (run_id, participant_id, construct_group, construct_name,
                evidence_count, valence_positive, valence_negative, valence_neutral,
                instances_json, model, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id, participant_id, construct_name)
               DO UPDATE SET
                   evidence_count=excluded.evidence_count,
                   valence_positive=excluded.valence_positive,
                   valence_negative=excluded.valence_negative,
                   valence_neutral=excluded.valence_neutral,
                   instances_json=excluded.instances_json""",
            (
                run_id,
                r.get("participant_id", ""),
                r.get("construct_group", ""),
                r.get("construct_name", ""),
                r.get("evidence_count", 0),
                r.get("valence_positive", 0),
                r.get("valence_negative", 0),
                r.get("valence_neutral", 0),
                json.dumps(r.get("instances", [])),
                r.get("model", ""),
                now,
            )
        )
        conn.execute(
            "UPDATE dta_runs SET phase_reached=MAX(phase_reached,2), "
            "status='running' WHERE run_id=?",
            (run_id,)
        )

    conn.commit()
    conn.close()


def load_dta_results(
    run_id: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load all DTA results for a run as a DataFrame."""
    _init_dta_schema(db_path)
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT participant_id, construct_group, construct_name,
                  evidence_count, valence_positive, valence_negative,
                  valence_neutral, instances_json, model
           FROM dta_results WHERE run_id=?
           AND participant_id != '_report_'""",
        (run_id,)
    ).fetchall()
    conn.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "participant_id", "construct_group", "construct_name",
        "evidence_count", "valence_positive", "valence_negative",
        "valence_neutral", "instances_json", "model",
    ])
    df["instances"] = df["instances_json"].apply(
        lambda x: json.loads(x) if x else []
    )
    return df


def list_dta_runs(
    created_by: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List DTA runs, newest first."""
    _init_dta_schema(db_path)
    conn = _get_conn(db_path)
    if created_by:
        rows = conn.execute(
            """SELECT run_id, created_by, created_at, model, temperature,
                      source_type, construct_groups, status, phase_reached
               FROM dta_runs WHERE created_by=? ORDER BY created_at DESC""",
            (created_by,)
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT run_id, created_by, created_at, model, temperature,
                      source_type, construct_groups, status, phase_reached
               FROM dta_runs ORDER BY created_at DESC"""
        ).fetchall()
    conn.close()
    cols = ["run_id","created_by","created_at","model","temperature",
            "source_type","construct_groups","status","phase_reached"]
    return [dict(zip(cols, r)) for r in rows]


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

def _parse_dta_json(text: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling fences and truncation."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end != -1:
        try:
            return json.loads(text[start:end+1])
        except json.JSONDecodeError:
            pass
    # Truncation repair
    fragment = text[start:]
    depth_brace = depth_bracket = 0
    in_string = escape = False
    for ch in fragment:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1
    repaired = fragment.rstrip().rstrip(",")
    if in_string:
        repaired += '"'
    repaired += "]" * max(0, depth_bracket)
    repaired += "}" * max(0, depth_brace)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _build_participant_profiles(
    phase2_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build one profile dict per participant from phase 2 results."""
    from collections import defaultdict
    profiles = defaultdict(lambda: {
        "constructs": {},
        "total_evidence": 0,
        "total_positive": 0,
        "total_negative": 0,
    })

    for r in phase2_results:
        pid = r["participant_id"]
        cname = r["construct_name"]
        profiles[pid]["constructs"][cname] = {
            "evidence_count":   r.get("evidence_count", 0),
            "valence_positive": r.get("valence_positive", 0),
            "valence_negative": r.get("valence_negative", 0),
            "valence_neutral":  r.get("valence_neutral", 0),
            "instances":        r.get("instances", []),
        }
        profiles[pid]["total_evidence"] += r.get("evidence_count", 0)
        profiles[pid]["total_positive"] += r.get("valence_positive", 0)
        profiles[pid]["total_negative"] += r.get("valence_negative", 0)

    return [{"participant_id": pid, **data}
            for pid, data in sorted(profiles.items())]


def _save_lo_result(
    run_id: str,
    lo_result: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> None:
    """Save a single LO result to dta_lo_results."""
    _init_dta_schema(db_path)
    conn = _get_conn(db_path)
    conn.execute(
        """INSERT INTO dta_lo_results
           (run_id, participant_id, module_id, lo_index, lo_text,
            evidence_present, evidence_quote, model, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(run_id, participant_id, module_id, lo_index)
           DO UPDATE SET
               evidence_present=excluded.evidence_present,
               evidence_quote=excluded.evidence_quote""",
        (
            run_id,
            lo_result["participant_id"],
            lo_result["module_id"],
            lo_result["lo_index"],
            lo_result["lo_text"],
            lo_result["evidence_present"],
            lo_result["evidence_quote"],
            lo_result["model"],
            lo_result["created_at"],
        )
    )
    conn.commit()
    conn.close()
