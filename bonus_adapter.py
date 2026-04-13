"""
Bonus: Reasoning-Aware Adapter with Query Routing
"""

import re
from enum import Enum
from typing import List, Dict

class QueryType(str, Enum):
    MATH    = "math"
    LEGAL   = "legal"
    SCIENCE = "science"
    CODE    = "code"
    GENERAL = "general"

CLASSIFIER_SIGNALS = {
    QueryType.MATH:    ["calculate","solve","equation","how many","total","sum",
                        "multiply","divide","percent","probability","algebra"],
    QueryType.LEGAL:   ["contract","law","statute","liability","clause",
                        "jurisdiction","regulation","court","legal","rights"],
    QueryType.SCIENCE: ["experiment","hypothesis","biology","chemistry","physics",
                        "molecule","atom","reaction","energy","quantum","dna"],
    QueryType.CODE:    ["code","function","algorithm","bug","debug","python",
                        "javascript","api","class","method","loop","error"],
}

def classify_query(query: str) -> QueryType:
    query_lower = query.lower()
    scores      = {qt: 0 for qt in QueryType}
    for query_type, signals in CLASSIFIER_SIGNALS.items():
        for signal in signals:
            try:
                if re.search(signal, query_lower):
                    scores[query_type] += 1
            except re.error:
                if signal in query_lower:
                    scores[query_type] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else QueryType.GENERAL


class MathReasoningModule:
    name = "MathReasoningModule"
    _SAFE = re.compile(r"[^0-9\+\-\*\/\.\(\)\s]")

    def process(self, query: str) -> dict:
        log    = ["[Math] Detected mathematical query."]
        answer = "Requires step-by-step chain-of-thought reasoning."
        conf   = 0.5
        expr_match = re.search(r"[\d\s\+\-\*\/\.\(\)]+", query)
        if expr_match:
            sanitised = self._SAFE.sub("", expr_match.group(0)).strip()
            if sanitised:
                try:
                    result = eval(sanitised, {"__builtins__": {}})
                    answer = f"Computed: {sanitised} = {result}"
                    conf   = 0.9
                    log.append(f"[Math] Expression: '{sanitised}' = {result}")
                except Exception as e:
                    log.append(f"[Math] Could not evaluate: {e}")
        log.append("[Math] Strategy: symbolic eval → chain-of-thought")
        return {"module": self.name, "log": log, "answer": answer, "confidence": conf}


class LegalReasoningModule:
    name     = "LegalReasoningModule"
    ENTITIES = ["contract","party","clause","statute","court",
                "jurisdiction","liability","plaintiff","defendant"]

    def process(self, query: str) -> dict:
        found = [e for e in self.ENTITIES if e in query.lower()]
        log   = [
            "[Legal] Detected legal query.",
            f"[Legal] Entities found: {found}",
            "[Legal] Strategy: structured retrieval + citation format",
        ]
        answer = (
            f"Legal concepts identified: {', '.join(found) or 'general legal matter'}. "
            "Retrieve relevant statutes, cross-reference jurisdiction, cite case law."
        )
        return {"module": self.name, "log": log, "answer": answer, "confidence": 0.7}


class ScienceReasoningModule:
    name    = "ScienceReasoningModule"
    DOMAINS = {
        "biology":   ["cell","dna","gene","evolution","organism"],
        "physics":   ["force","energy","velocity","quantum","particle"],
        "chemistry": ["molecule","atom","reaction","bond","element"],
    }

    def process(self, query: str) -> dict:
        q      = query.lower()
        domain = next((d for d, kws in self.DOMAINS.items() if any(k in q for k in kws)), "science")
        log    = [
            "[Science] Detected scientific query.",
            f"[Science] Sub-domain: {domain}",
            "[Science] Strategy: hypothesis → evidence → conclusion",
        ]
        answer = (
            f"This is a {domain} question. Identify the scientific principle, "
            "apply relevant model/equation, validate with empirical evidence."
        )
        return {"module": self.name, "log": log, "answer": answer, "confidence": 0.75}


class CodeReasoningModule:
    name  = "CodeReasoningModule"
    LANGS = {
        "Python":     ["python","pandas","numpy","flask"],
        "JavaScript": ["javascript","js","node","react"],
        "SQL":        ["sql","query","database","select"],
    }

    def process(self, query: str) -> dict:
        q    = query.lower()
        lang = next((l for l, kws in self.LANGS.items() if any(k in q for k in kws)), "general")
        log  = [
            "[Code] Detected programming query.",
            f"[Code] Language detected: {lang}",
            "[Code] Strategy: understand → write/fix → explain",
        ]
        answer = f"{lang} coding question. Write minimal reproducible code with inline comments."
        return {"module": self.name, "log": log, "answer": answer, "confidence": 0.8}


class GeneralReasoningModule:
    name = "GeneralReasoningModule"

    def process(self, query: str) -> dict:
        log    = ["[General] No specific domain.", "[General] Strategy: semantic search"]
        answer = "General knowledge query. Use semantic similarity search across knowledge base."
        return {"module": self.name, "log": log, "answer": answer, "confidence": 0.6}


class ReasoningRouter:
    def __init__(self):
        self._modules = {
            QueryType.MATH:    MathReasoningModule(),
            QueryType.LEGAL:   LegalReasoningModule(),
            QueryType.SCIENCE: ScienceReasoningModule(),
            QueryType.CODE:    CodeReasoningModule(),
            QueryType.GENERAL: GeneralReasoningModule(),
        }

    def route(self, query: str) -> dict:
        query_type = classify_query(query)
        module     = self._modules.get(query_type, self._modules[QueryType.GENERAL])
        result     = module.process(query)

        print(f"\n{'='*60}")
        print(f"  Query      : {query}")
        print(f"  Detected   : {query_type.value.upper()}")
        print(f"  Module     : {result['module']}")
        print(f"  Confidence : {result['confidence']:.0%}")
        print(f"{'─'*60}")
        for line in result["log"]:
            print(f"  {line}")
        print(f"{'─'*60}")
        print(f"  Answer     : {result['answer']}")
        print(f"{'='*60}")
        return result


if __name__ == "__main__":
    router  = ReasoningRouter()
    queries = [
        "If I have 15 apples and give away 6, how many are left?",
        "What are the liability clauses in an employment contract?",
        "How does CRISPR-Cas9 modify DNA in living organisms?",
        "How do I fix a KeyError in Python dictionary?",
        "What is the capital city of New Zealand?",
    ]
    for q in queries:
        router.route(q)
        
        