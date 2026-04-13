
"""
Part 1: Sliding Window + Knowledge Pyramid
"""

import re
import math
import string
from collections import Counter
from typing import List, Dict

CHARS_PER_PAGE    = 2500
WINDOW_SIZE_PAGES = 2
WINDOW_OVERLAP    = 0.25
SUMMARY_SENTENCES = 2

CATEGORY_RULES = {
    "Mathematics": ["equation","formula","algebra","calculus","number","sum"],
    "Science":     ["experiment","hypothesis","biology","chemistry","physics","atom"],
    "Technology":  ["algorithm","software","network","data","computer","model","code"],
    "Legal":       ["contract","law","clause","liability","jurisdiction","statute"],
    "Finance":     ["revenue","profit","investment","balance","equity","stock"],
    "Healthcare":  ["patient","diagnosis","treatment","disease","medicine","clinical"],
    "General":     []
}

def load_document(source: str) -> str:
    try:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"[Loader] Loaded file: {source} ({len(text)} chars)")
        return text
    except (FileNotFoundError, OSError):
        print(f"[Loader] Using raw string input ({len(source)} chars)")
        return source

def sliding_window_chunks(text: str,
                          window_chars: int = CHARS_PER_PAGE * WINDOW_SIZE_PAGES,
                          overlap_ratio: float = WINDOW_OVERLAP) -> List[str]:
    step   = int(window_chars * (1 - overlap_ratio))
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + window_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    print(f"[Chunker] Created {len(chunks)} windows")
    return chunks

def extract_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def build_chunk_summary(text: str, n: int = SUMMARY_SENTENCES) -> str:
    sentences = extract_sentences(text)
    return " ".join(sentences[:n]) if sentences else text[:200]

def classify_category(text: str) -> str:
    text_lower = text.lower()
    scores     = {}
    for category, keywords in CATEGORY_RULES.items():
        if category == "General":
            continue
        scores[category] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get) if scores else "General"
    return best if scores.get(best, 0) > 0 else "General"

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    translator = str.maketrans("", "", string.punctuation)
    words      = text.lower().translate(translator).split()
    stop_words = {
        "the","a","an","is","it","in","on","at","to","of","and","or","but",
        "for","with","that","this","was","are","be","as","by","from","not",
        "have","has","had","we","i","you","he","she","they","which","what",
        "when","where","how","all"
    }
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    freq     = Counter(filtered)
    return [word for word, _ in freq.most_common(top_n)]

def mock_embedding(keywords: List[str]) -> List[float]:
    dim    = 16
    vector = [0.0] * dim
    for kw in keywords:
        for i, ch in enumerate(kw):
            vector[i % dim] += ord(ch) / 10000.0
    magnitude = math.sqrt(sum(v ** 2 for v in vector)) or 1.0
    return [v / magnitude for v in vector]

def build_knowledge_pyramid(chunk: str, chunk_id: int) -> Dict:
    keywords  = extract_keywords(chunk)
    embedding = mock_embedding(keywords)
    return {
        "chunk_id":          chunk_id,
        "layer1_raw_text":   chunk,
        "layer2_summary":    build_chunk_summary(chunk),
        "layer3_category":   classify_category(chunk),
        "layer4_keywords":   keywords,
        "layer4_embedding":  embedding,
    }

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a ** 2 for a in vec_a))
    mag_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def retrieve(query: str, index: List[Dict], top_k: int = 1) -> List[Dict]:
    query_keywords  = extract_keywords(query, top_n=10)
    query_embedding = mock_embedding(query_keywords)
    scored = []
    for pyramid in index:
        cos_score  = cosine_similarity(query_embedding, pyramid["layer4_embedding"])
        text_bonus = 0.0
        query_lower = query.lower()
        for word in query_lower.split():
            if word in pyramid["layer2_summary"].lower():
                text_bonus += 0.05
            if word in " ".join(pyramid["layer4_keywords"]):
                text_bonus += 0.03
        scored.append((cos_score + text_bonus, pyramid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]

def build_index(document_source: str) -> List[Dict]:
    print("\n=== INGESTION PIPELINE START ===")
    text   = load_document(document_source)
    chunks = sliding_window_chunks(text)
    index  = []
    for i, chunk in enumerate(chunks):
        pyramid = build_knowledge_pyramid(chunk, chunk_id=i)
        index.append(pyramid)
        print(f"  Chunk {i}: category='{pyramid['layer3_category']}', "
              f"keywords={pyramid['layer4_keywords'][:4]}")
    print(f"\n=== INDEX BUILT: {len(index)} entries ===\n")
    return index

def query_index(query: str, index: List[Dict]) -> None:
    print(f"\n QUERY: \"{query}\"")
    print("-" * 60)
    results = retrieve(query, index, top_k=1)
    if not results:
        print("No results found.")
        return
    score, best = results[0]
    print(f" Best Match (Chunk #{best['chunk_id']}, score={score:.4f})")
    print(f"\n[Layer 3] Category : {best['layer3_category']}")
    print(f"[Layer 4] Keywords : {', '.join(best['layer4_keywords'][:6])}")
    print(f"[Layer 2] Summary  : {best['layer2_summary'][:300]}")
    print(f"\n[Layer 1] Raw Text : {best['layer1_raw_text'][:300]}...")
    print("-" * 60)

# ── SAMPLE DOCUMENT ──────────────────────────────────────────────────────────
SAMPLE_DOCUMENT = """
Artificial Intelligence has transformed numerous industries over the past decade.
Machine learning algorithms can now recognize images, translate languages, and
generate creative content. Deep learning models, particularly transformer architectures,
have pushed the boundaries of natural language processing.

In healthcare, AI-driven diagnostic tools assist doctors in identifying diseases
from medical imaging with high accuracy. Clinical trials and drug discovery have been
accelerated through computational models. Patient outcomes improve when AI systems
flag anomalies in vital signs.

The financial sector leverages AI for fraud detection, algorithmic trading, and
credit scoring. Revenue prediction models and asset management platforms now
incorporate machine learning to provide better investment advice.

Legal technology firms deploy natural language models to review contracts, identify
liability clauses, and parse complex jurisdiction-specific statutes. Document review
for litigation that once took weeks now completes in hours with AI assistance.

Education technology uses adaptive learning algorithms to personalise curricula for
students. Code tutoring systems help developers by suggesting correct completions.
Assessments and grading can be partially automated, freeing educators for mentorship.
"""

if __name__ == "__main__":
    index = build_index(SAMPLE_DOCUMENT)

    queries = [
        "How is AI used in healthcare for patient diagnosis?",
        "What are machine learning applications in finance?",
        "Explain legal tech and contract review automation",
        "What ethical issues arise from AI in decision making?",
    ]
    for q in queries:
        query_index(q, index)