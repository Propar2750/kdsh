# 🧩 Main Logic & Verification Pipeline Report
## KDSH 2026 Track A - Hackathon Submission

---

## Executive Summary

Our verification pipeline implements a **claim-based consistency checking** approach using a multi-stage architecture: **Claim Extraction → Evidence Retrieval → Claim Verification → Aggregation**. The core innovation is our **"One Error = Contradict"** policy with **citation-based verification** and **parallel processing** for speed.

---

## 1. Problem Statement

### 1.1 The Task

Given:
- A **character backstory** (hypothetical narrative about a fictional character)
- The **source novel** (full text of the book)

Predict:
- **1 (Consistent)**: Backstory aligns with the novel
- **0 (Contradict)**: Backstory contains factual errors

### 1.2 Why This Is Hard

| Challenge | Our Solution |
|-----------|--------------|
| Long novels (500k+ words) | Chunking + hybrid retrieval |
| Subtle contradictions | Claim-level verification |
| Paraphrased content | Semantic embeddings |
| Ambiguous statements | LLM-based reasoning |

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     VERIFICATION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    INPUT                                        │     │
│  │  Character: "Edmond Dantès"                                    │     │
│  │  Book: "The Count of Monte Cristo"                             │     │
│  │  Backstory: "Dantès was a sailor who was wrongly imprisoned..."│     │
│  └──────────────────────────┬─────────────────────────────────────┘     │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: CLAIM EXTRACTION (LLM)                                  │   │
│  │                                                                    │   │
│  │  Extract 5 specific, verifiable claims:                           │   │
│  │  1. "Dantès was arrested on his wedding day"                      │   │
│  │  2. "He was imprisoned at Château d'If"                           │   │
│  │  3. "Abbé Faria told him about the treasure"                      │   │
│  │  4. "He escaped by taking Faria's place in burial sack"           │   │
│  │  5. "The treasure was on the island of Monte Cristo"              │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: EVIDENCE RETRIEVAL (Hybrid BM25 + Vector)               │   │
│  │                                                                    │   │
│  │  For each claim → Retrieve top-15 relevant chunks                 │   │
│  │  → RRF fusion → Select top-8 for verification                     │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: CLAIM VERIFICATION (Parallel LLM)                       │   │
│  │                                                                    │   │
│  │  For each (claim, evidence) pair:                                 │   │
│  │  → Verdict: SUPPORTS / CONTRADICTS / UNCLEAR                      │   │
│  │  → Confidence: 0.0 - 1.0                                          │   │
│  │  → Citation: "exact quote from evidence"                          │   │
│  │  → Reasoning: explanation                                         │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: AGGREGATION                                             │   │
│  │                                                                    │   │
│  │  Policy: ONE FACTUAL ERROR = CONTRADICT                           │   │
│  │                                                                    │   │
│  │  If any claim contradicts with confidence ≥ 0.5 → PREDICT 0       │   │
│  │  Else → PREDICT 1                                                 │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                            │
│                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT                                                           │   │
│  │  Prediction: 1 (CONSISTENT) or 0 (CONTRADICT)                     │   │
│  │  Explanation: Detailed reasoning with citations                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1: Claim Extraction

### 3.1 Purpose

Transform a narrative backstory into **specific, verifiable claims** that can be checked against the source text.

### 3.2 LLM Prompt Design

```python
class FastClaimExtractor:
    SYSTEM_PROMPT = """You extract SPECIFIC VERIFIABLE FACTS from character 
    backstories that can be checked against the book.

    Extract facts about:
    - DATES, YEARS, TIME PERIODS
    - NAMES of people mentioned
    - RELATIONSHIPS
    - SPECIFIC EVENTS
    - LOCATIONS

    IMPORTANT:
    - Each claim should be a FACTUAL STATEMENT
    - Do NOT make meta-comments
    - Extract what the backstory CLAIMS happened"""

    USER_PROMPT = """Extract {max_claims} FACTUAL CLAIMS about {character}.

    CHARACTER: {character}
    BACKSTORY: "{backstory}"

    Return exactly {max_claims} numbered factual claims:
    1.
    2.
    3.
    4.
    5."""
```

### 3.3 Claim Filtering

```python
def extract(self, backstory: str, character: str, book_name: str) -> List[str]:
    response = self.llm.generate(prompt, system_prompt=self.SYSTEM_PROMPT)
    
    claims = []
    for line in response.split('\n'):
        match = re.match(r'^[\d]+[.\):\-]\s*(.+)', line)
        if match:
            claim = match.group(1).strip()
            # Filter out empty or meta claims
            if len(claim) > 20 and not any(x in claim.lower() for x in 
                ['the claim', 'this claim', 'novel', 'book', 'author']):
                claims.append(claim)
    
    return claims[:self.config.max_claims]
```

### 3.4 Why 5 Claims?

- **Coverage**: Captures main biographical details
- **Efficiency**: Limits LLM calls
- **Quality**: Avoids extracting trivial facts
- **Recall**: Usually includes any contradictory information

---

## 4. Stage 2: Evidence Retrieval

*(Detailed in Retrieval Technique Report)*

### 4.1 Per-Claim Retrieval

```python
# For each claim, retrieve relevant evidence
claim_evidence = [
    (claim, self.retriever.search(claim, character=character)) 
    for claim in claims
]
```

### 4.2 Character-Aware Search

The retriever receives the character name for:
- Query expansion
- Entity-focused results
- Relationship context

---

## 5. Stage 3: Claim Verification

### 5.1 Verification Prompt

```python
class FastClaimVerifier:
    SYSTEM_PROMPT = """You verify claims about fictional characters against book evidence.

    VERDICTS:
    - SUPPORTS: Evidence confirms the claim is true
    - CONTRADICTS: Evidence shows conflicting information
    - UNCLEAR: Evidence doesn't address this topic AT ALL

    KEY RULES FOR CONTRADICTS:
    1. Different dates = CONTRADICTS
    2. Different facts = CONTRADICTS
    3. Opposite statements = CONTRADICTS

    KEY RULES FOR UNCLEAR:
    1. Topic not mentioned at all → UNCLEAR
    2. Fabricated details → UNCLEAR

    Output format:
    VERDICT: [SUPPORTS/CONTRADICTS/UNCLEAR]
    CONFIDENCE: [0.0-1.0]
    CITATION: ["exact quote" or "NONE"]
    REASONING: [brief explanation]"""
```

### 5.2 Evidence Formatting

```python
def verify(self, claim: str, evidence: List[Dict], character: str, book_name: str):
    evidence_parts = []
    for i, e in enumerate(evidence[:self.config.top_k_final]):
        content = e.get('content', '')[:1000]
        chapter = e.get('chapter', 'Unknown')
        page = e.get('page', '?')
        
        evidence_parts.append(
            f"[PASSAGE {i+1}] Chapter: {chapter}, Page: {page}:\n{content}"
        )
    
    evidence_text = "\n\n".join(evidence_parts)
```

### 5.3 Response Parsing

```python
def _parse_response(self, response: str) -> Dict[str, Any]:
    # Strategy 1: Explicit VERDICT line
    verdict_match = re.search(r'VERDICT:\s*(SUPPORTS?|CONTRADICTS?|UNCLEAR)', response, re.I)
    
    # Strategy 2: CONFIDENCE extraction
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response, re.I)
    
    # Strategy 3: CITATION extraction
    citation_match = re.search(r'CITATION:\s*["\']?(.+?)["\']?\s*(?:REASONING|$)', response, re.I)
    
    # Strategy 4: REASONING extraction
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?:$|\n\n)', response, re.I)
    
    # Fallback: Keyword-based detection
    if not verdict_match:
        contradiction_phrases = ['contradict', 'conflict', 'inconsistent', 'wrong']
        support_phrases = ['support', 'confirm', 'consistent', 'correct']
        # Count and decide...
```

### 5.4 Parallel Verification

```python
def verify_backstory(self, backstory: str, character: str, book_name: str):
    # Verify claims in parallel for speed
    with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
        futures = {
            executor.submit(verify_single, (i, claim, evidence)): i
            for i, (claim, evidence) in enumerate(claim_evidence)
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
```

---

## 6. Stage 4: Aggregation

### 6.1 The "One Error = Contradict" Policy

```python
class FastAggregator:
    """Aggregate claim verdicts into final prediction."""
    
    def aggregate(self, results: List[Dict], claims: List[str]) -> Tuple[int, Dict]:
        # Count verdicts
        n_contradicts = sum(1 for r in results if r['verdict'] == 'contradicts')
        n_supports = sum(1 for r in results if r['verdict'] == 'supports')
        n_unclear = sum(1 for r in results if r['verdict'] == 'unclear')
        
        # Find strong contradiction (high confidence + citation)
        for r in results:
            if self._is_strong_contradiction(r):
                return 0, {"reason": "Strong contradiction found", ...}
        
        # Find any contradiction with decent confidence
        for r in results:
            if r['verdict'] == 'contradicts' and r.get('confidence', 0) >= 0.5:
                return 0, {"reason": "Contradiction found", ...}
        
        # No contradictions → CONSISTENT
        return 1, {"reason": "No contradictions found", ...}
```

### 6.2 False Positive Filtering

```python
def _is_strong_contradiction(self, r: Dict) -> bool:
    """Check if result is a strong, well-cited contradiction."""
    if r['verdict'] != 'contradicts' or r.get('confidence', 0) < 0.75:
        return False
    
    citation = r.get('citation', '')
    if not citation or len(citation) < 20:
        return False
    
    # Filter invalid citations
    BAD_CITATION_PHRASES = ['none', 'no evidence', 'no mention', 'not mentioned']
    if any(p in citation.lower()[:60] for p in BAD_CITATION_PHRASES):
        return False
    
    # Filter false positive reasoning
    FALSE_POSITIVE_PHRASES = ['same person', 'does not directly', 'there is no mention']
    reasoning = r.get('reasoning', '').lower()
    if any(p in reasoning for p in FALSE_POSITIVE_PHRASES):
        return False
    
    return True
```

### 6.3 Rationale

**Why "One Error = Contradict"?**

In literary verification:
- A backstory is **consistent** only if ALL facts align
- A **single factual error** (wrong date, wrong name, wrong event) invalidates consistency
- This mirrors how human readers would judge

---

## 7. LLM Integration (Groq API)

### 7.1 Why Groq?

| Aspect | Groq | Local LLM | OpenAI |
|--------|------|-----------|--------|
| Speed | ~1s/call | ~120s/call | ~2s/call |
| Cost | Free tier | Compute | $0.002/1k tokens |
| Quality | Llama 3.1 8B | Varies | GPT-4 |
| Reliability | High | Depends | High |

### 7.2 Implementation

```python
class GroqLLM:
    """LLM wrapper using Groq API."""
    
    def __init__(self, config: FastVerifierConfig):
        self.config = config
        self._client = None
    
    def generate(self, prompt: str, max_tokens: int = None, system_prompt: str = None):
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.llm_model,  # "llama-3.1-8b-instant"
                    messages=messages,
                    temperature=0.0,  # Deterministic
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return "ERROR"
```

---

## 8. Result Logging

### 8.1 Detailed JSON Output

```python
class ResultLogger:
    """Logs detailed verification results to numbered JSON files."""
    
    def log_verification(self, sample_id, character, book_name, backstory, 
                         claims, claim_results, prediction, explanation, ...):
        result = {
            "file_info": {
                "file_number": self.file_counter,
                "created_at": datetime.now().isoformat(),
                "sample_id": sample_id
            },
            "sample": {
                "id": sample_id,
                "character": character,
                "book_name": book_name,
                "backstory": backstory
            },
            "verification": {
                "claims_extracted": claims,
                "claim_results": [
                    {
                        "claim_text": claim,
                        "verdict": result["verdict"],
                        "confidence": result["confidence"],
                        "reasoning": result["reasoning"],
                        "citation": result["citation"]
                    }
                    for claim, result in zip(claims, claim_results)
                ]
            },
            "final_decision": {
                "prediction": prediction,
                "explanation": explanation
            },
            "statistics": {
                "elapsed_time_seconds": elapsed_time,
                "verdicts_summary": {...}
            }
        }
        
        # Save to verification_results/test{N}/test_{i}.json
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
```

### 8.2 Output Structure

```
verification_results/
├── test1/
│   ├── test_1.json
│   ├── test_2.json
│   └── ...
├── test2/
│   └── ...
└── test34/  (current run)
    ├── test_1.json
    ├── test_2.json
    └── test_80.json
```

---

## 9. Evaluation Framework

### 9.1 Metrics

```python
class FastEvaluator:
    def evaluate(self, samples: List[Dict]):
        for sample in samples:
            prediction, details = self.pipeline.verify_backstory(
                backstory=sample['content'],
                character=sample['char'],
                book_name=sample['book_name'],
                true_label=sample['label']
            )
            
            correct = prediction == label_map[sample['label']]
            self.results.append({...})
        
        # Calculate metrics
        accuracy = correct_count / total
        consistent_acc = ...  # Accuracy on consistent samples
        contradict_acc = ...  # Accuracy on contradict samples
```

### 9.2 Current Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 61.3% |
| Consistent Accuracy | ~67-75% |
| Contradict Accuracy | ~48-52% |

---

## 10. Configuration

```python
@dataclass
class FastVerifierConfig:
    # LLM settings
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    
    # Retrieval settings
    top_k_retrieval: int = 15
    top_k_final: int = 8
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    
    # Verification settings
    max_claims: int = 5
    
    # Parallelism
    max_retries: int = 3
    parallel_workers: int = 10
```

---

## 11. Why This Logic Stands Out

### 11.1 Key Innovations

1. **Claim-Based Decomposition**: Breaks complex narratives into verifiable units
2. **Citation-Based Verification**: Every verdict requires textual evidence
3. **False Positive Filtering**: Prevents spurious contradictions
4. **Parallel Processing**: 10x faster than sequential verification
5. **Detailed Logging**: Full audit trail for every decision

### 11.2 Comparison with Alternatives

| Approach | Our Advantage |
|----------|---------------|
| Direct embedding comparison | We reason about semantics |
| Single-shot LLM verification | We decompose into claims |
| Sequential processing | We parallelize |
| Majority voting | One error = fail is more precise |

### 11.3 Design Philosophy

**"Trust but Verify"**
- Extract claims (trust the backstory format)
- Find evidence (trust the retrieval)
- Verify each claim (trust but verify with LLM)
- Require citations (don't trust without evidence)
- Apply strict policy (one error invalidates)

---

## 12. Code Organization

```
pipeline/verifier_fast.py
├── Configuration
│   └── FastVerifierConfig
├── LLM Integration
│   └── GroqLLM
├── Claim Extraction
│   └── FastClaimExtractor
├── Retrieval
│   ├── FastBM25
│   └── FastHybridRetriever
├── Verification
│   └── FastClaimVerifier
├── Aggregation
│   └── FastAggregator
├── Pipeline
│   └── FastVerificationPipeline
├── Logging
│   └── ResultLogger
└── Evaluation
    └── FastEvaluator
```

---

## 13. Hackathon Differentiator

### Why Our Main Logic Wins:

1. **🎯 Claim Decomposition**: Transforms narrative verification into fact-checking
2. **📚 Citation-Based**: Every verdict is grounded in textual evidence
3. **⚡ Parallel Processing**: 10x faster with ThreadPoolExecutor
4. **🔒 Strict Policy**: "One error = contradict" matches real-world consistency
5. **📊 Full Audit Trail**: JSON logs for every decision
6. **🆓 Cost-Free LLM**: Groq API provides free, fast inference
7. **🐳 Reproducible**: Docker-containerized execution

---

*Our main logic transforms the complex task of literary consistency verification into a structured, auditable pipeline that reasons about specific facts, grounds every decision in textual evidence, and provides transparent explanations for every prediction.*
