# Verification Pipeline Change Log

## Goal
Achieve 60%+ accuracy on BOTH consistent and contradict classes.

---

## Run History

| Run | Overall | Consistent | Contradict | Notes |
|-----|---------|------------|------------|-------|
| test6 | 52.5% | 45.1% | 65.5% | Too aggressive - many false positives |
| test7 | 65.0% | 100% | 0% | Too conservative - everything marked unclear |
| test8 | 85.0% (20 samples) | 92.3% | 71.4% | Looked promising on small sample |
| test9 | 60.0% | 74.5% | 34.5% | Scaled poorly - missing contradictions |
| test10 | 43.3% | 11.1% | 91.7% | Fabrication detection too aggressive |
| test11 | 60.0% | 94.4% | 8.3% | Too conservative again |
| test12 | 63.3% | 94.4% | 16.7% | Sequential early-stop, still too conservative |
| test14 | 60.0% | 94.4% | 8.3% | Lowered thresholds, LLM still marking unclear |
| test15 | 56.7% | 50.0% | 66.7% | **BOTH >50%!** New prompt working, but FP rate high |
| test17 | 66.7% | 72.2% | 58.3% | Query expansion helped consistent! |
| test20 | 70.0% | 77.8% | 58.3% | **Best on 30 samples** - improved prompt |
| test21 | 54.0% | 67.7% | 31.6% | Degraded on 50 samples - not representative |
| test24 | 60.0% | 96.0% | 0.0% | **CRITICAL** - 0% contradict, prompt too conservative |
| test25 | 60.0% | 52.0% | 73.3% | Swung too far - now too many false positives |
| test26 | 67.5% | 96.0% | 20.0% | Swung back - missing contradictions again |
| test27 | 72.5% | 72.0% | 73.3% | **🎉 BOTH >60%!** Balanced prompt working |
| test28 | 65.0% | 72.5% | 51.7% | Full 80 samples - contradict dropped below 60% |
| test29 | 70.0% | 68.0% | 73.3% | **BOTH >60%** on 40 samples with bio retrieval |
| test30 | 61.3% | 66.7% | 51.7% | Full 80 - contradict dropped again |
| test31 | 65.0% | 96.0% | 13.3% | Aggressive filtering - too conservative |
| test32 | **75.0%** | **76.0%** | **73.3%** | **🎉 BOTH >60%!** on 40 samples |
| test33 | 65.0% | 74.5% | 48.3% | Full 80 - contradict at 48.3% (below 60%) |
| test34 | 61.3% | 66.7% | 51.7% | Full 80 - consistent pattern ~50% contradict |
| test35 | 62.5% | 60.0% | 66.7% | Lowered threshold to 0.5 - too aggressive |
| test36 | 62.5% | 88.0% | 20.0% | Strict "same topic" prompt - too conservative |
| test37 | 70.0% | 72.0% | 66.7% | **BOTH >60%** on 40 samples - reverted to balanced |
| test39 | 57.5% | 60.0% | 53.3% | Added weak contradiction tier (0.5-0.6 with citations) |
| test40 | 67.5% | 68.0% | 66.7% | **BOTH >60%** on 40 samples - improved false positive filtering |
| test41 | 61.3% | 68.6% | 48.3% | Full 80 - contradict still ~48% |
| test44 | **63.8%** | **68.6%** | **55.2%** | Full 80 - improved prompt with verification priority |
| test45 | 58.8% | 66.7% | 44.8% | ❌ FAILED - "detective" instruction degraded contradict |

---

## Current Status Summary
**Goal**: 60%+ on BOTH consistent and contradict classes.

**Best Results**:
- On 40 samples: test32 (76%/73.3%), test37 (72%/66.7%), test40 (68%/66.7%) all achieve goal ✅
- On 80 samples: test44 achieved 68.6% consistent ✅, 55.2% contradict (close to 60%)

**Analysis**:
- 40-sample goal consistently achieved (4+ successful runs)
- 80-sample contradict improved from ~48% to 55.2% with verification priority prompt
- Added "actively search for supporting statements" instruction helped balance

**Key Components**:
1. Verification prompt with priority order (contradicts → supports → unclear)
2. Biographical query expansion for retrieval
3. False positive filtering for "same person", "no mention of", etc.
4. Weak contradiction tier (0.5-0.6 confidence with citations)

**Key Components Working**:
1. Verification prompt with clear CONTRADICTS vs UNCLEAR distinction
2. Biographical query expansion for better retrieval
3. False positive filtering for "same person" and "does not directly" patterns
4. Aggregation: single 0.6+ contradiction = CONTRADICT

---

## Change #14: ❌ FAILED - More aggressive contradiction detection
**Date**: test45

**Problem Identified**:
- test44 achieved 68.6% consistent, 55.2% contradict
- Still below 60% contradict target

**Solution Attempted**:
Modified system prompt to be more aggressive in detecting contradictions.

**Result**: test45 - 58.8% overall, 66.7% consistent, 44.8% contradict
- **DEGRADED**: Contradict dropped from 55.2% to 44.8% (-10.4%)
- Overall accuracy dropped from 63.8% to 58.8% (-5%)

**Analysis**: 
Making the prompt more aggressive backfired - likely caused the LLM to be less precise
or introduced confusion, leading to more false negatives (missed contradictions).

**Decision**: Reverted this change - keeping test44 as best configuration.

**Note**: Detective-style instruction planned for next run (test46).

---

## Change #13: Enhanced false positive filtering
**Date**: Latest iteration

**Problem Identified**:
- test30 on 80 samples: 66.7% consistent, 51.7% contradict
- False positives: "no mention of X" being treated as contradiction
- Example: "rescued Yurook" → LLM says CONTRADICTS because "no mention of Yurook"

**Solution**:
Added more FALSE_POSITIVE_PHRASES:
- "no mention of", "there is no mention", "does not mention"
- "no evidence of", "no information about"
- Applied filter to 0.6+ confidence contradictions too

**Expected Impact**: Fewer false positives from "no mention = contradiction" pattern

---

## Analysis: Why test28 contradict dropped

Looking at missed contradictions (e.g., test_42 - Faria):
- Claim: "born in Parma" vs Book: "born at Rome" 
- Problem: Retrieval didn't find the "born at Rome" passage
- All 5 claims marked UNCLEAR → aggregator defaults to CONSISTENT

**Root Cause**: Retrieval not finding relevant evidence for some contradiction patterns.
**Next Step**: Improve retrieval targeting or adjust aggregation for edge cases.

---

## Change #12: Improved biographical query expansion
**Date**: Latest iteration

**Problem Identified**:
- Missed contradictions due to retrieval not finding key passages
- Example: "Faria born in Parma" vs book's "born at Rome" - retrieval didn't search for "Faria born"

**Solution**:
Enhanced `_extract_key_terms()` to add biographical query patterns:
1. Add "character born" query for birth claims
2. Add "character died death" query for death claims  
3. Add key biographical terms: arrested, imprisoned, escaped, born, died, married, father, mother
4. Increase query limit from 5 to 6

**Expected Impact**:
- Better retrieval of biographical facts (birthplace, dates, family)
- More contradictions detected when evidence exists but was missed before

---

## Change #11: Sharper CONTRADICTS vs UNCLEAR distinction
**Date**: Latest iteration

**Problem Identified**:
- test26: 96% consistent, 20% contradict - back to missing contradictions
- Analysis: LLM found conflicting evidence but marked UNCLEAR instead of CONTRADICTS
- Example: Claim "arrested in 1815", evidence "arrested in 1811" → LLM said UNCLEAR
- The reasoning found the difference but didn't classify it as contradiction

**Root Cause**:
The prompt gave too many examples of UNCLEAR (fabricated details), making LLM default to UNCLEAR even when real conflicts exist.

**Solution**:
Restructured prompt to be more decisive:
1. Moved KEY RULES for CONTRADICTS to top (dates, facts, opposite statements)
2. IMPORTANT callout: "If claim states specific fact and evidence shows DIFFERENT specific fact about SAME topic → CONTRADICTS"
3. Clearer examples showing the same-topic/different-fact pattern
4. User prompt reinforces: "If evidence shows DIFFERENT facts about SAME topic → CONTRADICTS"

**Expected Impact**:
- LLM should recognize that different dates/facts about same topic = CONTRADICTS
- Maintain UNCLEAR for truly fabricated/absent topics
- Better balance between the two classes

---

## Change #10: Clarify Absence vs Contradiction
**Date**: Latest iteration

**Problem Identified**:
- test25: 52% consistent, 73% contradict - too many false positives
- Analysis showed LLM marking CONTRADICTS when there's no evidence (fabricated details)
- Example: "rescued elder Yurook" → LLM said CONTRADICTS because "evidence doesn't mention Yurook"
- This is incorrect logic: absence of evidence is NOT contradiction

**Root Cause**:
The prompt said "CONTRADICTS = Evidence shows DIFFERENT facts" but LLM interpreted 
"no mention of X" as "different from X". Need to make distinction crystal clear.

**Solution**:
Rewrote prompt with explicit examples of what is and isn't a contradiction:
1. Added CRITICAL RULE: "A contradiction requires CONFLICTING EVIDENCE, not missing evidence"
2. Added examples of UNCLEAR (fabricated details): "rescued Yurook but evidence never mentions Yurook → UNCLEAR"
3. Added reminder in user prompt: "Only CONTRADICTS if evidence shows DIFFERENT facts. No mention = UNCLEAR"

**Expected Impact**:
- Fewer false positives on consistent samples (fabricated details → UNCLEAR, not CONTRADICTS)
- Should maintain ability to catch real contradictions (conflicting facts)

---

## Change #9: Balanced Verification Prompt
**Date**: Latest iteration

**Problem Identified**:
- test24: 96% consistent, 0% contradict - pipeline completely unable to detect contradictions
- Analysis showed LLM marking everything as UNCLEAR or SUPPORTS
- Old prompt: "Only say CONTRADICTS if you can cite the specific conflicting fact"
- This made LLM overly cautious, avoiding CONTRADICTS entirely

**Root Cause**:
The verification prompt was asymmetric:
- SUPPORTS was the easy default when evidence existed
- CONTRADICTS had a high bar ("must cite specific conflicting fact")
- LLM preferred UNCLEAR over risking a wrong CONTRADICTS

**Solution**:
Rewrote verification prompt to be symmetric:
1. CONTRADICTS: "Evidence shows DIFFERENT facts than claimed"
2. SUPPORTS: "Evidence actually confirms" - not just "no contradiction"
3. UNCLEAR: "ONLY when evidence is about completely different topics"
4. Added example: "father supported revolution" vs "father was royalist" = CONTRADICTS

---

## Change #1: Initial Setup
- Created folder structure (test1/, test2/, etc.)
- Basic claim extraction and verification

## Change #2: Aggressive Contradiction Detection
- Added fabrication detection
- **Result**: Too many false positives

## Change #3: Conservative Approach  
- Required 2+ high-confidence contradictions with citations
- Strict citation filtering
- **Result**: Missed too many contradictions

## Change #4: Balanced Approach (Current)
- 1 high-confidence cited contradiction OR 2+ any contradictions
- Moderate citation requirements
- **Result**: Still oscillating

---

## Change #6: Improved Retrieval (NEW)
**Date**: 2026-01-11

**Approach**:
- Query expansion: Extract key terms (possessives, entities) for multi-query retrieval
- Character-aware search: Pass character name to retriever
- Balanced BM25/vector weights: 0.5/0.5 (was 0.4/0.6)

**Rationale**:
- False positives often caused by retrieving info about character X when claim is about "X's parents"
- Multi-query retrieval should find more targeted evidence
- Higher BM25 weight helps exact name matching

**Implementation**:
- `_extract_key_terms()`: Extracts possessives, quoted phrases, capitalized words
- Multi-query RRF: Searches with multiple query variants, merges results
- Character passed to retriever for entity-focused search

---

## Change #8: Decisive Verification Prompt
**Date**: 2026-01-11

**Problem Identified**:
- LLM defaulting to UNCLEAR too often (many verdicts are unclear even with relevant evidence)
- No examples of what SUPPORTS looks like
- Aggregation rule 2 too aggressive (single 0.7 contradiction triggers CONTRADICT)

**Changes Made**:

1. **New Verification Prompt**:
   - Added CRITICAL RULES requiring decision when evidence mentions character+topic
   - "If evidence mentions CHARACTER and TOPIC, you MUST give SUPPORTS or CONTRADICTS"
   - Added SUPPORTS examples with confidence ranges
   - "Absence of information is NOT a contradiction"
   - UNCLEAR only when evidence is about completely different character/topic

2. **Improved Aggregation Logic**:
   - Rule 2: Now requires contradiction_score > support_score + 0.3 (not just >= 0.7)
   - Added Rule 3: Strong support majority (3+ supports OR 2+ with 0 contradicts) → CONSISTENT
   - More balanced comparison between support and contradiction scores

3. **Evidence Formatting**:
   - Increased evidence content from 800 to 1000 chars for more context
   - Simplified metadata headers (removed Story, ID - less clutter)
   - Added character name to user prompt for context

**Expected Impact**:
- Fewer UNCLEAR verdicts → more decisive SUPPORTS/CONTRADICTS
- Better balance between finding contradictions vs false positives
- More robust aggregation with score comparison

---

## Change #7: False Positive Filtering
**Date**: 2026-01-11

**Problem Identified**:
- test21 showed 54% on 50 samples (degraded from 70% on 30)
- False positives from:
  1. LLM confused "Tom Ayrton/Ben Joyce" as two separate people
  2. Speculative reasoning ("implies", "suggests")
  3. Weak reasoning ("does not directly")

**Solution**:
- Added reasoning quality check in `_is_strong_contradiction()`
- Filter out contradictions with speculative/weak reasoning phrases
- Detect alias confusion ("same person", "same individual")

**Key Error Patterns**:
1. **False Positives**: Misinterpretation, alias confusion, speculative reasoning
2. **False Negatives**: Fabricated details (no evidence to contradict)

---

## Key Learnings

1. **Fabrication detection is unreliable** - "no evidence" != contradiction
2. **Citation quality matters** - filter out meta-text like "evidence does not..."
3. **Balance is key** - single high-confidence contradiction should be enough
4. **Sample size matters** - 30-sample results don't always scale to 50/80 samples
5. **Sequential processing** - helps reduce noise from aggregating weak signals
6. **Reasoning quality** - check for speculative/weak phrases to filter false positives
