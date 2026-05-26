# Friends RAG — Part 2: When Paraphrases Aren't Really Paraphrases

## The Problem

The Part 1 evaluation produced an unexpected result: BM25 keyword search outperformed vector retrieval across all question categories, including ones that should have favored vector — paraphrased questions.

That contradicted the textbook narrative. Either the textbook was wrong, or the data was misleading. Spot-checking the Emory "paraphrases" suggested the second:

| Original | Emory "paraphrase" |
|---|---|
| What does Julio say to Jeannine? | What is Julio's parting comment to Jeannine? |
| Who said their first word? | Who just said his first word? |

These are *syntactic* paraphrases — different phrasing, same content words. The key vocabulary survives intact, which means BM25's "match rare terms" advantage survives too. Whatever vector retrieval was tested against in the Emory dataset, it wasn't lexical paraphrasing.

## The Experiment

I used Claude Haiku to generate 200 lexically-distinct paraphrases of original questions. The prompt instructed the model to replace proper nouns with descriptive equivalents ("Ross" → "the paleontologist") and replace content words with synonyms ("say" → "remark"). I then re-ran the same four retrieval modes against this new test set.

Sample generated paraphrases:

| Original | Generated lexical paraphrase |
|---|---|
| What is the gang watching on TV? | What program is the group viewing on the television set? |
| Where are Rachel and Ross having their conversation? | In what location are the fashion enthusiast and the paleontologist conducting their discussion? |

**A known limitation:** the generated paraphrases skew toward strong descriptive substitution — harder than typical user queries. They serve as a *worst-case* test for lexical robustness, not a representative one.

## The Results

| Mode | Emory paraphrased R@5 | LLM lexical R@5 | Drop |
|---|---|---|---|
| BM25 | 0.470 | **0.020** | ÷23 |
| Vector (scene) | 0.235 | 0.090 | ÷2.6 |
| Vector (naive 500-char) | 0.258 | 0.085 | ÷3.0 |
| Hybrid | 0.448 | 0.085 | ÷5.3 |

[Placeholder: bar chart comparing recall@5 across the three test sets]

## What This Means

**Three things became clear:**

1. **BM25's strength on the Emory dataset was an artifact of paraphrase quality.** Once paraphrases actually replace content words, BM25 collapses to 2% recall — essentially random. Its earlier dominance was the test set's, not its own.

2. **Vector retrieval handles lexical paraphrasing better than BM25 — by a factor of 4.5x in this experiment.** This is the textbook story finally appearing, just on a harder test set than the original benchmark provides.

3. **All four modes performed poorly in absolute terms.** Even vector's 9% recall@5 is bad. This points at the deeper issue: chunking granularity. When every scene's embedding represents many concepts at once, even semantically-correct queries can't reliably surface the right one.

**Hybrid retrieval also revealed a failure mode:** RRF fusion gives both retrievers equal weight. When BM25 contributes near-zero signal (as on the lexical set), its bad rankings drag down the fused result. Naive hybrid retrieval assumes both retrievers contribute meaningfully; when they don't, fusion hurts.

## What's Next

The granularity hypothesis is now the focal question. If scene-level chunks dilute the signal for fine-grained questions, finer chunks should help — across both test sets. The next experiment indexes the same corpus at two finer granularities (per-utterance and 3-utterance windows) and re-runs the same evaluation. We'll know whether granularity-matching is the right design principle or whether something else is going on.