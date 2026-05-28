# Holdout test queries — DO NOT RUN UNTIL PIPELINE IS COMPLETE

These queries are the human-curated final test of the system. They are intentionally not used during development to avoid overfitting.

Run them once, at the end of Phase 1 (after parent-child + reranker + smart hybrid are integrated). Record results honestly, including failures.

## Query: "Who was Joey in love with?"

Expected ground-truth scenes:

1. **S08E15** — "The One With The Birthing Video"
   Joey opens up to Ross about "this woman" he's in love with — describes her as with another guy long-term, mutual friend. Ross doesn't realize it's Rachel and encourages him.

2. **S08E16** — "The One Where Joey Tells Rachel"
   - Central Perk: Joey tells Ross it's Rachel. Ross spirals ("I gotta go," leaves milk).
   - Restaurant: Joey confesses to Rachel directly. Rachel gently declines.

3. **S08E17** — "The One With The Tea Leaves"  *(Expected to fail — heavily paraphrased)*
   Rachel and Joey's reconciliation conversation. Joey: "I haven't thought at all about how I put myself out there..." — meta-awkwardness.

## Scoring rubric (when test is run)

- **Perfect:** Top 5 includes scenes from S08E15, S08E16, and the answer (Rachel) is named in Claude's response
- **Good:** Top 5 includes S08E15 or S08E16; Claude names Rachel
- **Partial:** Top 5 hits some scenes but Claude doesn't confidently identify Rachel
- **Miss:** Top 5 has no scenes from S08E15-17

## Other queries to add to holdout set

(Build this list during Weekend 4 as you think of them — but only add queries we haven't already tested during development.)