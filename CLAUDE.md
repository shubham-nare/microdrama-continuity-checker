# Kuku AI Generalist Assignment — project memory

You are helping me (the candidate) with a take-home for Kuku (AI microdrama company).
The full brief is in `brief.pdf`. Read it fully once at the start of every new session.
I will be interviewed on every decision, so every decision must be written down with its reason in `state/DECISIONS.md`.

## The brief in one screen
- Pick ONE of: P1 Find the Promos (input pack = 50 screenplays in `data/`), P2 AI Lawyer for India, P3 Find your own microdrama problem.
- P1 promo = 20–60 s clip (one stretch, or several joined segments, same or different episodes). It must open on a hook in the first 3 s, end on an unresolved cliffhanger, and make sense to a viewer who has never seen the show. For each promo, output: the episode, start/end timestamps of every segment in order, the hook, the cliffhanger, one line on why it works, plus a ranking (what to post first and why) and a quality check (where the tool picks badly).
- Deliverables: a solution doc (PDF, ≤ 2 pages), a prototype link (repo / Space / Colab), a raw outputs folder (unedited outputs on the FULL input pack), and an optional ≤ 3-minute video.
- Graded on 5 lenses: (1) problem choice and framing, (2) approach and scale (what's AI vs code/rules, where a human stays in the loop, cost at volume), (3) quality checking with real runs and honest failures, (4) a prototype that runs on real input, (5) communication that is clear, specific and short.
- Hard constraints: free tiers only. The model must be swappable through one config value or env var (they'll plug in their own API). The tool must re-run on a NEW show during the interview with one command. The doc must name the AI tools used and one place where they got it wrong. Honesty beats polish.
- Confidentiality: the `data/` screenplays must never be committed to a public repo or published anywhere. `data/` is in `.gitignore`.

## Files that are your memory (the loop relies on these)
- `state/DECISIONS.md`: each decision, the alternatives, and why. Phase 0 output lives here.
- `state/ITERATION_LOG.md`: one entry per iteration (template inside). Read it at the START of every turn; it is your memory across context compaction.
- `state/SCORECARD.md`: the latest critic scores. Only the `critic` subagent's output goes here, copied verbatim.
- `state/MISTAKES.md`: every bug or bad output you found and how you fixed it. Feeds the doc's "where AI got it wrong" and "where it fails" sections.

## Iteration protocol (one iteration = one turn while /goal is active)
1. Read `state/ITERATION_LOG.md`, `state/SCORECARD.md` and `state/MISTAKES.md`.
2. Pick the ONE highest-impact weakness from the last critic report (blockers first). Write it down before you touch code.
3. Every 3rd iteration, call the `contrarian` subagent first and consider its top idea. Adopt it only if it beats the current weakness on expected score gain per hour. Log the decision either way.
4. Implement the change. Keep it small, and keep the model behind the provider interface.
5. Re-run the tool end-to-end on the FULL input pack with the single entry command. Save the outputs to `outputs/iter_NN/`. Never hand-edit outputs.
6. Run the deterministic checks (below) and print the results.
7. Call `fresh-eyes` on a sample of outputs, then call `critic`. Give both ONLY file paths. Never pass your own opinion, reasoning or the score you expect.
8. Copy the critic's scorecard into `state/SCORECARD.md`, append an entry to `state/ITERATION_LOG.md`, and add any newly found mistakes to `state/MISTAKES.md`.
9. End the turn by printing the `SCORECARD` block (format at the bottom), so the /goal evaluator can read it.

Rules: never edit `.claude/agents/*`, the rubric or the goal mid-loop. Never lower a bar to pass it. If something was cut or doesn't work, say so in the doc instead of hiding it. If two iterations in a row don't improve the score, change approach (not just parameters) and log why.

## Deterministic checks (code, not model judgement). Applies if P1 is chosen.
- Every segment's start/end timestamp exists in (or lies inside) a timed element in the source screenplay. Parse the files with code, don't trust the LLM.
- Total promo duration is computed by code and is between 20 and 60 s. Segments are in order and don't overlap.
- The first 3 s of segment 1 contains the claimed hook line or action (check with a text match against the script).
- The last segment ends on the claimed cliffhanger element and does not continue into its resolution.
- No two promos in the top 10 are near-duplicates (heavy overlap of the same moments).
- Episode coverage and promo-type diversity (slap / reveal / betrayal / romance / power flip …) are reported.
- Re-running on one held-out episode folder with a different model name still produces valid output (swap test).
For P2/P3, write an equivalent list into `state/DECISIONS.md` during Phase 0 (e.g. P2: a fixed test-case suite, a citation-grounding check that every cited law/section exists in the bundled source text, and refusal/"see a lawyer" cases).

## Directions worth exploring (ideas, not mandates; the generic solution is "prompt an LLM per episode, ask for promos")
Think beyond that. Examples of axes: a code-first candidate generator with LLM scoring (cost and determinism); cross-episode stitched promos; a separate cold-viewer judge for standalone-ness; ranking that balances strength and diversity of hooks; multiple cut variants (A/B) per moment; a human-editor review screen with accept/reject that feeds back; cost per show at Kuku's weekly volume; a failure taxonomy backed by real examples.

## SCORECARD block format (print at the end of every turn)
```
SCORECARD iter=NN
lenses: framing=x/10 approach=x/10 quality=x/10 prototype=x/10 communication=x/10 overall=x.x
blockers: <none | list>
delta_vs_prev: +/-x.x   delta_vs_prev2: +/-x.x
full_run: <command> -> <exit code, #promos>
deterministic_checks: <passed n/m>
deliverables: doc[ ] repo[ ] run_steps[ ] raw_outputs[ ] model_swap[ ] ai_tools_section[ ]
next: <the single next weakness>
```
