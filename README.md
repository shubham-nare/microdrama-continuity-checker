# Script continuity checker for AI-written microdramas

Reads a show's episode screenplays (Kuku's timestamped format). Before any video is generated, it lists continuity problems for a script editor to accept or dismiss:
- a name spelled several ways
- a character's trait, prop, injury or family tie changing with no explanation
- one speaker label used for two people
- one character appearing under several labels

Each flag quotes the script with the episode, element ID and timestamp, and code checks every quote. The tool also writes a series bible and a per-scene reference card, the "who is on screen and what they look like right now" for shot prompts.

What it catches today, and what it misses, is measured and reported in `outputs/<run>/eval.json` and in the solution doc. It does not catch everything.

## Before a live run (e.g. a new show)
```bash
python run.py --preflight --data path/to/new_show     # checks format, model, key with one probe call, estimates calls/time
```
If an episode's model call fails, the run continues. The episode is listed as "not read" in the report, and the code-only name check still covers it.

## Run
```bash
pip install -r requirements.txt
cp .env.example .env                                     # add GEMINI_API_KEY (free tier: aistudio.google.com)
python run.py --data demo/show --out outputs/demo        # 3-episode fictional demo show, safe to share
python run.py --data data --out outputs/run1             # any folder of episode .txt files
python run.py --data data --out outputs/run1 --eval      # + seeded-error test, gold scoring, all checks
```
Outputs in the `--out` folder:
- `report.html`: open this first
- `flags.csv`: fill the `reviewer_decision` column
- `bible.json`
- `reference_cards.json`
- `eval.json` and `checks.json` (with `--eval`)
- `run_log.json`: calls, tokens, cost

## Swap the model (env vars only, no code change)
```bash
LLM_PROVIDER=gemini        LLM_MODEL=gemini-3.5-flash-lite
LLM_PROVIDER=openai_compat LLM_MODEL=<model> LLM_BASE_URL=<https://host/v1> LLM_API_KEY=<key>   # Groq, OpenRouter, vLLM, a company gateway
```
Responses are cached in `.cache/`, so a re-run is free and gives identical output.

## Human review
`review/aliases.yaml` holds confirmed decisions, for example "label OLD MAN in ep 1 is the grandfather". They override the model on the next run.

## Confidentiality
The Kuku screenplays (`data/`) and every output that quotes them (`outputs/`) are git-ignored and never committed.

## How this was built (AI use)
The code was written with Claude Code. `CLAUDE.md`, `PROMPTS.md` and `.claude/agents/` are the harness I used:
- a planning prompt
- an improvement loop where every iteration ran the full pack, the automatic checks and an independent "critic" reviewer, plus a "fresh-eyes" reviewer and a "contrarian" for ideas

The decisions, failures and fixes from that loop are summarised in the solution doc. The full logs are in the private outputs bundle. `demo/heldout/` is a fictional show written after the rules were final, used only as a held-out test.
