# Hosted on
https://www.samvaad-sathi.barabaricollective.org/

# genAI Interview-Coaching Toolkit

This repository hosts a collection of **speech & text-analysis utilities** that power an
interview-coaching API.  It receives audio answers from candidates, transcribes
them, and produces actionable feedback on pacing, pauses, pronunciation, and
domain knowledge.

The project is split into small, focused modules so that individual components
can be reused or tested in isolation.  The sections below give a bird-eye view
of the directory layout and explain how the pieces interact at runtime.

---

## 1. Directory structure

```
.
├── genAI/                         # Main Python package
│   ├── server.py                  # FastAPI service – single entry-point
│   ├── pacing.py                  # WPM & speaking-rate analysis
│   ├── pauses.py                  # Pause extraction / LLM-assisted feedback
│   ├── test_pauses_samples.py     # Unit-style samples for pause logic
│   ├── example.env                # Template for required env variables
│   │
│   ├── OpenSmile_Approach/        # Paralinguistic feature extractor (openSMILE)
│   │   ├── app.py                 # Stand-alone script to pull eGeMAPS features
│   │   └── *.wav / *.md           # Demo audio + output
│   │
│   ├── misspronounciation/        # Pronunciation evaluation helpers
│   │   ├── main.py                # G2P + alignment proof-of-concept
│   │   ├── custom_lexicon.py      # Domain-specific phoneme additions
│   │   └── ... (dataset/, audios/)# Audio assets & notebooks
│   │
│   ├── pauses_input_samples/      # JSON snippets for manual pause testing
│   └── prompts/                  # LLM prompt templates & notebooks
│       ├── prompts.py             # Re-usable system / user prompt strings
│       ├── gen_que_prompt.py      # Question-generation helper
│       └── gen_que_testing/       # Sample analyses + datasets
│
├── sample_jsons/                  # End-to-end request / response examples
│   └── *.json
│
├── test/                          # Lightweight regression harness
│   ├── testing_script.py          # Downloads audio & hits local API
│   ├── Test_Cases_1.csv           # Drive links to sample answers
│   └── audios/                    # Small speaker excerpts for CI
│
├── requirements.txt               # Minimal runtime dependencies
└── readme.md                      # 🠔 you are here
```

### What each key file/folder is for

| Path | Purpose |
|------|---------|
| `genAI/server.py` | Launches the FastAPI app; exposes endpoints for transcription (`/transcribe_whisper`), pacing, pause, and communication analyses.  Orchestrates calls to OpenAI, Deepgram, Sarvam.
| `genAI/pacing.py` | Sliding-window WPM computation, pace classification (too slow / ideal / too fast), and human-readable feedback generator.
| `genAI/pauses.py` | Detects silences between `words[]` timestamps, classifies them (long, rushed, strategic) and produces LLM-guided improvement tips.
| `genAI/OpenSmile_Approach/app.py` | Self-contained script that demonstrates how to extract acoustic features (eGeMAPS) via openSMILE.
| `genAI/misspronounciation/` | Experimental playground for phoneme-based pronunciation scoring using whisper + `phonemizer` alignment.
| `genAI/prompts/` | Centralised store for all system / user prompts used by the API (resume extraction, text analysis, question generation, etc.).
| `sample_jsons/` | Canonical request / response pairs that serve as documentation as well as regression fixtures.
| `test/` | Automated smoke tests – download sample audio, hit local API, and collate results into a CSV report.


---

## 2. High-level code flow

1. **Incoming request**  
   A client (web or CLI) uploads an audio answer to one of the `/…-analysis`
   endpoints exposed by **`server.py`**.

2. **Transcription**  
   The audio is forwarded to *Deepgram* (or Whisper) – the verbose JSON
   containing word-level timestamps is cached and returned to the caller.

3. **Pace Assessment**  
   `pacing.calculate_pace_metrics()` computes WPM over 5-second hop windows,
   classifies each window, aggregates percentages, and returns granular
   segments.  `pacing.provide_pace_feedback()` converts that into a coach-style
   paragraph.

4. **Pause Analysis**  
   `pauses.analyze_pauses()` extracts silences, derives dynamic thresholds from
   the user’s own speaking rate, queries an LLM for *strategic* pause
   suggestions, and outputs an overview + rubric-based score (1–5).

5. **Pronunciation (optional / experimental)**  
   If enabled, `misspronounciation/main.py` uses *phonemizer* + G2P alignment
   to detect word-level deviations from expected phonemes.

6. **Summary & Response**  
   `server.py` bundles the individual analyses together with auxiliary
   metadata (process time, job role prompts, etc.) and sends the JSON back to
   the frontend.


---

## 3. Running locally

```bash
# 0. Python >=3.10 recommended
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# 1. Export required keys (see genAI/example.env)
$ export OPENAI_API_KEY=…
$ export DEEPGRAM_API_KEY=…
$ export SARVAM_API_KEY=…

# 2. Fire up the API
$ uvicorn genAI.server:app --reload

# 3. Hit an endpoint
$ curl -X POST -F "file=@path/to/audio.wav" http://localhost:8000/transcribe_whisper
or
open http://localhost:8000/docs for swagger
```
