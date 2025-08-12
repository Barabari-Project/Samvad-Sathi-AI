import pdb

# -------------
#  Pause analysis utilities
# -------------
#  This module analyses the pauses in the timestamp information returned by
#  Whisper / Deepgram word–level transcripts.  It now returns two additional
#  pieces of information expected by the product team:
#   • “actionable_feedback” – a concise, user-friendly coaching paragraph.
#   • “score”               – an integer in the range 1-5 indicating how well
#                             the candidate managed pauses.
#
#  The holistic score is produced by the LLM.  The exact rubric that the model
#  must follow is injected verbatim inside the prompt so that the evaluation is
#  reproducible and transparently communicated to users.
#
#    5  – Excellent pause management (<5 % long pauses, <10 % rushed, ≥15 %
#         strategic/mid-length pauses)
#    4  – Good (5-10 % long OR 10-20 % rushed, strategic ≥ 10 %)
#    3  – Fair (10-20 % long OR 20-35 % rushed, strategic < 10 %)
#    2  – Poor (>20 % long OR >35 % rushed)
#    1  – Very poor (long > 30 % or rushed > 50 %)
#
#  The rest of the dictionary structure that downstream code relies on remains
#  unchanged for backwards-compatibility (``overview``, ``details`` and
#  ``distribution``).
# ---------------------------------------------------------------------------

import json
from typing import Dict, List

# ---------------------------------------------------------------------------
# Public helper – suggest_pauses ------------------------------------------------
# ---------------------------------------------------------------------------
# This lightweight wrapper around the LLM is intentionally **self-contained** so
# that projects can call ``pauses.suggest_pauses`` without having to prepare the
# entire, more complex ``analyze_pauses`` pipeline.  The function takes the raw
# Whisper/Deepgram *verbose_json* dictionary, extracts the ``words`` list and
# asks the LLM to insert explicit "[PAUSE]" tokens where a short pause would
# improve clarity or emphasis.
#
# To keep the wider package fully offline-capable the implementation falls back
# to a no-op stub when a global ``call_llm`` helper is **not** available (for
# example inside the automated test runner that has no network access).  In
# that case the model is simply skipped and the function returns an empty list
# which is still a perfectly valid – albeit rather boring – suggestion set.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NOTE: *call_llm* is an optional override so that callers such as the main
# ``analyze_pauses`` pipeline – which already carries an explicit LLM wrapper
# argument – can forward it directly without polluting the global namespace.
# ---------------------------------------------------------------------------


def suggest_pauses(asr_output: dict, call_llm=None) -> List[int]:
    """Return word indices where a brief pause *before* the word is advised.

    Parameters
    ----------
    asr_output : dict
        Whisper / Deepgram *verbose_json* transcript – must contain a top-level
        ``"words"`` list where each item has the keys ``start``, ``end`` and
        ``word``.

    Returns
    -------
    list[int]
        Zero-based indices into ``words`` that indicate recommended pause
        boundaries.  The index refers to the *following* word so callers can
        insert a break **before** that token when rendering subtitles or
        generating coaching feedback.
    """

    # ------------------------------------------------------------------
    # 0. Basic validation ------------------------------------------------
    words = asr_output.get("words", [])
    if not words:
        print("Words Empty Line 80")
        return []

    # ------------------------------------------------------------------
    # 1. Build the LLM prompt -------------------------------------------
    # ------------------------------------------------------------------
    transcript = " ".join(w["word"] for w in words)

    example_input = "My name is bond James Bond"
    example_output = "My name is bond [PAUSE] James Bond"

    prompt = (
        "You are an expert in spoken communication. Analyze this transcript "
        "and insert \"[PAUSE]\" tokens where brief pauses would improve "
        "clarity, emphasis, or natural flow. Follow these rules:\n\n"
        "1. PRESERVE all original words exactly as given\n"
        "2. ONLY insert \"[PAUSE]\" tokens – no other changes\n"
        "3. Insert pauses only at natural break points:\n"
        "   - Before important words for emphasis\n"
        "   - Between logical thought groups\n"
        "   - After conjunctions or transitional phrases\n"
        "   - Before appositives or clarifying information\n"
        "4. Never add punctuation or modify words\n"
        "5. Never insert a pause before the first word\n\n"
        f"Example Input: \"{example_input}\"\n"
        f"Example Output: \"{example_output}\"\n\n"
        "Now process this transcript:\n"
        f"\"{transcript}\"\n\n"
        "Output ONLY the modified transcript with \"[PAUSE]\" tokens. "
        "Do not include any other text or explanations."
    )

    # ------------------------------------------------------------------
    # 2. Call the LLM (if available) ------------------------------------
    # ------------------------------------------------------------------
    # The main FastAPI service injects a convenience wrapper called
    # ``call_llm`` into the global namespace.  When the module is executed in
    # isolation – for instance inside a CI environment or during unit tests –
    # this helper may be missing.  We therefore attempt to locate it first and
    # fall back to a stub that simply returns the *unmodified* transcript so
    # that this function stays side-effect free and never crashes due to the
    # absence of network connectivity.

    _call_llm = call_llm if callable(call_llm) else globals().get("call_llm")

    if callable(_call_llm):
        try:
            response = _call_llm(prompt).strip().strip('"')
            print("LINE 127")
            print(response)
        except Exception:
            # Any error – network, model, rate limit – gracefully degrade.
            response = transcript.strip('"')
    else:
        # Offline / test mode – skip LLM
        print("WARNING SKIPING THE [PAUSE] RECOMMENDATION")
        response = transcript

    # ------------------------------------------------------------------
    # 3. Derive pause indices by aligning tokens ------------------------
    # ------------------------------------------------------------------
    return _find_pause_indices(words, response)

def _find_pause_indices(original_words: List[dict], paused_transcript: str) -> List[int]:
    """Compare *paused_transcript* with *original_words* to locate [PAUSE] tags.

    The helper performs a simple token-by-token alignment, tolerant to minor
    mismatches that can happen when the LLM accidentally drops or duplicates a
    word.  Whenever a \"[PAUSE]\" token is encountered **and** the following
    response token matches the *current* original word we record the index.  A
    pause marker that appears right at the end of the string is ignored as it
    cannot be mapped to a *following* word.
    """

    # pdb.set_trace()
    original_tokens = [w["word"] for w in original_words]
    response_tokens = paused_transcript.split()

    pause_indices: List[int] = []
    orig_idx = 0
    resp_idx = 0

    while orig_idx < len(original_tokens) and resp_idx < len(response_tokens):
        token = response_tokens[resp_idx]

        if token == "[PAUSE]":
            # Look-ahead: next response token *should* correspond to the current
            # original word. If it does, we mark the current orig_idx as the
            # boundary *after* the previous word (i.e. before the current
            # word).  We explicitly skip index 0 so callers never pause before
            # the first word.
            resp_idx += 1  # advance to next response token

            if resp_idx >= len(response_tokens):
                break  # dangling [PAUSE] at the very end – ignore

            if response_tokens[resp_idx] == original_tokens[orig_idx]:
                if orig_idx > 0:
                    pause_indices.append(orig_idx)
            # Whether or not the word matched, we continue the outer loop – no
            # increment of orig_idx here because we still need to consume the
            # current original token in the next iteration.
            continue

        # Regular token: try to align with current original token -------------
        if token == original_tokens[orig_idx]:
            orig_idx += 1
            resp_idx += 1
        else:
            # Mismatch – advance only the *response* pointer.  This heuristic
            # lets us recover from minor hallucinations without completely
            # derailing the alignment.
            resp_idx += 1

    # Ensure uniqueness & stable ordering
    seen = set()
    deduped: List[int] = []
    for idx in pause_indices:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)

    return deduped


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _format_ts(seconds: float) -> str:
    """Convert raw seconds to "MM:SS" string for human-friendly references."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _extract_pauses(words: List[dict]) -> List[dict]:
    """Return a list of pauses with useful metadata."""
    pauses: List[Dict] = []
    for i in range(len(words) - 1):
        pause_duration = words[i + 1]["start"] - words[i]["end"]
        if pause_duration <= 0:
            # overlapping words – ignore
            continue
        pauses.append(
            {
                "index": i,
                "start": words[i]["end"],
                "end": words[i + 1]["start"],
                "duration": pause_duration,
                "before_word": words[i]["word"],
                "after_word": words[i + 1]["word"],
            }
        )
    return pauses


def analyze_pauses(asr_output: dict, call_llm, extract_json_dict):
    """Analyse pauses and generate actionable feedback.

    Parameters
    ----------
    asr_output : dict
        The *verbose_json* output of Whisper / Deepgram.  Must contain a
        top-level ``"words"`` list with ``start``, ``end`` & ``word`` keys.
    call_llm : Callable[[str], str]
        Convenience wrapper around OpenAI chat completion that the main API
        code already provides.  Must take a *prompt* and return the raw model
        response.

    Returns
    -------
    dict
        A dictionary with the following keys:

        overview        – single-sentence summary
        details         – list of granular comments with timestamps
        distribution    – % distribution of pause types
        actionable_feedback – concise coaching paragraph generated by LLM
        score           – integer (1-5) following the explicit rubric
    """

    words = asr_output.get("words", [])
    if not words:
        return {
            "overview": "No word-level timestamps provided – unable to analyse pauses.",
            "details": [],
            "distribution": {},
            "actionable_feedback": "Please re-upload the audio so that word timings are included.",
            "score": 1,
        }

    # ------------------------------------------------------------------
    # 1. Extract raw pauses ------------------------------------------------
    pauses = _extract_pauses(words)

    # ------------------------------------------------------------------
    # 2. Derive **recommended pause indices** via LLM helper -------------
    # ------------------------------------------------------------------

    try:
        recommended_pause_indices = suggest_pauses(asr_output, call_llm)
        print("LINE 282")
        print(recommended_pause_indices)
    except Exception:
        # Any issue – fall back to an empty list so downstream logic keeps
        # working without interruption.
        print("WARNING ANALYZE PAUSES > recommended_pause_indices is not realised as expected. LINE 287")
        recommended_pause_indices = []

    # ------------------------------------------------------------------
    # 3. Classify pauses ---------------------------------------------------
    # ------------------------------------------------------------------
    # Dynamic thresholding based on the *speaker's own* statistics --------
    # ------------------------------------------------------------------
    # A one-size-fits-all threshold (e.g. “long pause > 3 s”) implicitly
    # assumes ~120 WPM which breaks down for very quick or very slow
    # speakers.  Instead of hard-coding numbers we derive **all** pause
    # categories from the observed distribution in the current answer.
    #
    #   • rushed      – shorter than the 25-th percentile (Q1)
    #   • long        – longer than the 75-th percentile (Q3) *and* at least
    #                   1 s so that tiny datasets do not cause absurdisms.
    #   • strategic   – between 0.8×median .. 1.5×median (roughly
    #                   “noticeable but not disruptive”) *and* before an
    #                   important technical term **or** a new sentence.
    #
    # When there are fewer than 8 pauses (very short answers) the quartile
    # estimation becomes unstable.  In that case we fall back to a secondary
    # heuristic based on words-per-minute (WPM) ‑ the previous, simpler
    # implementation – so we never fully lose coverage.
    # ------------------------------------------------------------------

    import statistics

    pause_durations = [p["duration"] for p in pauses]

    # Helper: fallback WPM-scaled numbers ---------------------------------
    def _wpm_scaled_thresholds() -> tuple[float, float, float, float]:
        """Return (long_thr, rushed_thr, strategic_min, strategic_max)."""

        total_words = len(words)
        if words:
            total_time = words[-1]["end"] - words[0]["start"]
        else:
            total_time = 0.0

        if total_time <= 0:
            wpm = 120  # assume average rate
        else:
            wpm = (total_words / total_time) * 60

        scale = 120 / wpm if wpm > 0 else 1.0

        # Ensure sane defaults that hold across typical speaking rates (80‒180
        # WPM).  We intentionally keep the numbers aligned with the fixed
        # bounds used in the quartile-based branch so that callers see
        # consistent behaviour regardless of answer length.

        long_thr = 1.0 * scale if scale > 1 else 1.0  # minimum 1 s
        long_thr = min(long_thr, 3.0)  # never mark >3 s as acceptable

        rushed_thr = 0.1 * scale if scale < 1 else 0.1  # ≈100 ms @120 WPM
        rushed_thr = max(0.05, min(rushed_thr, 0.2))

        # Broaden strategic pause window – see detailed explanation further
        # below in the quartile–based branch.
        strat_min = 0.05
        strat_max = 2.5
        return long_thr, rushed_thr, strat_min, strat_max

    # Determine thresholds -------------------------------------------------
    if len(pause_durations) >= 8:
        # Use robust Tukey's five-number summary for larger datasets
        try:
            q1, q3 = statistics.quantiles(pause_durations, n=4)[0], statistics.quantiles(pause_durations, n=4)[2]
        except Exception:
            # Very unlikely, but keep the code safe
            print("WARNING analyze_pauses > Determine thresholds > statistics.quatiles > no quntiles found")
            q1 = q3 = None

        if q1 is not None and q3 is not None:
            median_p = statistics.median(pause_durations)

            # ------------------------------------------------------------------
            # Derive **robust** thresholds while keeping them within sensible
            # physiological limits.  Using the raw quartiles alone leads to
            # unrealistically small numbers for short samples (e.g. Q3 ≈ 0.7 s
            # → *anything* above 0.7 s would be flagged as a long pause).  We
            # therefore clamp the automatically derived values to a proven
            # lower / upper bound that reflects typical human speech patterns.
            # ------------------------------------------------------------------

            # 1.  Rushed – instead of the full first quartile we use **half**
            #     of Q1 which empirically separates *true* word-sandwiching
            #     (no audible gap at all) from legitimate quick pacing.  We
            #     still cap the lower bound at 20 ms to avoid classifying
            #     timestamp noise as rushed, and cap the upper bound at
            #     120 ms which is around the shortest silence most listeners
            #     reliably perceive.

            rushed_threshold = max(0.02, min(0.12, q1 * 0.5))

            # 2.  Long – a pause only becomes disruptive when it *clearly* sits
            #     outside the speaker's usual rhythm.  We therefore set the
            #     threshold to **max(1.5 s, 2×Q3)** so that isolated dramatic
            #     pauses of ~1.8 s (common in keynote-style delivery) are not
            #     penalised.  The cap of 3 s from the earlier version is
            #     retained implicitly because 1.5 s ≤ long_threshold ≤ 3 s in
            #     typical data.

            # Choose the larger of a fixed 2-second cut-off or three times the
            # 75th-percentile so that occasional dramatic pauses (~1.8 s) are
            # not marked as disruptive, while still flagging *truly* lengthy
            # gaps above ≈3 s.
            long_threshold = max(2.0, q3 * 3)

            # 3.  Strategic – a *good* pause varies greatly with speaking
            #     style and context.  Real-world recordings show useful pauses
            #     as short as ~50 ms up to a bit more than two seconds.  We
            #     therefore broaden the acceptance window so that pauses that
            #     were **explicitly** suggested by the LLM are not
            #     mis-classified just because they lie outside the narrow
            #     0.3-1.5 s range that was previously hard-coded.

            # Allow very brief emphasising hesitations (≥50 ms) and also
            # extended dramatic pauses (≤2.5 s) while still excluding
            # outliers that would almost certainly feel disruptive (>3 s).
            strategic_min = 0.05
            strategic_max = 2.5

            # Sanity printout useful during development / unit tests
            print("long_threshold, rushed_threshold, strategic_min, strategic_max")
            print(long_threshold, rushed_threshold, strategic_min, strategic_max)
        else:
            long_threshold, rushed_threshold, strategic_min, strategic_max = _wpm_scaled_thresholds()
    else:
        # Not enough data → revert to WPM-based scaling
        print("WARNING VERY SHORT ANSWER > shifting to _WPM_SCALED_THRESHOLD")
        long_threshold, rushed_threshold, strategic_min, strategic_max = _wpm_scaled_thresholds()

    long_pauses: List[Dict] = []
    rushed_pauses: List[Dict] = []
    strategic_pauses: List[Dict] = []
    

    for pause in pauses:
        i = pause["index"]  # index of the word *before* the pause

        # --------------------------------------------------------------
        # Determine basic categories via duration thresholds ----------
        # --------------------------------------------------------------
        # 3. Strategic – lies inside the *noticeable but not disruptive* band
        #    AND was explicitly suggested by the LLM.
        # Prioritise pauses that the LLM explicitly recommended.  In most
        # cases these will fall inside the *strategic* window.  However, when
        # the actual silence is either shorter or longer than the ideal
        # range we still want to classify it rather than silently discarding
        # the event.  Therefore we *only* short-circuit when the pause is a
        # genuine strategic one.  Otherwise we drop through to the generic
        # duration-based checks so that an overly long recommended pause is
        # still reported as "long" and an extremely brief one as "rushed".

        if (i + 1) in recommended_pause_indices and strategic_min <= pause["duration"] <= strategic_max:
            strategic_pauses.append(pause)
            continue

        if pause["duration"] > long_threshold:
            # 1. Long pause – noticeably disruptive
            long_pauses.append(pause)
            continue

        if pause["duration"] < rushed_threshold:
            # 2. Potentially rushed – only flag when not part of a recommended
            #    strategic break (the LLM might still suggest a very short
            #    hesitation before emphasis, especially in quick speech).
            if (i + 1) not in recommended_pause_indices:
                rushed_pauses.append(pause)
            continue
        

    # ------------------------------------------------------------------
    # 4. Build deterministic feedback (examples, distribution)  -----------
    feedback: Dict = {"overview": "", "details": [], "distribution": {}}

    templates = {
        "long": (
            long_pauses,
            "⚠️ Long pause ({duration:.1f}s) after '{before_word}' at {timestamp}: consider a short linking phrase to keep the flow.",
            f"{len(long_pauses)} overly long pauses (> {long_threshold:.2f}s)",
        ),
        "rushed": (
            rushed_pauses,
            "⚠️ Rushed transition ({duration:.1f}s) between '{before_word}' → '{after_word}' at {timestamp}: add a tiny pause so listeners can follow.",
            f"{len(rushed_pauses)} rushed transitions (< {rushed_threshold:.2f}s)",
        ),
        "strategic": (
            strategic_pauses,
            "✅ Good pause ({duration:.1f}s) before '{after_word}' at {timestamp}: nice emphasis.",
            f"{len(strategic_pauses)} well-placed strategic pauses",
        ),
    }

    for kind, (examples, template, summary) in templates.items():
        if not examples:
            continue
        # add up to two illustrative examples
        for ex in examples[:2]:
            # Provide human-readable timestamp at the start of the pause
            ex_with_time = {**ex, "timestamp": _format_ts(ex["start"])}
            feedback["details"].append(template.format(**ex_with_time))
        feedback["overview"] += (", " if feedback["overview"] else "") + summary

    total_pauses = len(pauses)
    if total_pauses:
        feedback["distribution"] = {
            "long": f"{len(long_pauses) / total_pauses:.1%}",
            "rushed": f"{len(rushed_pauses) / total_pauses:.1%}",
            "strategic": f"{len(strategic_pauses) / total_pauses:.1%}",
            "normal": f"{(total_pauses - len(long_pauses) - len(rushed_pauses) - len(strategic_pauses)) / total_pauses:.1%}",
        }
        
    

    if not feedback["overview"]:
        feedback["overview"] = "Good pause management overall"
        feedback["details"].append("✅ Pause patterns support clear communication")

    # ------------------------------------------------------------------
    # 5. Ask LLM for actionable feedback + score --------------------------
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Updated, more forgiving rubric ------------------------------------
    # ------------------------------------------------------------------
    #  The original thresholds were found to penalise natural-sounding,
    #  studio-quality samples that contain intentional dramatic pauses or
    #  micro-pauses produced by alignment jitter.  We now align the
    #  categories closer to real-world data:
    #
    #    •  “Long” pauses become disruptive only when they make up >10 % of
    #       all silences (instead of 5 %).
    #    •  “Rushed” transitions start to hurt intelligibility once they
    #       exceed ~15 % of pauses (instead of 10 %).
    #    •  Helpful “strategic” pauses are rewarded at a lower threshold of
    #       8 % so that concise answers can still hit the top score.
    #
    #  These numbers were calibrated against the curated sample set in
    #  pauses_input_samples where *eleven_pause_x.json* serves as a reference
    #  for near-ideal pacing.
    # ------------------------------------------------------------------

    rubric = (
        "### Pause Management Scoring Rubric (1‒5)\n"
        "5 – Excellent: long pauses ≤10 % AND rushed pauses ≤15 % AND strategic pauses ≥8 %.\n"
        "4 – Good: 10–15 % long OR 15–25 % rushed; strategic ≥6 %.\n"
        "3 – Fair: 15–20 % long OR 25–35 % rushed; strategic <6 %.\n"
        "2 – Poor: >20 % long OR >35 % rushed.\n"
        "1 – Very poor: long pauses >30 % OR rushed pauses >50 %.\n"
    )

    stats_for_prompt = (
        f"Long pauses : {feedback['distribution'].get('long', '0%')}\n"
        f"Rushed pauses: {feedback['distribution'].get('rushed', '0%')}\n"
        f"Strategic    : {feedback['distribution'].get('strategic', '0%')}\n"
    )

    coaching_prompt = (
        "You are an interview communication coach. Use **simple, everyday language** "
        "(aim for a grade-6 reading level). Your task:\n"
        "1. Evaluate the speaker's pauses based on the stats below.\n"
        "2. Give **actionable** advice. Cite the exact word(s) and the timestamp you are referring to in parentheses so users know where to improve "
        "(e.g., after 'model' 01:22).\n"
        "3. Assign a holistic score from 1-5 following the rubric.\n\n"
        f"{rubric}\n"
        "---\n"
        "STATISTICS\n"
        f"{stats_for_prompt}\n"
        "EXAMPLE ISSUES\n"
        + "\n".join(feedback["details"]) + "\n---\n"
        "Return a JSON object with exactly these keys: 'actionable_feedback' (string) and 'score' (integer)."
    )

    actionable_feedback = "Could not generate feedback – LLM error."
    score = 3
    try:
        llm_response_raw = call_llm(coaching_prompt)
        llm_json = extract_json_dict(llm_response_raw)
        actionable_feedback = llm_json.get("actionable_feedback", actionable_feedback)
        score = int(llm_json.get("score", score))
    except Exception as e:
        # If the LLM call fails or returns invalid JSON, fall back to heuristic scoring
        print("Falling to heuristic scoring ",e)
        long_pct = float(feedback["distribution"].get("long", "0%")[:-1])
        rushed_pct = float(feedback["distribution"].get("rushed", "0%")[:-1])
        strategic_pct = float(feedback["distribution"].get("strategic", "0%")[:-1])

        # Simple heuristic consistent with rubric
        if long_pct <= 10 and rushed_pct <= 15 and strategic_pct >= 8:
            score = 5
        elif (10 < long_pct <= 15) or (15 < rushed_pct <= 25):
            score = 4
        elif (15 < long_pct <= 20) or (25 < rushed_pct <= 35):
            score = 3
        elif long_pct > 30 or rushed_pct > 50:
            score = 1
        else:
            score = 2

        # Provide a simple feedback sentence with reference to the first detected issue
        first_ref = feedback["details"][0] if feedback["details"] else "your answer"
        actionable_feedback = (
            f"Try to slow down or add a tiny pause {first_ref}. "
            "A quick breath before key points (about half a second) will make ideas clearer."
        )

    # Attach to feedback dict for downstream consumers
    feedback["actionable_feedback"] = actionable_feedback
    feedback["score"] = score

    return feedback

# ---------------------------------------------------------------------------
# Demonstration block (executed only when run directly) ---------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from pathlib import Path
    from openai import OpenAI
    import json
    import os

    from dotenv import load_dotenv
    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def call_llm(prompt: str, system:str = None,model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages = [{"role":"system","content":system}]
        messages.append({"role": "user", "content": prompt})
        if model == "gpt-4o-mini" or model == "gpt-4o":
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error in call_llm func: {e}"
            
    def extract_json_dict(text: str):
        try:
            start = min(
                (text.index('{') if '{' in text else float('inf')),
                (text.index('[') if '[' in text else float('inf'))
            )
            end = max(
                (text.rindex('}') + 1 if '}' in text else -1),
                (text.rindex(']') + 1 if ']' in text else -1)
            )
            json_str = text[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(text)
            print(json_str)
            raise ValueError(f"Invalid JSON found: {e}")
        
    samples_dir = Path(__file__).with_suffix("").parent / "pauses_input_samples"
    
    json_files = sorted(samples_dir.glob("*.json"))

    if not json_files:
        raise SystemExit("No sample transcripts found in pauses_input_samples/.")

    for sample_path in json_files:
        print("########", sample_path)
        with sample_path.open() as f:
            asr_output = json.load(f)
        print(json.dumps(analyze_pauses(asr_output, call_llm, extract_json_dict), indent=2))
        print()