"""Batch-run the pause analysis on all JSON transcript samples in
`pauses_input_samples/` and pretty-print the results.

If an OpenAI key is available in the environment (`OPENAI_API_KEY`) the real
LLM will be used.  Otherwise the script falls back to a stub that skips the
LLM call so that it can still run inside restricted environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict


from pauses import analyze_pauses


# ---------------------------------------------------------------------------
# Helper: choose real or stubbed LLM call depending on env
# ---------------------------------------------------------------------------


def _real_llm(prompt: str) -> str:  # pragma: no cover – requires network
    """Call the OpenAI Chat Completion endpoint using the default model.

    Requires `OPENAI_API_KEY` to be present in the environment.  Falls back to
    `_stub_llm` if an exception occurs so that the script never crashes.
    """

    import os

    try:
        from openai import OpenAI
    except ImportError:
        return _stub_llm(prompt)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _stub_llm(prompt)

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _stub_llm(prompt)


def _stub_llm(prompt: str) -> str:  # noqa: D401 – simple stub
    """Return an empty JSON string so downstream parsing succeeds."""

    return "{}"


def _extract_json_dict(text: str) -> Dict:  # simplified version
    """Best-effort JSON extraction from an LLM response string."""

    import re

    match = re.search(r"[{\[][^{}\[\]]*[}\]]", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _choose_llm() -> Callable[[str], str]:
    """Return the real LLM caller when possible, otherwise the stub."""

    return _real_llm if os.getenv("OPENAI_API_KEY") else _stub_llm


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    samples_dir = Path(__file__).with_suffix("").parent / "pauses_input_samples"
    json_files = sorted(samples_dir.glob("*.json"))

    if not json_files:
        raise SystemExit("No sample transcripts found in pauses_input_samples/.")

    call_llm = _choose_llm()

    for sample_path in json_files:
        print("=" * 80)
        print(sample_path.name)
        print("-" * 80)

        with sample_path.open() as f:
            asr_output = json.load(f)

        result = analyze_pauses(asr_output, call_llm, _extract_json_dict)

        # Pretty-print a concise summary for quick inspection
        print("Overview:", result.get("overview"))
        print("Distribution:", result.get("distribution"))
        print("Score:", result.get("score"))
        print("Actionable feedback:\n", result.get("actionable_feedback"))
        print()


if __name__ == "__main__":  # pragma: no cover
    main()
