

import json

def analyze_pauses(asr_output,call_llm):
    """
    Analyzes pauses in speech using ASR word timings and LLM context
    Returns structured feedback with timestamps and improvement suggestions
    """
    # Step 1: Extract word timings and calculate pauses
    words = asr_output["words"]
    pauses = []
    for i in range(len(words) - 1):
        pause_duration = words[i+1]["start"] - words[i]["end"]
        if pause_duration > 0:  # Only consider positive pauses (ignore overlaps)
            pauses.append({
                "index": i,
                "start": words[i]["end"],
                "end": words[i+1]["start"],
                "duration": pause_duration,
                "before_word": words[i]["word"],
                "after_word": words[i+1]["word"]
            })

    # Step 2: Generate context-rich transcript using LLM
    word_list = [w["word"] for w in words]
    llm_prompt = f"""
    Analyze this interview transcript. Perform:
    1. Add punctuation at natural boundaries
    2. Identify disfluencies (fillers, repetitions)
    3. Extract important technical terms
    4. Mark sentence boundaries

    Return JSON with:
    - "punctuated_text": string with punctuation
    - "words": list of dicts with keys: 
        "index" (original position), "word", "punctuation" (following symbol), 
        "tag" ("filler", "technical", "sentence_start")
    - "technical_terms": list of important terms

    Transcript: {word_list}
    """
    
    try:
        llm_output = json.loads(call_llm(llm_prompt))
    except Exception as e:
        # Fallback if LLM fails
        llm_output = {
            "words": [{"index": i, "word": w, "punctuation": "", "tag": ""} 
                      for i, w in enumerate(word_list)]
        }

    # Create word tag lookup
    word_tags = {w["index"]: w for w in llm_output.get("words", [])}
    
    # Step 3: Categorize pauses with contextual analysis
    long_pauses = []
    rushed_pauses = []
    strategic_pauses = []
    
    for pause in pauses:
        i = pause["index"]
        next_word_tag = word_tags.get(i+1, {})
        
        # Long pause detection (>3s)
        if pause["duration"] > 3.0:
            long_pauses.append(pause)
            
        # Rushed speech detection (<0.2s at non-boundaries)
        elif pause["duration"] < 0.2:
            # Check if natural boundary or after filler
            current_tag = word_tags.get(i, {})
            if not (current_tag.get("punctuation") in {",", ".", "?", "!"} or 
                    current_tag.get("tag") == "filler"):
                rushed_pauses.append(pause)
                
        # Strategic pause detection (0.8-1.5s before key terms)
        elif 0.8 <= pause["duration"] <= 1.5:
            if "technical" in next_word_tag.get("tag", "") or "sentence_start" in next_word_tag.get("tag", ""):
                strategic_pauses.append(pause)

    # Step 4: Generate feedback report
    feedback = {"overview": "", "details": [], "distribution": {}}
    
    # Feedback templates
    feedback_types = {
        "long": {
            "examples": long_pauses,
            "template": "⚠️ Long pause ({duration:.1f}s) after '{before_word}': Consider using a bridge phrase like 'To elaborate...' or 'The key point is...'",
            "summary": f"{len(long_pauses)} excessively long pauses (>3s) disrupting flow"
        },
        "rushed": {
            "examples": rushed_pauses,
            "template": "⚠️ Rushed transition ({duration:.1f}s) between '{before_word}' → '{after_word}': Add brief breath before important terms",
            "summary": f"{len(rushed_pauses)} rushed transitions (<0.2s) making speech sound abrupt"
        },
        "strategic": {
            "examples": strategic_pauses,
            "template": "✅ Effective pause ({duration:.1f}s) before '{after_word}': Helps emphasize important concepts",
            "summary": f"{len(strategic_pauses)} well-placed strategic pauses for emphasis"
        }
    }
    
    # Build detailed feedback
    for f_type, data in feedback_types.items():
        if data["examples"]:
            # Add first 2 examples to details
            for ex in data["examples"][:2]:
                feedback["details"].append(data["template"].format(**ex))
            
            # Add to overview summary
            if feedback["overview"]:
                feedback["overview"] += ", " + data["summary"]
            else:
                feedback["overview"] = data["summary"]
    
    # Add distribution statistics
    total_pauses = len(pauses)
    if total_pauses > 0:
        feedback["distribution"] = {
            "long": f"{len(long_pauses)/total_pauses:.1%}",
            "rushed": f"{len(rushed_pauses)/total_pauses:.1%}",
            "strategic": f"{len(strategic_pauses)/total_pauses:.1%}",
            "normal": f"{(total_pauses - len(long_pauses) - len(rushed_pauses) - len(strategic_pauses))/total_pauses:.1%}"
        }
    
    # Final overview if no issues found
    if not feedback["overview"]:
        feedback["overview"] = "Good pause management overall"
        feedback["details"].append("✅ Pause patterns support clear communication")
    
    return feedback