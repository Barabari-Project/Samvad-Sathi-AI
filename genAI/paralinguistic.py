def calculate_pace_metrics(words):
    # Handle invalid words and zero-duration words
    valid_words = []
    for w in words:
        start, end = w['start'], w['end']
        if end <= start:
            end = start + 0.01  # Add minimal duration
        valid_words.append({'start': start, 'end': end, 'word': w['word']})
    
    if not valid_words:
        return None
    
    # Calculate average WPM
    total_words = len(valid_words)
    first_start = min(w['start'] for w in valid_words)
    last_end = max(w['end'] for w in valid_words)
    total_time = last_end - first_start
    if total_time <= 0:
        return None
    
    avg_wpm = (total_words / total_time) * 60

    # Initialize counters for pace classification
    too_slow_sec = 0.0
    ideal_sec = 0.0
    too_fast_sec = 0.0
    total_windows = 0
    
    # Initialize segment tracking
    segments = []
    current_start = first_start
    current_label = None
    
    # Use 5-second sliding windows with 1-second step
    current = first_start
    while current <= last_end - 5:
        window_start = current
        window_end = current + 5
        window_duration = 5.0
        
        # Count words in this window
        word_count = 0
        for w in valid_words:
            # Check if word overlaps with window
            if w['start'] < window_end and w['end'] > window_start:
                word_count += 1
        
        # Calculate WPM for this window
        wpm = (word_count / window_duration) * 60
        
        # Classify pace
        if wpm < 105:
            label = 'too_slow'
            too_slow_sec += 1
        elif 105 <= wpm <= 170:
            label = 'ideal'
            ideal_sec += 1
        elif wpm > 170:
            label = 'too_fast'
            too_fast_sec += 1
        
        # Track segments
        if label != current_label:
            if current_label:  # Finalize previous segment
                segments.append({
                    'start': current_start,
                    'end': current,
                    'label': current_label,
                    'text': get_text_in_interval(valid_words, current_start, current)
                })
            current_start = current
            current_label = label
        
        total_windows += 1
        current += 1  # Move to next window
    
    # Finalize last segment
    if current_label:
        segments.append({
            'start': current_start,
            'end': last_end,
            'label': current_label,
            'text': get_text_in_interval(valid_words, current_start, last_end)
        })
    
    # Calculate percentages
    if total_windows > 0:
        too_slow_pct = (too_slow_sec / total_windows) * 100
        ideal_pct = (ideal_sec / total_windows) * 100
        too_fast_pct = (too_fast_sec / total_windows) * 100
    else:
        too_slow_pct = ideal_pct = too_fast_pct = 0
    
    return {
        'avg_wpm': round(avg_wpm, 1),
        'too_slow_pct': round(too_slow_pct, 1),
        'ideal_pct': round(ideal_pct, 1),
        'too_fast_pct': round(too_fast_pct, 1),
        'segments': segments
    }

def get_text_in_interval(words, start_time, end_time):
    """Extract text from words overlapping with the time interval"""
    words_in_interval = []
    for w in words:
        if w['end'] > start_time and w['start'] < end_time:
            words_in_interval.append(w)
    
    # Sort by start time and extract text
    words_in_interval.sort(key=lambda x: x['start'])
    return " ".join(w['word'] for w in words_in_interval)

def format_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def provide_pace_feedback(input_dict):
    words = input_dict.get('words', [])
    
    # Calculate metrics and segments
    result = calculate_pace_metrics(words)
    if not result:
        return "Insufficient data for pace analysis"
    
    # Prepare feedback
    feedback = "Quantitative Feedback:\n"
    feedback += f"Words Per Minute (WPM): Your average pace: {result['avg_wpm']} WPM\n"
    feedback += "Benchmarking: Aim for 120-150 WPM in interviews\n\n"
    
    feedback += "Pace Range Classification:\n"
    feedback += f"- Too Slow: Your pace was slow {result['too_slow_pct']}% of the time\n"
    feedback += f"- Ideal: You spoke at ideal pace for {result['ideal_pct']}% of the time\n"
    feedback += f"- Too Fast: Your pace exceeded 170 WPM for {result['too_fast_pct']}% of the time\n\n"
    
    # Add detailed segments
    feedback += "Detailed Pace Segments:\n"
    
    # Group segments by type
    segment_types = {
        'too_slow': [],
        'ideal': [],
        'too_fast': []
    }
    
    for seg in result['segments']:
        if seg['label'] in segment_types:
            segment_types[seg['label']].append(seg)
    
    # Format each segment type
    for label, segments in segment_types.items():
        if not segments:
            continue
            
        feedback += f"\n{label.capitalize().replace('_', ' ')} segments:\n"
        for seg in segments:
            start_time = format_time(seg['start'])
            end_time = format_time(seg['end'])
            feedback += f"- [{start_time} - {end_time}]: {seg['text']}\n"
    
    return feedback

# {
#   "feedback": "**Overall Assessment:**\nYour average speaking rate is currently at 108.1 words per minute (WPM), which is below the ideal range for interviews of 120-150 WPM. This may impact the clarity and engagement of your responses. However, you have segments where you're closer to the ideal pace, which highlights your potential for improvement in this area.\n\n**Strengths:**\nYou've effectively utilized an ideal speaking pace in several segments, such as from **0:07 to 0:18**, where you clearly articulated your points about the ODIA language and its data limitations. This segment flows well and showcases your ability to convey complex information at a comfortable pace. Additionally, sections like **1:18 to 1:21** and **1:41 to 1:51** reflect thoughtful clarity, contributing positively to your overall communication.\n\n**Areas for Improvement:**\n1. **Slow Segments:** About **24.3% of your speech was slow**, which can cause listeners to lose focus. For instance, the segment from **0:24 to 0:32** included filler phrases like \"um\" and \"it had just wasn't working good,\" which can interrupt your train of thought. \n\n   **Techniques to Speak More Concisely:**\n   - **Reduce Filler Words:** Instead of using \"um\" and \"so,\" practice taking brief pauses. This can help you gather your thoughts without using fillers, enhancing the perceived professionalism of your delivery.\n   - **Pre-plan Key Points:** Structure your thoughts before speaking. Knowing the main points you wish to convey can minimize hesitations and keep your speech flowing smoothly. For example, practice summarizing the key challenges of your project in a single, clear statement.\n\n2. **Fast Segments:** On the contrary, the segment from **1:51 to 1:57** where you spoke quickly might have caused your audience to miss important information. Fast speech can indicate nervousness or rushing through your thoughts, which may detract from the impact of your message. \n\n   **Pausing and Breathing Techniques:**\n   - **Include Strategic Pauses:** After important points, take a brief pause to emphasize your message. For example, after stating \"I learned a lot but still like next time we can do better,\" pause to let the information settle with your audience before proceeding.\n   - **Practice Controlled Breathing:** Before speaking, take a few deep breaths which can help calm nerves, allowing for a more measured delivery. Employing this technique can help maintain a steady pace throughout your speech.\n\n**Actionable Steps:**\n1. Record yourself practicing responses, focusing on eliminating unnecessary fillers while working to maintain a steady pace.\n2. Participate in mock interviews or speaking exercises where you can receive feedback on both speed and clarity.\n3. Consider using pacing tools or software to analyze and adjust your speaking rate effectively.\n\nRemember, the key to effective communication in interviews is clarity and engagement. You're already demonstrating the ability to articulate your thoughts well; with a bit of practice on pacing, you'll enhance your overall delivery. Keep up the hard work, and don’t hesitate to experiment with these techniques in your next practice session!"
# }

# {
#   "feedback": "Quantitative Feedback:\nWords Per Minute (WPM): Your average pace: 108.1 WPM\nBenchmarking: Aim for 120-150 WPM in interviews\n\nPace Range Classification:\n- Too Slow: Your pace was slow 25.2% of the time\n- Ideal: You spoke at ideal pace for 47.7% of the time\n- Too Fast: Your pace exceeded 170 WPM for 0.9% of the time\n\nDetailed Pace Segments:\n\nToo slow segments:\n- [00:02 - 00:04]: so um one\n- [00:05 - 00:06]: was like\n- [00:24 - 00:32]: all and um it had just wasn't working good so um we\n- [01:00 - 01:06]: you know what I mean and so um yeah so\n- [01:07 - 01:12]: the model itself was like tough it's\n- [01:21 - 01:22]: tha and\n- [01:28 - 01:33]: accumulation and stuff and processed data in\n\nIdeal segments:\n- [00:07 - 00:18]: ODIA language you know it had like way less data than others so model was like kind of struggling to learn it properly I'm\n- [00:19 - 00:23]: you know the loss was and all\n- [00:33 - 00:34]: we tried to\n- [00:35 - 00:37]: like fix that by making batches\n- [00:39 - 00:42]: like you know make sure each batch\n- [00:46 - 00:50]: so like even if ODIA ka data kam tha\n- [00:51 - 00:53]: still came in training I\n- [00:54 - 01:00]: it helped kind of but not fully cause data hi kam tha\n- [01:18 - 01:21]: memory ka kaafi issue de raha tha\n- [01:23 - 01:24]: be limited tha\n- [01:25 - 01:27]: so um we did\n- [01:34 - 01:38]: like um small parts so it doesn't\n- [01:41 - 01:51]: like haan it was kind of difficult but I mean we did jugaad and somehow trained it and yeah I\n\nToo fast segments:\n- [01:51 - 01:57]: I learned a lot but still like next time we can do better you know what I mean\n"
# }


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