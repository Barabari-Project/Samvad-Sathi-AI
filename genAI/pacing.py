def calculate_pace_metrics(words):
    """
    Calculates detailed pace metrics from a list of word timestamps.

    Args:
        words (list): List of dicts, each with 'start', 'end', and 'word' keys representing word timings.

    Returns:
        dict: {
            'avg_wpm': float,         # Average words per minute
            'too_slow_pct': float,    # Percentage of time speaking too slow (<105 WPM)
            'ideal_pct': float,       # Percentage of time at ideal pace (105-170 WPM)
            'too_fast_pct': float,    # Percentage of time speaking too fast (>170 WPM)
            'segments': list          # List of segments with pace classification and text
        }
        or None if insufficient data.
    """
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
    """Generate pacing feedback and a numerical score.

    Args:
        input_dict (dict): Transcript dict as produced by Whisper (expects
            a "words" key with word-level timestamps).

    Returns:
        dict: {
            "feedback": str,   # Human-readable feedback paragraph
            "score": float     # 0-100 pacing score (higher is better)
        }
    """
    if input_dict.get('words_timestamp',None) is not None:
        input_dict = input_dict['words_timestamp']
        
    words = input_dict.get('words', [])
    
    # preprocess
    # Remove elements without 'start' or 'end' keys
    words = [w for w in words if 'start' in w and 'end' in w]
    if len(words)<1:
        return {
            "feedback": "len(words) less than 1",
            "score": -1
        }
    # Calculate metrics and segments
    result = calculate_pace_metrics(words)
    if not result:
        return "Insufficient data for pace analysis"
    
    # Prepare feedback text
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
    
    # ------------------------------------------------------------
    # Scoring rubric
    # ------------------------------------------------------------
    # We evaluate pacing on two dimensions:
    #   1. Pace consistency – percentage of time spent in the ideal
    #      window (105-170 WPM). Worth 60 points.
    #        pace_consistency_score = ideal_pct * 0.6  (0-60)
    #   2. Average speed accuracy – how close the *average* WPM is
    #      to the recommended 120-150 WPM band. Worth 40 points.
    #         deviation = distance, in WPM, from the nearest bound
    #         accuracy_score = max(0, 40 - 2 * deviation)
    #         (we lose 2 points for every WPM outside the band)
    #   Total possible = 100. We clip the final score to [0, 100].

    # Calculate score components
    # 1. Pace consistency
    pace_consistency_score = result['ideal_pct'] * 0.6  # 0-60

    # 2. Average speed accuracy
    recommended_min, recommended_max = 120, 150
    avg_wpm = result['avg_wpm']
    if recommended_min <= avg_wpm <= recommended_max:
        accuracy_score = 40.0
    else:
        # deviation outside the recommended range
        if avg_wpm < recommended_min:
            deviation = recommended_min - avg_wpm
        else:
            deviation = avg_wpm - recommended_max
        accuracy_score = max(0.0, 40.0 - 2.0 * deviation)

    total_score = pace_consistency_score + accuracy_score
    # Ensure the score is between 0 and 100
    total_score = max(0.0, min(100.0, total_score))
    total_score /= 20
    total_score = round(total_score,1)
    
    return {
        "feedback": feedback,
        "score": total_score
    }

# {
#   "feedback": "**Overall Assessment:**\nYour average speaking rate is currently at 108.1 words per minute (WPM), which is below the ideal range for interviews of 120-150 WPM. This may impact the clarity and engagement of your responses. However, you have segments where you're closer to the ideal pace, which highlights your potential for improvement in this area.\n\n**Strengths:**\nYou've effectively utilized an ideal speaking pace in several segments, such as from **0:07 to 0:18**, where you clearly articulated your points about the ODIA language and its data limitations. This segment flows well and showcases your ability to convey complex information at a comfortable pace. Additionally, sections like **1:18 to 1:21** and **1:41 to 1:51** reflect thoughtful clarity, contributing positively to your overall communication.\n\n**Areas for Improvement:**\n1. **Slow Segments:** About **24.3% of your speech was slow**, which can cause listeners to lose focus. For instance, the segment from **0:24 to 0:32** included filler phrases like \"um\" and \"it had just wasn't working good,\" which can interrupt your train of thought. \n\n   **Techniques to Speak More Concisely:**\n   - **Reduce Filler Words:** Instead of using \"um\" and \"so,\" practice taking brief pauses. This can help you gather your thoughts without using fillers, enhancing the perceived professionalism of your delivery.\n   - **Pre-plan Key Points:** Structure your thoughts before speaking. Knowing the main points you wish to convey can minimize hesitations and keep your speech flowing smoothly. For example, practice summarizing the key challenges of your project in a single, clear statement.\n\n2. **Fast Segments:** On the contrary, the segment from **1:51 to 1:57** where you spoke quickly might have caused your audience to miss important information. Fast speech can indicate nervousness or rushing through your thoughts, which may detract from the impact of your message. \n\n   **Pausing and Breathing Techniques:**\n   - **Include Strategic Pauses:** After important points, take a brief pause to emphasize your message. For example, after stating \"I learned a lot but still like next time we can do better,\" pause to let the information settle with your audience before proceeding.\n   - **Practice Controlled Breathing:** Before speaking, take a few deep breaths which can help calm nerves, allowing for a more measured delivery. Employing this technique can help maintain a steady pace throughout your speech.\n\n**Actionable Steps:**\n1. Record yourself practicing responses, focusing on eliminating unnecessary fillers while working to maintain a steady pace.\n2. Participate in mock interviews or speaking exercises where you can receive feedback on both speed and clarity.\n3. Consider using pacing tools or software to analyze and adjust your speaking rate effectively.\n\nRemember, the key to effective communication in interviews is clarity and engagement. You're already demonstrating the ability to articulate your thoughts well; with a bit of practice on pacing, you'll enhance your overall delivery. Keep up the hard work, and don’t hesitate to experiment with these techniques in your next practice session!"
# }

# {
#   "feedback": "Quantitative Feedback:\nWords Per Minute (WPM): Your average pace: 108.1 WPM\nBenchmarking: Aim for 120-150 WPM in interviews\n\nPace Range Classification:\n- Too Slow: Your pace was slow 25.2% of the time\n- Ideal: You spoke at ideal pace for 47.7% of the time\n- Too Fast: Your pace exceeded 170 WPM for 0.9% of the time\n\nDetailed Pace Segments:\n\nToo slow segments:\n- [00:02 - 00:04]: so um one\n- [00:05 - 00:06]: was like\n- [00:24 - 00:32]: all and um it had just wasn't working good so um we\n- [01:00 - 01:06]: you know what I mean and so um yeah so\n- [01:07 - 01:12]: the model itself was like tough it's\n- [01:21 - 01:22]: tha and\n- [01:28 - 01:33]: accumulation and stuff and processed data in\n\nIdeal segments:\n- [00:07 - 00:18]: ODIA language you know it had like way less data than others so model was like kind of struggling to learn it properly I'm\n- [00:19 - 00:23]: you know the loss was and all\n- [00:33 - 00:34]: we tried to\n- [00:35 - 00:37]: like fix that by making batches\n- [00:39 - 00:42]: like you know make sure each batch\n- [00:46 - 00:50]: so like even if ODIA ka data kam tha\n- [00:51 - 00:53]: still came in training I\n- [00:54 - 01:00]: it helped kind of but not fully cause data hi kam tha\n- [01:18 - 01:21]: memory ka kaafi issue de raha tha\n- [01:23 - 01:24]: be limited tha\n- [01:25 - 01:27]: so um we did\n- [01:34 - 01:38]: like um small parts so it doesn't\n- [01:41 - 01:51]: like haan it was kind of difficult but I mean we did jugaad and somehow trained it and yeah I\n\nToo fast segments:\n- [01:51 - 01:57]: I learned a lot but still like next time we can do better you know what I mean\n"
# }


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
        # print(asr_output)
        print(json.dumps(provide_pace_feedback(asr_output), indent=2))
        print()