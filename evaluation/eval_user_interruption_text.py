import os
import re
import json
import argparse
from typing import List, Tuple, Dict
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI

turn_duration_threshold = 1
turn_num_words_threshold = 3


def parse_output(data: str) -> Dict[str, object]:
    """
    Parse judge output in the format:
    Analysis: ...
    I would rate the AI's response as <int>
    """
    example = {}
    example_pattern = re.compile(r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL)
    for match in example_pattern.finditer(data):
        analysis = match.group(1).strip()
        rating = match.group(2).strip()
        example = {"analysis": analysis, "rating": int(rating)}
    return example


def extract_timestamps_and_text(pred_text: str) -> List[Tuple[float, str]]:
    """
    Extract timestamps and text from pred_text field.
    Format: <|0.23|>some text<|0.45|>more text
    Returns: list of (timestamp, text) tuples
    """
    pattern = r'<\|([0-9.]+)\|>([^<]*)'
    matches = re.findall(pattern, pred_text)

    timestamps_and_text: List[Tuple[float, str]] = []
    for timestamp_str, text in matches:
        try:
            timestamp = float(timestamp_str)
            timestamps_and_text.append((timestamp, text.strip()))
        except ValueError:
            continue
    return timestamps_and_text


def build_key_to_pred_text_map(pred_text_file: str) -> Dict[str, str]:
    """
    Read a JSONL file where each line contains at least:
      - audio_path: string
      - pred_text: string that includes timestamp markers
    Create a mapping from extracted key -> pred_text.
    """
    key_to_pred_text: Dict[str, str] = {}
    with open(pred_text_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            key = data["id"]
            if key is None:
                continue
            # Remove special tags like <SPECIAL_12> from pred_text
            clean_pred_text = re.sub(r"<SPECIAL_\d+>", "", data["pred_text"])
            key_to_pred_text[key] = clean_pred_text
    return key_to_pred_text


def build_key_to_interrupt_file_map(data_dir: str) -> Dict[str, str]:
    """
    Find all interrupt.json files under data_dir and map:
      folder name (key) -> interrupt.json path
    """
    key_to_interrupt_path: Dict[str, str] = {}
    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store") or folder.endswith(".md"):
            continue
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file_i in os.listdir(folder_path):
            if file_i == "interrupt.json":
                key_to_interrupt_path[folder] = os.path.join(folder_path, file_i)
                break
    return key_to_interrupt_path


def get_interrupt_end_time(interrupt_file: str) -> float:
    """
    interrupt.json format example:
      [{"context": "...", "interrupt": "...", "timestamp": [start, end]}]
    We use the end timestamp as the boundary after the interruption.
    """
    with open(interrupt_file, "r") as f:
        meta = json.load(f)
    return float(meta[0]["timestamp"][1])


def get_out_after_interrupt(pred_text: str, interrupt_end_time: float) -> str:
    """
    From pred_text timestamps, find the first segment that starts AFTER the interruption end.
    Then concatenate all text from that segment to the end.
    """
    segments = extract_timestamps_and_text(pred_text)
    if not segments:
        return ""

    start_index = None
    for idx, (ts, _) in enumerate(segments):
        if ts >= interrupt_end_time:
            start_index = idx
            break

    if start_index is None:
        return ""

    # Concatenate text from start_index onward
    out_text_parts: List[str] = [text for _, text in segments[start_index:]]
    return " ".join(part for part in out_text_parts if part).strip()


def eval_user_interruption_text(
    data_dir: str,
    pred_text_file: str,
    write_outputs: bool = False,
    llm_judge: str = None,
    openai_key: str = None,
    nv_oneapi_key: str = None,
    openai_key_file: str = None,
    verbose: bool = False,
) -> None:
    """
    For each sample where we have both pred_text (with timestamps) and interrupt.json,
    extract the agent's response text after the interruption and either print it
    or write it to <sample_dir>/out_after_interrupt.json depending on write_outputs.
    If an API key is provided, also obtain a GPT judge rating using the original user interruption template.
    """
    key_to_pred_text = build_key_to_pred_text_map(pred_text_file)
    key_to_interrupt = build_key_to_interrupt_file_map(data_dir)

    common_keys = sorted(set(key_to_pred_text.keys()) & set(key_to_interrupt.keys()), key=lambda x: int(x) if x.isdigit() else x)

    print(f"Found {len(key_to_interrupt)} interrupt.json files")
    print(f"Found {len(key_to_pred_text)} pred_text entries")
    print(f"Common keys: {len(common_keys)}")

    results: Dict[str, str] = {}
    take_turn_list: List[int] = []
    latency_list: List[float] = []
    score_list: List[int] = []
    # Per-sample metrics for optional printing
    per_sample_tor = {}
    per_sample_latency = {}
    per_sample_score = {}
    per_sample_parsed = {}
    # Ratings output directory: same directory as pred_text_file
    ratings_output_dir = os.path.dirname(pred_text_file)

    # Prepare optional judge client
    api_client = None
    # If no direct key provided, try reading OpenAI key from the provided file path
    if not openai_key and openai_key_file:
        try:
            if os.path.isfile(openai_key_file):
                with open(openai_key_file, "r") as f:
                    file_key = f.read().strip()
                if file_key:
                    openai_key = file_key
        except Exception:
            # Fail silently; will proceed without API client if still missing
            pass
    if openai_key:
        api_client = OpenAI(api_key=openai_key)
    elif nv_oneapi_key:
        api_client = AzureOpenAI(
            api_version="2025-02-01-preview",
            azure_endpoint="https://llm-proxy.perflab.nvidia.com",
            api_key=nv_oneapi_key,
        )

    # System prompt from original eval_user_interruption.py
    system_msg = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.

   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.

   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """

    for key in tqdm(common_keys, desc="compute_out_after_interrupt"):
        interrupt_file = key_to_interrupt[key]
        pred_text = key_to_pred_text[key]
        interrupt_end_time = get_interrupt_end_time(interrupt_file)

        # Load meta once for this key
        with open(interrupt_file, "r") as f:
            meta = json.load(f)
        in_interrupt_text = meta[0].get("interrupt", "")
        in_before_interrupt_text = meta[0].get("context", "")

        segments = extract_timestamps_and_text(pred_text)
        out_after_interrupt = get_out_after_interrupt(pred_text, interrupt_end_time)
        results[key] = out_after_interrupt

        if verbose:
            print(f"[{key}] User interrupting turn: {in_interrupt_text}")
            print(f"[{key}] AI response after interrupt: {out_after_interrupt}")

        # Compute TOR and latency following original script logic
        TOR = 0
        latency = None
        if segments:
            # Find first segment starting after interrupt end
            start_index = None
            for idx, (ts, _) in enumerate(segments):
                if ts >= interrupt_end_time:
                    start_index = idx
                    break

            if start_index is not None:
                output_start_time = segments[start_index][0]
                last_time = segments[-1][0]
                duration = max(0.0, last_time - output_start_time)

                # Count words from start_index onward
                words_text = " ".join(text for _, text in segments[start_index:] if text)
                num_words = len([w for w in words_text.split() if w.strip()])

                if duration < turn_duration_threshold:
                    if num_words <= turn_num_words_threshold:
                        TOR = 0
                    else:
                        TOR = 1
                        latency = output_start_time - interrupt_end_time
                else:
                    TOR = 1
                    latency = output_start_time - interrupt_end_time

        take_turn_list.append(TOR)
        if TOR == 1 and latency is not None:
            latency_list.append(latency if latency >= 0 else 0.0)
        # Track per-sample metrics
        per_sample_tor[key] = TOR
        if TOR == 1 and latency is not None:
            per_sample_latency[key] = latency if latency >= 0 else 0.0
        else:
            per_sample_latency[key] = None

        # Judge scoring (only when API client and the model took a turn and we have text)
        if api_client is not None and llm_judge and TOR == 1 and out_after_interrupt:
            user_msg = f"""
                - Contextual user turn: {in_before_interrupt_text}
                - User interrupting turn: {in_interrupt_text}
                - AI's response: {out_after_interrupt}
                """
            try:
                response = api_client.chat.completions.create(
                    model=llm_judge,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                prediction = response.choices[0].message.content
                parsed = parse_output((prediction or "") + "\n")
                if "rating" in parsed:
                    score = parsed["rating"]
                    score_list.append(score)
                    per_sample_score[key] = score
                    per_sample_parsed[key] = parsed
                    if verbose:
                        print(f"[{key}] GPT judge score: {score}; TOR: {TOR}; Latency: {latency if latency is not None else 'N/A'}; Num words: {num_words if 'num_words' in locals() else 'N/A'}; Duration: {duration if 'duration' in locals() else 'N/A'}")
            except Exception:
                # Fail silently per sample to not block bulk processing
                pass

        if write_outputs:
            sample_dir = os.path.join(data_dir, key)
            out_path = os.path.join(sample_dir, "out_after_interrupt.json")
            with open(out_path, "w") as f:
                json.dump({"out_after_interrupt": out_after_interrupt}, f)

        # Always write ratings for each sample to the same directory as pred_text_file
        rating_payload = {
            "TOR": per_sample_tor.get(key),
            "latency": per_sample_latency.get(key),
        }
        if key in per_sample_parsed:
            rating_payload.update({
                "analysis": per_sample_parsed[key].get("analysis"),
                "rating": per_sample_parsed[key].get("rating"),
                "gpt_score": per_sample_parsed[key].get("rating"),
            })
        else:
            # Ensure gpt_score exists even if no judge result
            rating_payload["gpt_score"] = None
        # Use per-key file naming to avoid overwrite
        per_key_rating_path = os.path.join(ratings_output_dir, f"{key}_rating.json")
        try:
            with open(per_key_rating_path, "w") as f:
                json.dump(rating_payload, f)
        except Exception:
            # Fail silently if directory is not writable
            pass

    # If not writing files, print a compact JSON per key for easy piping
    if not write_outputs:
        for key in common_keys:
            print(json.dumps({
                "key": key,
                "out_after_interrupt": results.get(key, ""),
                "TOR": per_sample_tor.get(key),
                "latency": per_sample_latency.get(key),
                "gpt_score": per_sample_score.get(key),
            }))

    # Always print final averages
    if take_turn_list:
        print("---------------------------------------------------")
        print("[Result]")
        if score_list:
            print("Average rating: ", sum(score_list) / len(score_list))
        else:
            print("Average rating: N/A")
        print("Average take turn: ", sum(take_turn_list) / len(take_turn_list))
        if latency_list:
            print("Average latency: ", sum(latency_list) / len(latency_list))
        else:
            print("Average latency: N/A")
        print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract agent response text after user interruption from pred_text JSONL")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with per-sample folders containing interrupt.json")
    parser.add_argument("--pred_text_file", type=str, required=True, help="JSONL containing 'audio_path' and 'pred_text' with timestamps")
    parser.add_argument("--write_outputs", action="store_true", help="Write per-sample out_after_interrupt.json files")
    parser.add_argument("--llm_judge", type=str, default=None, help="Judge model name, e.g., gpt-4o-mini-20240718")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--nv_oneapi_key", type=str, default=None, help="NVIDIA OneAPI key for AzureOpenAI-compatible endpoint")
    parser.add_argument("--openai_key_file", type=str, default="/lustre/fsw/portfolios/llmservice/users/kevinhu/HFCACHE/zh_openai_key.txt", help="Path to file containing OpenAI API key")
    parser.add_argument("--verbose", action="store_true", help="Print interrupt text, output text, and GPT score per sample")
    args = parser.parse_args()

    eval_user_interruption_text(
        args.data_dir,
        args.pred_text_file,
        args.write_outputs,
        llm_judge=args.llm_judge,
        openai_key=args.openai_key,
        nv_oneapi_key=args.nv_oneapi_key,
        openai_key_file=args.openai_key_file,
        verbose=args.verbose,
    )


