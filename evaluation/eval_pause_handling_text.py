import os
import json
import re
import argparse
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

# Thresholds consistent with other text-based evaluators
turn_duration_threshold = 1
turn_num_words_threshold = 3


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
    pred_texts_dict: Dict[str, str] = {}
    with open(pred_text_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            # Try to get id first, if not available extract from audio_path
            if 'id' in data and data['id'] is not None:
                key = data['id']
            elif 'audio_path' in data:
                # Extract number from audio_path like "pred_wavs/synthetic_pause_handling_synthetic_pause_handling_0103.wav"
                audio_path = data['audio_path']
                # Extract the last sequence of digits before the file extension
                match = re.search(r'_(\d+)\.wav$', audio_path)
                if match:
                    # Convert to int and back to string to remove leading zeros
                    key = str(int(match.group(1)))
                else:
                    continue
            else:
                continue
            if key is None:
                continue
            # Remove special tags like <SPECIAL_12> from pred_text
            clean_pred_text = re.sub(r'<SPECIAL_\d+>', '', data['pred_text'])
            pred_texts_dict[key] = clean_pred_text
    return pred_texts_dict


def build_key_to_pause_file_map(data_dir: str) -> Dict[str, str]:
    """
    Find all pause.json files under data_dir and map:
      folder name (key) -> pause.json path
    """
    key_to_pause_path: Dict[str, str] = {}
    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store") or folder.endswith(".md"):
            continue
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        for file_i in os.listdir(folder_path):
            if file_i == "pause.json":
                key_to_pause_path[folder] = os.path.join(folder_path, file_i)
                break
    return key_to_pause_path


def get_pause_end_time(pause_file: str) -> float:
    """
    pause.json format example:
    [
        {
            "text": "[PAUSE]",
            "timestamp": [start, end]
        }
    ]
    We use the end timestamp as the boundary after the pause.
    """
    with open(pause_file, "r") as f:
        meta = json.load(f)
    return float(meta[0]["timestamp"][1])


def get_pause_times(pause_file: str) -> Tuple[float, float]:
    """
    Return (pause_start_time, pause_end_time) from pause.json.
    """
    with open(pause_file, "r") as f:
        meta = json.load(f)
    start = float(meta[0]["timestamp"][0])
    end = float(meta[0]["timestamp"][1])
    return start, end

def get_transcription_last_end_time(transcription_file: str) -> Optional[float]:
    """
    Return the end timestamp of the last word from transcription.json if available.
    transcription.json format: list of { "text": str, "timestamp": [start, end] }
    """
    try:
        with open(transcription_file, "r") as f:
            items = json.load(f)
        if not isinstance(items, list) or not items:
            return None
        last_item = items[-1]
        ts = last_item.get("timestamp")
        if isinstance(ts, list) and len(ts) >= 2:
            return float(ts[1])
    except Exception:
        return None
    return None

def eval_pause_handling_text(data_dir: str, pred_text_file: str) -> None:
    """
    Evaluate pause handling using text-based extraction from JSONL file.
    TOR definition for pause handling:
      - TOR = 1 if an agent onset occurs during the pause window, OR
        if an onset occurs within the latency threshold after the pause end.
      - Otherwise TOR = 0.
    """
    # Build maps
    pred_texts_dict = build_key_to_pred_text_map(pred_text_file)
    key_to_pause = build_key_to_pause_file_map(data_dir)

    print(f"Found {len(key_to_pause)} pause.json files")
    print(f"Found {len(pred_texts_dict)} pred_text entries")

    # Common keys between pred_text and pause.json dirs
    common_keys = sorted(set(pred_texts_dict.keys()) & set(key_to_pause.keys()), key=lambda x: int(x) if x.isdigit() else x)
    print(f"Common keys: {len(common_keys)}")

    take_turn_list: List[int] = []

    for key in tqdm(common_keys, desc="evaluate_pause_handling"):
        pause_file = key_to_pause[key]
        pred_text = pred_texts_dict[key]
        pause_start_time, pause_end_time = get_pause_times(pause_file)
        transcription_file = os.path.join(os.path.dirname(pause_file), "transcription.json")
        last_transcription_end_time = get_transcription_last_end_time(transcription_file)

        # Parse timestamps and look for first agent onset within
        # [pause_start_time, pause_end_time + turn_duration_threshold]
        segments = extract_timestamps_and_text(pred_text)

        TOR = 0
        if segments:
            window_start = pause_start_time
            window_end = pause_end_time + turn_duration_threshold

            start_index = None
            for idx, (ts, text) in enumerate(segments):
                if text and window_start <= ts <= window_end:
                    start_index = idx
                    break

            if start_index is not None:
                output_start_time = segments[start_index][0]
                onset_latency = max(0.0, output_start_time - pause_start_time)
                TOR = 1
            else:
                TOR = 0

        take_turn_list.append(TOR)

        # Per-sample debug prints
        print(f"Key: {key}")
        print(f"Pause start time: {pause_start_time:.3f}")
        print(f"Pause end time: {pause_end_time:.3f}")
        if last_transcription_end_time is not None:
            print(f"Transcription last word end time: {last_transcription_end_time:.3f}")
        else:
            print("Transcription last word end time: N/A")
        if segments and start_index is not None:
            print(f"Agent onset after pause: {output_start_time:.3f}")
            print(f"Onset latency: {onset_latency:.3f}")
            print(f"Within latency threshold after pause: {onset_latency <= turn_duration_threshold}")
        print(f"TOR (1 = agent spoke after pause): {TOR}")
        print("---")

    # Summary
    average_take_turn = sum(take_turn_list) / len(take_turn_list) if take_turn_list else 0.0
    print("---------------------------------------------------")
    print("[Result]")
    print("Average take turn: ", average_take_turn)
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pause handling using text-based extraction")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing per-sample pause.json")
    parser.add_argument("--pred_text_file", type=str, required=True, help="Path to JSONL file containing pred_text field")
    args = parser.parse_args()

    eval_pause_handling_text(args.data_dir, args.pred_text_file)


