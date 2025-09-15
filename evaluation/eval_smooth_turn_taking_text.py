import os
import json
import re
import argparse
from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3


def remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s\[\]]", "", text)


def round_to_quarter(number):
    return round(number * 4) / 4


def extract_key_from_audio_path(audio_path):
    """Extract key from audio path like 'pred_wavs/candor_105.wav' -> '105'"""
    import re
    # Extract number from the filename
    match = re.search(r'(\d+)\.wav$', audio_path)
    if match:
        return match.group(1)
    return None


def extract_timestamps_and_text(pred_text):
    """
    Extract timestamps and text from pred_text field.
    Format: <|0.23|>some text<|0.45|>more text
    Returns: list of (timestamp, text) tuples
    """
    # Pattern to match <|timestamp|>text
    pattern = r'<\|([0-9.]+)\|>([^<]*)'
    matches = re.findall(pattern, pred_text)
    
    timestamps_and_text = []
    for timestamp_str, text in matches:
        try:
            timestamp = float(timestamp_str)
            timestamps_and_text.append((timestamp, text.strip()))
        except ValueError:
            continue
    
    return timestamps_and_text


def count_words_after_first_timestamp(timestamps_and_text):
    """
    Count words in all text segments after the first timestamp.
    """
    timestamps_and_text = timestamps_and_text[0]
    if len(timestamps_and_text) <= 1:
        return 0

    # Get all text after the first timestamp
    text_after_first = ""
    for text in timestamps_and_text[1:]:
        text_after_first += text + " "
    
    # Count words (split by whitespace and filter empty strings)
    words = [word for word in text_after_first.split() if word.strip()]
    return len(words)


def eval_smooth_turn_taking_text(data_dir, pred_text_file):
    """
    Evaluate smooth turn taking using text-based extraction from JSONL file.
    
    Args:
        data_dir: Directory containing turn_taking.json files
        pred_text_file: Path to JSONL file containing pred_text field
    """
    # Read pred_text data from JSONL file and create key->pred_text mapping
    pred_texts_dict = {}
    with open(pred_text_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'pred_text' in data and 'audio_path' in data:
                # Extract key from audio_path
                key = extract_key_from_audio_path(data['audio_path'])
                if key is not None:
                    pred_texts_dict[key] = data['pred_text']
                else:
                    print(f"Warning: Could not extract key from audio_path: {data['audio_path']}")
            else:
                missing_fields = []
                if 'pred_text' not in data:
                    missing_fields.append('pred_text')
                if 'audio_path' not in data:
                    missing_fields.append('audio_path')
                print(f"Warning: Missing fields {missing_fields} in line: {line.strip()}")
    
    # Find all turn_taking.json files and create key->file_path mapping
    audio_input_files_dict = {}
    for folder in os.listdir(data_dir):
        if folder.endswith(".DS_Store") or folder.endswith(".md"):
            continue
        
        # Use folder name as key
        key = folder
        
        for file_i in os.listdir(os.path.join(data_dir, folder)):
            if file_i.endswith("turn_taking.json"):
                audio_input_files_dict[key] = os.path.join(data_dir, folder, file_i)
                break  # Only take the first turn_taking.json file found
    
    take_turn_list = []
    latency_list = []
    
    print(f"Found {len(audio_input_files_dict)} turn_taking.json files")
    print(f"Found {len(pred_texts_dict)} pred_text entries")
    
    # Find common keys
    common_keys = set(pred_texts_dict.keys()) & set(audio_input_files_dict.keys())
    print(f"Common keys: {len(common_keys)}")
    
    for key in tqdm(sorted(common_keys), desc="evaluate"):
        audio_input_file = audio_input_files_dict[key]
        pred_text = pred_texts_dict[key]
        
        # Get input turn end time
        if not os.path.exists(audio_input_file):
            raise FileNotFoundError(f"Required file '{audio_input_file}' not found.")

        with open(audio_input_file, "r") as f:
            input_turn = json.load(f)
        input_end_time = input_turn[0]["timestamp"][0]

        TOR = None
        latency = None
        
        # Extract timestamps and text from pred_text
        timestamps_and_text = extract_timestamps_and_text(pred_text)
        
        # if no timestamps found, means model does not take turn
        if len(timestamps_and_text) == 0:
            TOR = 0
        else:
            output_start_time = timestamps_and_text[0][0]  # Onset timestamp
            # Count words after first timestamp
            num_words_after_first = count_words_after_first_timestamp(timestamps_and_text)
            if num_words_after_first <= turn_num_words_threshold:
                TOR = 0
            else:
                TOR = 1
                latency = output_start_time - input_end_time

        take_turn_list.append(TOR)
        if TOR == 1:
            if latency < 0:
                latency = 0
            latency_list.append(latency)

        print(f"Key: {key}")
        print(f"File: {os.path.basename(audio_input_file)}")
        print(f"User offset time: {input_end_time:.2f}")
        if len(timestamps_and_text) > 0:
            print(f"Agent onset time: {output_start_time:.2f}")
        print(f"the TOR is {TOR}")
        print(f"the latency is {latency}")
        print("---")

    # Check there is no negative latency
    for i in latency_list:
        if i < 0:
            print(i)

    average_take_turn = sum(take_turn_list) / len(take_turn_list)
    average_latency = sum(latency_list) / len(latency_list)

    print("---------------------------------------------------")
    print("[Result]")
    print("Average take turn: {:.2f}%".format(average_take_turn * 100))
    print("Average latency: {:.2f} ms".format(average_latency * 1000))
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate smooth turn taking using text-based extraction")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing turn_taking.json files")
    parser.add_argument("--pred_text_file", type=str, required=True,
                       help="Path to JSONL file containing pred_text field")
    args = parser.parse_args()

    eval_smooth_turn_taking_text(args.data_dir, args.pred_text_file)
