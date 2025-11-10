import os
import json
import argparse
from tqdm import tqdm


def extract_key_from_audio_path(audio_path):
    """Extract key from audio path like 'pred_wavs/candor_74.wav' -> '74'"""
    import re
    # Extract number from the filename
    match = re.search(r'(\d+)\.wav$', audio_path)
    if match:
        return match.group(1)
    return None


def eval_latency_eou(data_dir, jsonl_file):
    """
    Compute latency using EOU (End of Utterance) time from JSONL file.
    
    Args:
        data_dir: Directory containing turn_taking.json files
        jsonl_file: Path to JSONL file containing src_eou field
    """
    # Read EOU times from JSONL file and create key->eou_time mapping
    eou_times_dict = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if 'src_eou' in data and 'audio_path' in data:
                # Extract key from audio_path
                key = extract_key_from_audio_path(data['audio_path'])
                if key is not None:
                    # Parse comma-separated floating numbers and get the last one
                    src_eou_str = data['src_eou']
                    if src_eou_str.strip() == "":
                        eou_time = -1
                    else:
                        eou_values = [float(x.strip()) for x in src_eou_str.split(',')]
                        eou_time = eou_values[-1]  # Last number is the EOU time
                    eou_times_dict[key] = eou_time
                else:
                    print(f"Warning: Could not extract key from audio_path: {data['audio_path']}")
            else:
                missing_fields = []
                if 'src_eou' not in data:
                    missing_fields.append('src_eou')
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
    
    latency_list = []
    valid_pairs = 0
    skipped_negative_eou = 0
    negative_eou_count = 0
    
    print(f"Found {len(audio_input_files_dict)} turn_taking.json files")
    print(f"Found {len(eou_times_dict)} EOU times from JSONL file")
    
    # Find common keys
    common_keys = set(eou_times_dict.keys()) & set(audio_input_files_dict.keys())
    print(f"Common keys: {len(common_keys)}")
    
    for key in tqdm(sorted(common_keys), desc="Computing latency"):
        audio_input_file = audio_input_files_dict[key]
        eou_time = eou_times_dict[key]
        
        # Skip if EOU time is negative
        if eou_time < 0:
            negative_eou_count += 1
            skipped_negative_eou += 1
            print(f"Skipping key {key} - EOU time is negative: {eou_time}")
            continue
        
        # Get input turn end time
        if not os.path.exists(audio_input_file):
            print(f"Warning: File '{audio_input_file}' not found, skipping")
            continue
            
        try:
            with open(audio_input_file, "r") as f:
                input_turn = json.load(f)
            input_end_time = input_turn[0]["timestamp"][0]
            
            # Calculate latency: EOU time - input end time
            latency = eou_time - input_end_time
            if latency < 0:
                latency = 0
            
            latency_list.append(latency)
            valid_pairs += 1
            
            print(f"Key: {key}")
            print(f"File: {os.path.basename(audio_input_file)}")
            print(f"Input end time: {input_end_time}")
            print(f"EOU time: {eou_time}")
            print(f"Latency: {latency}")
            print("---")
            
        except Exception as e:
            print(f"Error processing {audio_input_file}: {e}")
            continue
    
    if len(latency_list) == 0:
        print("No valid latency calculations completed.")
        return
    
    # Calculate statistics
    average_latency = sum(latency_list) / len(latency_list)
    min_latency = min(latency_list)
    max_latency = max(latency_list)
    
    # Count negative latencies
    negative_count = sum(1 for x in latency_list if x < 0)
    
    # Calculate skip percentages
    total_common_keys = len(common_keys)
    skipped_percentage = (skipped_negative_eou / total_common_keys) * 100 if total_common_keys > 0 else 0
    negative_eou_percentage = (negative_eou_count / total_common_keys) * 100 if total_common_keys > 0 else 0
    
    print("=" * 50)
    print("[Results]")
    print(f"Total common keys: {total_common_keys}")
    print(f"Negative EOU times: {negative_eou_count} ({negative_eou_percentage:.2f}%)")
    print(f"Skipped due to negative EOU: {skipped_negative_eou} ({skipped_percentage:.2f}%)")
    print(f"Valid pairs processed: {valid_pairs}")
    print(f"Average latency: {average_latency:.4f}")
    print(f"Min latency: {min_latency:.4f}")
    print(f"Max latency: {max_latency:.4f}")
    print(f"Negative latencies: {negative_count}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute latency using EOU time from JSONL file")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Directory containing turn_taking.json files")
    parser.add_argument("--jsonl_file", type=str, required=True,
                       help="Path to JSONL file containing src_eou field")
    args = parser.parse_args()
    
    eval_latency_eou(args.data_dir, args.jsonl_file)
