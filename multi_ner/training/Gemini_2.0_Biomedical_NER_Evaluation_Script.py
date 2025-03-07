import os
import json
import argparse
import time
from typing import List, Dict, Tuple, Any
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
import google.generativeai as genai

# Configuration for Gemini API
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.0-flash"

# Initialize the Google Generative AI SDK
genai.configure(api_key=API_KEY)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Gemini 2.0-flash on biomedical NER tasks')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the evaluation data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the evaluation results')
    parser.add_argument('--system_prompt', type=str, required=True,
                        help='Path to the system prompt file for Gemini')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of examples to process in each batch')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum sequence length for processing')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate entity extraction results')
    return parser.parse_args()

def read_bio_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Read BIO-tagged data file and convert to a list of examples.
    Each example contains tokens and their gold labels.
    """
    examples = []
    current_tokens = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Empty line indicates the end of a sentence
            if not line:
                if current_tokens:
                    examples.append({
                        'tokens': current_tokens,
                        'labels': current_labels,
                        'text': ' '.join(current_tokens)
                    })
                    current_tokens = []
                    current_labels = []
                continue

            # Split line into token and BIO tag
            parts = line.split()
            if len(parts) >= 2:
                token, label = parts[0], parts[-1]
                current_tokens.append(token)
                current_labels.append(label)

    # Add the last example if it exists
    if current_tokens:
        examples.append({
            'tokens': current_tokens,
            'labels': current_labels,
            'text': ' '.join(current_tokens)
        })

    return examples

def convert_bio_to_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Convert BIO labels to entity spans.
    Returns a list of entity dictionaries with begin, end, mention, and type.
    """
    entities = []
    i = 0
    while i < len(labels):
        if labels[i].startswith('B'):
            entity_type = labels[i][2:]  # Remove 'B-' prefix
            start = i
            end = i

            # Find the end of this entity (all consecutive I- tags)
            i += 1
            while i < len(labels) and labels[i].startswith('I') and labels[i][2:] == entity_type:
                end = i
                i += 1

            # Extract the entity text and add to entities list
            entity_text = ' '.join(tokens[start:end+1])
            entities.append({
                'begin': start,
                'end': end,
                'mention': entity_text,
                'type': entity_type
            })
        else:
            i += 1

    return entities

def create_gemini_prompt(text: str) -> str:
    """Create a prompt for Gemini to extract biomedical named entities."""
    # No need for specific prompt construction as the system prompt will handle this
    # Just return the text for processing
    return text

def call_gemini_api(prompt: str, system_prompt_path: str) -> Dict[str, Any]:
    """
    Send a request to the Gemini API using Google Generative AI SDK and return the response.
    Reads the system prompt from a file.
    """
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    try:
        # Read the system prompt from file
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_instruction = f.read()

        # Get the model
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_instruction
        )

        # Configure generation parameters
        generation_config = genai.GenerationConfig(
            temperature=0,
            top_p=1,
            top_k=1,
            max_output_tokens=8192,
        )

        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Convert response to dictionary format for compatibility with the rest of the code
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": response.text}]
                    }
                }
            ]
        }

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": str(e)}

def extract_entities_from_response(response: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from the Gemini API response with the new format.
    """
    try:
        # Get the response text
        if "candidates" not in response or not response["candidates"]:
            return []

        response_text = response["candidates"][0]["content"]["parts"][0]["text"]

        # Try to parse the JSON directly from the response text
        try:
            # Clean up the response text to ensure it's valid JSON
            # Find where JSON starts (typically at the beginning or after code blocks)
            json_start = response_text.find("{")
            if json_start >= 0:
                response_text = response_text[json_start:]

            # Find where JSON ends (typically at the end or before additional commentary)
            json_end = response_text.rfind("}")
            if json_end >= 0:
                response_text = response_text[:json_end+1]

            # Parse the JSON
            result = json.loads(response_text)

            # Extract annotations
            if "annotations" in result:
                entities = []
                for annotation in result["annotations"]:
                    # Convert the new format to our internal format
                    if all(k in annotation for k in ["mention", "type", "begin", "end"]):
                        entities.append({
                            "mention": annotation["mention"],
                            "type": annotation["type"],
                            "begin": annotation["begin"],
                            "end": annotation["end"] - 1  # Convert exclusive end to inclusive end
                        })
                return entities
            return []

        except json.JSONDecodeError:
            # If direct parsing fails, use regex to extract JSON
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            match = re.search(json_pattern, response_text)
            if match:
                try:
                    result = json.loads(match.group(0))
                    if "annotations" in result:
                        entities = []
                        for annotation in result["annotations"]:
                            if all(k in annotation for k in ["mention", "type", "begin", "end"]):
                                entities.append({
                                    "mention": annotation["mention"],
                                    "type": annotation["type"],
                                    "begin": annotation["begin"],
                                    "end": annotation["end"] - 1  # Convert exclusive end to inclusive end
                                })
                        return entities
                except Exception:
                    pass

            # If all else fails, look for individual annotations
            entities = []
            pattern = r'"mention"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"\s*,\s*"begin"\s*:\s*(\d+)\s*,\s*"end"\s*:\s*(\d+)'
            matches = re.finditer(pattern, response_text)

            for match in matches:
                entities.append({
                    "mention": match.group(1),
                    "type": match.group(2),
                    "begin": int(match.group(3)),
                    "end": int(match.group(4)) - 1  # Convert exclusive end to inclusive end
                })

            return entities

    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []

def align_token_to_char_positions(tokens: List[str], text: str) -> List[Tuple[int, int]]:
    """
    Align token positions to character positions in the original text.
    Returns a list of (start, end) character positions for each token.
    """
    positions = []
    start = 0
    for token in tokens:
        # Find the token in the text, starting from the current position
        token_start = text.find(token, start)
        if token_start == -1:
            # If exact token not found, try with flexible whitespace
            token_clean = token.strip()
            token_start = text.find(token_clean, start)
            if token_start == -1:
                # If still not found, use approximate matching
                token_start = start

        token_end = token_start + len(token) - 1
        positions.append((token_start, token_end))
        start = token_end + 1

    return positions

def convert_char_entities_to_token_labels(entities: List[Dict[str, Any]],
                                         token_positions: List[Tuple[int, int]],
                                         num_tokens: int) -> List[str]:
    """
    Convert character-based entity mentions to token-level BIO labels.
    """
    labels = ['O'] * num_tokens

    for entity in entities:
        entity_begin = entity['begin']
        entity_end = entity['end']
        entity_type = entity['type']

        # Find the tokens that overlap with this entity
        for i, (token_start, token_end) in enumerate(token_positions):
            # Check if token overlaps with the entity
            if (token_start <= entity_end and token_end >= entity_begin):
                # Determine if this is the beginning or inside of the entity
                if i == 0 or token_positions[i-1][1] < entity_begin:
                    labels[i] = f'B-{entity_type}'
                else:
                    labels[i] = f'I-{entity_type}'

    return labels

def evaluate(examples: List[Dict[str, Any]], system_prompt_path: str, batch_size: int = 1,
           output_dir: str = None) -> Dict[str, Any]:
    """
    Evaluate Gemini on the given examples and return metrics.
    If output_dir is provided, save intermediate results.
    """
    all_true_labels = []
    all_pred_labels = []

    # For intermediate results - organized by example
    gold_entities_by_example = []
    predicted_entities_by_example = []

    for i in tqdm(range(0, len(examples), batch_size)):
        batch = examples[i:i+batch_size]

        for example_idx, example in enumerate(batch):
            # Get the gold entities and text
            tokens = example['tokens']
            gold_labels = example['labels']
            text = example['text']

            # Convert gold BIO tags to entity spans
            gold_entities = convert_bio_to_entities(tokens, gold_labels)

            # Save gold entities for this example
            gold_entities_by_example.append({
                'example_id': i + example_idx,
                'source_text': text,
                'entities': gold_entities
            })

            # Call Gemini API with just the text (system prompt handles the rest)
            prompt  = create_gemini_prompt(text)
            response = call_gemini_api(prompt, system_prompt_path)

            # Extract predicted entities
            pred_entities = extract_entities_from_response(response, text)

            # Save predicted entities with the raw response
            predicted_entities_by_example.append({
                'example_id': i + example_idx,
                'source_text': text,
                'entities': pred_entities,
                'raw_api_response': response.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            })

            # Map character positions to token positions
            token_positions = align_token_to_char_positions(tokens, text)

            # Convert predicted entities to BIO labels
            pred_labels = convert_char_entities_to_token_labels(
                pred_entities, token_positions, len(tokens))

            all_true_labels.append(gold_labels)
            all_pred_labels.append(pred_labels)

            # Avoid rate limiting
            time.sleep(0.5)

    # Save intermediate results if output directory is provided
    if output_dir:
        gold_entities_path = os.path.join(output_dir, "gold_entities.json")
        pred_entities_path = os.path.join(output_dir, "predicted_entities.json")

        save_intermediate_results(gold_entities_by_example, gold_entities_path)
        save_intermediate_results(predicted_entities_by_example, pred_entities_path)

    # Calculate metrics
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    report = classification_report(all_true_labels, all_pred_labels, output_dict=True)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

def save_intermediate_results(data: List[Dict[str, Any]], file_path: str):
    """
    Save intermediate results to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved intermediate results to {file_path}")

def main():
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the evaluation data
    data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.txt')]

    all_results = {}

    for file_name in data_files:
        print(f"Processing {file_name}...")
        file_path = os.path.join(args.data_dir, file_name)

        # Read the data
        examples = read_bio_data(file_path)
        print(f"Loaded {len(examples)} examples from {file_name}")

        # Evaluate and save intermediate results
        file_output_dir = os.path.join(args.output_dir, os.path.splitext(file_name)[0])
        os.makedirs(file_output_dir, exist_ok=True)

        metrics = evaluate(
            examples,
            args.system_prompt,
            args.batch_size,
            output_dir=file_output_dir
        )

        # Save results for this file
        all_results[file_name] = metrics

        # Save detailed results for this file
        file_results_path = os.path.join(args.output_dir, f"{file_name}_results.json")
        with open(file_results_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Results for {file_name}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print("-----------------------------------")

    # Calculate and save aggregate results
    aggregate_precision = np.mean([res['precision'] for res in all_results.values()])
    aggregate_recall = np.mean([res['recall'] for res in all_results.values()])
    aggregate_f1 = np.mean([res['f1'] for res in all_results.values()])

    aggregate_results = {
        'precision': float(aggregate_precision),
        'recall': float(aggregate_recall),
        'f1': float(aggregate_f1),
        'file_results': all_results
    }

    # Save aggregate results
    aggregate_results_path = os.path.join(args.output_dir, "aggregate_results.json")
    with open(aggregate_results_path, 'w') as f:
        json.dump(aggregate_results, f, indent=2)

    print("Aggregate Results:")
    print(f"Precision: {aggregate_precision:.4f}")
    print(f"Recall: {aggregate_recall:.4f}")
    print(f"F1: {aggregate_f1:.4f}")

if __name__ == "__main__":
    main()
