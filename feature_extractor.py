import fitz
import json
import pandas as pd
from collections import defaultdict
import re
import os

def clean_text(text):
    """Removes extra spaces and special characters."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_features(pdf_path: str, json_path: str) -> pd.DataFrame:
    """
    Parses a PDF and its corresponding JSON to create a feature-rich DataFrame
    for training a machine learning model.
    """
    # Load ground truth
    with open(json_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    gt_map = {}
    gt_map[clean_text(ground_truth['title'])] = 'Title'
    for item in ground_truth['outline']:
        gt_map[clean_text(item['text'])] = item['level']

    # --- PDF Parsing ---
    doc = fitz.open(pdf_path)
    all_features = []

    # Get page dimensions and common font size for relative calculations
    page_widths = [page.rect.width for page in doc]

    all_text_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0: # Text block
                for line in block["lines"]:
                    # Consolidate spans into a single line
                    line_text = clean_text(" ".join([span['text'] for span in line['spans']]))
                    if not line_text:
                        continue

                    # Use properties from the first span as representative
                    first_span = line['spans'][0]
                    all_text_blocks.append({
                        'text': line_text,
                        'size': first_span['size'],
                        'font': first_span['font'],
                        'bold': 'bold' in first_span['font'].lower(),
                        'x0': line['bbox'][0],
                        'y0': line['bbox'][1],
                        'page_num': page_num,
                        'page_width': page_widths[page_num]
                    })

    # Calculate most common font size (likely body text)
    if all_text_blocks:
        font_size_counts = defaultdict(int)
        for block in all_text_blocks:
            font_size_counts[round(block['size'])] += 1
        body_font_size = max(font_size_counts, key=font_size_counts.get)
    else:
        body_font_size = 12 # Default

    # --- Feature Engineering ---
    for i, block in enumerate(all_text_blocks):
        # Determine label from ground truth
        label = gt_map.get(block['text'], 'Body')

        # Feature creation
        features = {
            'text': block['text'],
            'font_size': block['size'],
            'is_bold': block['bold'],
            'relative_font_size': block['size'] / body_font_size,
            'x_position_normalized': block['x0'] / block['page_width'],
            'vertical_spacing': block['y0'] - all_text_blocks[i-1]['y0'] if i > 0 else 0,
            'word_count': len(block['text'].split()),
            'char_count': len(block['text']),
            'is_all_caps': block['text'].isupper() and len(block['text']) > 1,
            'starts_with_numbering': bool(re.match(r'^\d+(\.\d+)*\s|\([a-zA-Z0-9]+\)|[A-Z]\.', block['text'])),
            'label': label
        }
        all_features.append(features)

    doc.close()
    return pd.DataFrame(all_features)


if __name__ == '__main__':
    # --- MODIFIED SCRIPT TO PROCESS ALL FILES ---

    # Define the directory where your files are located
    base_dir = '/content/'

    all_dfs = []

    # Loop through file numbers (e.g., from 01 to 10)
    for i in range(1, 6):
        file_num_str = f"{i:02d}" # Formats number as 01, 02, etc.
        pdf_path = os.path.join(base_dir, f'file{file_num_str}.pdf')
        json_path = os.path.join(base_dir, f'file{file_num_str}.json')

        # Check if both PDF and JSON files exist before processing
        if os.path.exists(pdf_path) and os.path.exists(json_path):
            print(f"Processing '{pdf_path}'...")
            try:
                # Extract features for the current file pair
                features_df = extract_features(pdf_path, json_path)
                if not features_df.empty:
                    all_dfs.append(features_df)
            except Exception as e:
                print(f"Could not process {pdf_path}. Error: {e}")

    # Combine all the extracted features into a single DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Save the final dataset to a single CSV file
        output_csv = 'training_data.csv'
        combined_df.to_csv(output_csv, index=False)

        print(f"\n✅ All files processed. Combined training data saved to '{output_csv}'.")
        print(f"Total labeled text lines extracted: {len(combined_df)}")
    else:
        print("No matching file pairs were found in the specified directory.")