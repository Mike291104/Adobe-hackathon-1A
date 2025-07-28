# run_inference.py
import fitz
import json
import pandas as pd
import joblib
import re
from collections import defaultdict
import os
import glob

def clean_text(text):
    """Removes extra spaces and special characters."""
    return re.sub(r'\s+', ' ', text).strip()

def create_inference_features(pdf_path: str):
    """
    Extracts features from a new PDF for inference.
    """
    doc = fitz.open(pdf_path)
    all_features_list = []
    page_widths = [page.rect.width for page in doc]
    all_text_blocks = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block['type'] == 0:
                for line in block["lines"]:
                    line_text = clean_text(" ".join([span['text'] for span in line['spans']]))
                    if not line_text:
                        continue
                    first_span = line['spans'][0]
                    all_text_blocks.append({
                        'text': line_text, 'size': first_span['size'], 'font': first_span['font'],
                        'bold': 'bold' in first_span['font'].lower(), 'x0': line['bbox'][0],
                        'y0': line['bbox'][1], 'page_num': page_num + 1, 'page_width': page_widths[page_num]
                    })

    if not all_text_blocks:
        doc.close()
        return pd.DataFrame(), []

    font_size_counts = defaultdict(int)
    for block in all_text_blocks:
        font_size_counts[round(block['size'])] += 1
    body_font_size = max(font_size_counts, key=font_size_counts.get) if font_size_counts else 12

    for i, block in enumerate(all_text_blocks):
        features = {
            'font_size': block['size'], 'is_bold': block['bold'],
            'relative_font_size': block['size'] / body_font_size,
            'x_position_normalized': block['x0'] / block['page_width'],
            'vertical_spacing': block['y0'] - all_text_blocks[i-1]['y0'] if i > 0 else 0,
            'word_count': len(block['text'].split()), 'char_count': len(block['text']),
            'is_all_caps': block['text'].isupper() and len(block['text']) > 1,
            'starts_with_numbering': bool(re.match(r'^\d+(\.\d+)*\s|\([a-zA-Z0-9]+\)|[A-Z]\.', block['text'])),
        }
        all_features_list.append(features)
        
    doc.close()
    return pd.DataFrame(all_features_list), all_text_blocks

def run_prediction(pdf_path: str, model_path: str, output_json_path: str):
    """
    Runs the full inference pipeline on a single PDF.
    """
    saved_model = joblib.load(model_path)
    model = saved_model['model']
    label_encoder = saved_model['label_encoder']
    
    features_df, original_blocks = create_inference_features(pdf_path)
    
    if features_df.empty:
        print(f"Warning: No text found in {os.path.basename(pdf_path)}. Skipping.")
        return

    for col in features_df.select_dtypes(include=['bool']).columns:
        features_df[col] = features_df[col].astype(int)

    predictions_encoded = model.predict(features_df)
    predictions = label_encoder.inverse_transform(predictions_encoded)
    
    title = ""
    outline = []
    
    for i, pred in enumerate(predictions):
        if pred == 'Title':
            title = original_blocks[i]['text']
            break
            
    if not title:
        page1_blocks = [b for b in original_blocks if b['page_num'] == 1]
        if page1_blocks:
            max_size = max(b['size'] for b in page1_blocks)
            title = " ".join([b['text'] for b in page1_blocks if b['size'] == max_size])

    for i, pred in enumerate(predictions):
        if pred in ['H1', 'H2', 'H3']:
            outline.append({
                'level': pred,
                'text': original_blocks[i]['text'],
                'page': original_blocks[i]['page_num']
            })

    final_output = {
        'title': title,
        'outline': outline
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)
    
    print(f"✅ Successfully processed. Output saved to '{os.path.basename(output_json_path)}'.")


if __name__ == '__main__':
    input_dir = '/app/input'
    output_dir = '/app/output'
    # Assume the model is placed in the root of the app directory
    model_file = 'document_structure_model.joblib'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, '*.pdf'))
    
    if not os.path.exists(model_file):
        print(f"Error: Model file not found at '{model_file}'. Exiting.")
    elif not pdf_files:
        print(f"No PDF files found in '{input_dir}'. Exiting.")
    else:
        print(f"Found {len(pdf_files)} PDF(s) to process.")
    
        for pdf_path in pdf_files:
            print(f"--- Processing {os.path.basename(pdf_path)} ---")
            try:
                # [cite_start]Determine the output filename based on the input filename [cite: 105]
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = os.path.join(output_dir, f'{base_name}.json')
                
                # Run the prediction pipeline for the current file
                run_prediction(pdf_path, model_file, output_path)
                
            except Exception as e:
                print(f"Failed to process {os.path.basename(pdf_path)}. Error: {e}")

        print("\nBatch processing complete.")