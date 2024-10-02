import pandas as pd
import json
from collections import defaultdict
import glob

def merge_csvs_to_json(csv_folder_path, output_file_path):
    # Initialize a dictionary to hold the claims with claim_id as the key
    claims_dict = defaultdict(lambda: {'evidence': []})
    
    # Get all CSV files in the specified folder
    csv_files = glob.glob(f"{csv_folder_path}/*.csv")
    
    for file_path in csv_files:
        # Read each CSV file
        df = pd.read_csv(file_path)
        
        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            # Extract claim_id, claim, and pred_label
            claim_id = row['claim_id']
            claim = row['claim']
            judgement = row['judgement']
            # pred_label = row['label_1'] # Assuming the predicted label is the first label
            
            # Initialize the claim if it doesn't already exist
            if claim_id not in claims_dict:
                claims_dict[claim_id] = {}
                claims_dict[claim_id]['claim'] = claim
                claims_dict[claim_id]['claim_id'] = claim_id
                claims_dict[claim_id]['final_judgement'] = judgement
                # claims_dict[claim_id]['pred_label'] = pred_label
                claims_dict[claim_id]['evidence'] = []
            
            # Loop through the generated questions, judgements, summaries, labels, and URLs
            question_num = 1
            while f'generated_question_{question_num}' in row:
                question = row[f'generated_question_{question_num}']
                answer = row[f'summary_{question_num}']
                url = row[f'document_url_question_{question_num}']
                scraped_text = row[f'summary_{question_num}']  # Assuming summary_ is the scraped text
                
                claims_dict[claim_id]['evidence'].append({
                    'question': question,
                    'answer': answer,
                    'url': url,
                    'scraped_text': scraped_text
                })
                
                question_num += 1
    
    json_list = []
    for claim_id in claims_dict:
        json_list.append(claims_dict[claim_id])
    
    # Convert the dictionary to a sorted list
    json_list = sorted(json_list, key=lambda x: x['claim_id'])

    # Save the JSON list to a single JSON file
    with open(output_file_path, 'w') as f:
        json.dump(json_list, f, indent=4)

# Example usage
merge_csvs_to_json('../final/out_csvs/', '../final/output.json')
