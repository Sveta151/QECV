import os
import json
import pandas as pd
import numpy as np
import csv
import faiss
from sentence_transformers import SentenceTransformer

# To avoid duplicate library errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys

CSV_FILE = sys.argv[1]
QUESTION_NUM = "0"

# Load pre-trained SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_chunks(line):
    """
    Load JSON-encoded line and concatenate its contents into a single string.
    """
    line = json.loads(line)
    return ' '.join(line)

def run_similarity(idx, question, claim_data):
    """
    Run similarity search for a given question and claim.
    
    Args:
        idx (int): Index for the JSON file and output files.
        question (str): The question to search for similar sentences.
        claim (str): The claim associated with the question.
    """
    claim = claim_data["claim"]
    data_list = []
    url_list = []
    websites_list = []

    try:
        # Load data from the specified JSON file
        with open(f"../output_dev/{idx}.json", "r") as f:
            for line in f:
                entry = json.loads(line)
                
                # skip the gold standard documents
                if entry.get("type") == "gold":
                    continue

                url = entry.get("url")
                outs = url.split("https://")
                test_urls = []
                for each in outs:
                    website = each.split("/")[0]
                    if website == "https:" or website == "":
                        continue
                    print(website)
                    if website not in test_urls: test_urls.append(website)
                    
                outs = url.split("http://")
                for each in outs:
                    website = each.split("/")[0]
                    if website == "https:" or website == "":
                        continue
                    print(website)
                    if website not in test_urls: test_urls.append(website)
                    
                # Extract and preprocess the relevant part of the JSON content
                line = line.strip().split("url2text")[1][3:-1]
                line = load_chunks(line)
                data_list.append(line)
                websites_list.append(test_urls)
                url_list.append(url)
    except: 
        print(str(idx) + " error!")
        return

    try:
        index = faiss.read_index(f'../embds/index_embds_{idx}')
    except:
        # Encode the loaded data using the SentenceTransformer model
        embds = model.encode(data_list)

        # Create a FAISS index and add the encoded data
        index = faiss.IndexFlatL2(embds.shape[1])
        index.add(embds)
        
        # Save the FAISS index
        faiss.write_index(index, f'../embds/index_embds_{idx}')

    gold_urls = np.load("../normal_and_gold_urls/gold_urls.npy", allow_pickle=True).item()
    normal_urls = np.load("../normal_and_gold_urls/normal_urls.npy", allow_pickle=True).item()

    def search(query, url_list, websites_list, data_list, gold_urls, normal_urls):
        """
        Perform a semantic search to find the most similar sentences to the query.
        
        Args:
            query (str): The query string to search for.
        
        Returns:
            List[str]: List of the most similar sentences from the data.
        """
        query_vector = model.encode([query])
        k = 20  # Number of top results to return
        top_k = index.search(query_vector, k)
        retrieved_dataset = [data_list[_id] for _id in top_k[1][0]]
        retrieved_websites = [websites_list[_id] for _id in top_k[1][0]]
        retrieved_urls = [url_list[_id] for _id in top_k[1][0]]
        retrieved_scores = top_k[0][0]
        gold_scores = []
        normal_scores = []
        max_gold_score = 0
        max_normal_score = 0
        original_scores = retrieved_scores.copy()
        for i in range(len(retrieved_websites)):
                gold_score = 0
                normal_score = 0
                for each in retrieved_websites[i]:
                    if each in gold_urls:
                        gold_score += gold_urls[each]
                    if each in normal_urls:
                        normal_score += normal_urls[each]
                if max_normal_score < normal_score:
                    max_normal_score = normal_score
                if max_gold_score < gold_score:
                    max_gold_score = gold_score
                gold_scores.append(gold_score)
                normal_scores.append(normal_score)
        if max_normal_score > 0:
            for i in range(len(normal_scores)):
                normal_scores[i] /= max_normal_score
                normal_scores[i] = (1 - normal_scores[i])*0.5 + 0.5
        if max_gold_score > 0:
            for i in range(len(gold_scores)):
                gold_scores[i] /= max_gold_score
                gold_scores[i] = (1 - gold_scores[i])*0.5
        for i in range(len(retrieved_scores)):
            if(gold_scores[i] != 0.5):
                retrieved_scores[i] *= gold_scores[i]
            elif(normal_scores[i] > 0):
                retrieved_scores[i] *= normal_scores[i]

        retrieved_dataset = np.array(retrieved_dataset)
        retrieved_urls = np.array(retrieved_urls)
        sorted_indices = np.argsort(retrieved_scores)
        weighted_retrieved_dataset = retrieved_dataset[sorted_indices]
        weighted_retrieved_urls = retrieved_urls[sorted_indices]
        weighted_retrieved_scores = retrieved_scores[sorted_indices]

        return weighted_retrieved_dataset[:5], weighted_retrieved_urls[:5], weighted_retrieved_scores[:5], retrieved_dataset[:5], retrieved_urls[:5], retrieved_scores[:5]
    
    # Get the top 5 similar sentences to the question
    weighted_retrieved_dataset, weighted_retrieved_urls, weighted_retrieved_scores, retrieved_dataset, retrieved_urls, retrieved_scores = search(question, url_list, websites_list, data_list, gold_urls, normal_urls)

    # Write the dictionary to a CSV file
    with open(f"../final/q" + QUESTION_NUM + f"_document_retrieval/similar_results_{idx}.csv", 'a', newline='') as csvfile:
        keys = list(claim_data.keys())
        fieldnames = keys + ["document_weight_question_" + QUESTION_NUM, "document_rank_question_" + QUESTION_NUM, "document_url_question_" + QUESTION_NUM, "document_question_" + QUESTION_NUM]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        claim_data = claim_data.to_dict()
        for i in range(len(weighted_retrieved_dataset)):
            try:
                claim_data["document_weight_question_" + QUESTION_NUM] = "weighted"
                claim_data["document_rank_question_" + QUESTION_NUM] = str(i+1)
                claim_data["document_url_question_" + QUESTION_NUM] = weighted_retrieved_urls[i]
                claim_data["document_question_" + QUESTION_NUM] = weighted_retrieved_dataset[i]
                writer.writerow(claim_data)
            except: continue


        for i in range(len(retrieved_dataset)):
            try:
                claim_data["document_weight_question_" + QUESTION_NUM] = "unweighted"
                claim_data["document_rank_question_" + QUESTION_NUM] = str(i+1)
                claim_data["document_url_question_" + QUESTION_NUM] = retrieved_urls[i]
                claim_data["document_question_" + QUESTION_NUM] = retrieved_dataset[i]
                writer.writerow(claim_data)
            except: continue

# Load the JSON data from a file
with open('../data_dev.json', 'r') as file:
    data = json.load(file)

for csv_file in ["../final/" + CSV_FILE]:
    # Load the CSV file containing questions and claims
    df = pd.read_csv(csv_file)
    df.rename(columns= {'generated_question_1': 'generated_question_0'}, inplace=True)
    for _, row in df.iterrows():
        claim_id = row['claim_id']
        row['generated_question_' + QUESTION_NUM] = 'Is it true that "' + row["claim"] + '"'
        print(str(claim_id) + ": "  + row['generated_question_' + QUESTION_NUM])
        question = row['generated_question_' + QUESTION_NUM]
        run_similarity(claim_id, question, row)