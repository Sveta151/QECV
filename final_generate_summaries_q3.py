import pandas as pd
import os

# Directory containing the CSV files
directory = '../final/q3_document_retrieval'

# Initialize an empty list to store the DataFrames
dataframes = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.startswith('similar_results_') and filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(os.path.join(directory, filename))
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all the DataFrames in the list into a single DataFrame
merged_df = pd.concat(dataframes, ignore_index=True)

# Optionally, save the merged DataFrame to a new CSV file
merged_df.to_csv(directory + '/q3_merged_results.csv', index=False)

df_filtered = merged_df[merged_df['document_weight_question_3'] != 'unweighted']
merged_df = df_filtered

from openai import OpenAI
import pandas as pd
import json

client = OpenAI(api_key="API_KEY")

MAX_TOKEN_LIMIT = 128000

def chunk_document(document, max_chunk_size):
    chunks = []
    for i in range(0, len(document), max_chunk_size):
        chunks.append(document[i:i + max_chunk_size])
    return chunks

MODEL="gpt-4o"
def summarize_chunk(chunk, question):
    prompt = f"""
    You are a helpful assistant tasked with generating a summary based on the provided document and question. The summary should aim at answering the question. Use information provided ONLY in the document.

    Text: {chunk}

    Question: {question}

    Provide a brief summary of the text, focusing on information relevant to the question. If there is no relevant information, state that briefly.

    Response:
    """
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return completion.choices[0].message.content.strip()

def generate_summary(document, question):
    chunks = chunk_document(document, MAX_TOKEN_LIMIT)
    chunk_summaries = [summarize_chunk(chunk, question) for chunk in chunks]
    
    combined_summary = "\n".join(chunk_summaries)
    
    final_prompt = f"""
    Based on the following summarized information from a longer text, answer the question in 2-3 sentences. If the text contradicts the question, clearly state this contradiction. If there's no information directly relevant to the question, summarize the main points of the text that might be indirectly related.

    Summarized Information: {combined_summary}

    Question: {question}

    Response:
    """
    
    final_completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": final_prompt}
        ]
    )
    
    return final_completion.choices[0].message.content.strip()

results = []
for i in merged_df.index:
    print(i)
    question = merged_df['generated_question_3'][i]
    document = merged_df['document_question_3'][i]
    summary = generate_summary(document, question)
    results.append(summary)

merged_df['summary_3'] = results



merged_df.to_csv('../final/q3_with_summaries.csv', index = False)