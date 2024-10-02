import pandas as pd
from openai import OpenAI
import os
client = OpenAI(api_key="API_KEY")
df = pd.read_csv('../final/q1_with_summaries.csv')
df['label'] = ""


df_filtered = df[df['document_weight_question_1'] != 'unweighted']
df = df_filtered

MODEL="gpt-4o"
CHECK_PROMPT = """ Given the provided claim, question, and summary of the text, determine the most appropriate label for the relationship between the claim and the summary, assuming the summary serves as an answer to the question. The possible labels are:

1. **Refuted**: The summary clearly contradicts the claim.
2. **Supported**: The summary clearly supports the claim.
3. **Not Enough Evidence**: The summary does not provide sufficient evidence to either support or refute the claim. If there are no information in the text it means its Not Enough Evidence.

Claim is : [[CLAIM]]
Question is : [[QUESTION]]
Summary is: [[SUMMARY]]
Based on this information, what label can you give?
Just provide final label.
"""
CHOOSING_BEST_PROMPT = """
Based on the provided claim, question, and summaries, determine which summary provides the most relevant information related to the claim and question.

Claim: [[CLAIM]]
Question: [[QUESTION]]

Summaries:
[[SUMMARIES]]

Please choose the summary that best addresses the claim and question by providing its index.
"""
JUDGEMENT_PROMPT = """
Based on the provided claim, question ans summary answer the question.
Claim: [[CLAIM]]
Question: [[QUESTION]]
Summary: [[SUMMARY]]
Answer:
"""

CODE_DEMO_STOP = '''Claim = Superdrag and Collective Soul are both rock bands.
To validate the above claim, we have asked the following questions: 
Question 1 = Is Superdrag a rock band?
Answer 1 = Yes
Can we know whether the claim is true or false now?
Prediction = No, we cannot know. 

Claim = Superdrag and Collective Soul are both rock bands.
To validate the above claim, we have asked the following questions: 
Question 1 = Is Superdrag a rock band?
Answer 1 = Yes
Question 2 = Is Collective Soul a rock band?
Answer 2 = Yes
Can we know whether the claim is true or false now?
Prediction = Yes, we can know.

Claim = Jimmy Garcia lost by unanimous decision to a professional boxer that challenged for the WBO lightweight title in 1995. 
To validate the above claim, we have asked the following questions:
Question 1 = Who is the professional boxer that challenged for the WBO lightweight title in 1995? 
Answer 1 = Orzubek Nazarov
Can we know whether the claim is true or false now?
Prediction = No, we cannot know.

Claim = Jimmy Garcia lost by unanimous decision to a professional boxer that challenged for the WBO lightweight title in 1995. 
To validate the above claim, we have asked the following questions:
Question 1 = Who is the professional boxer that challenged for the WBO lightweight title in 1995? 
Answer 1 = Orzubek Nazarov
Question 2 = Did Jimmy Garcia lose by unanimous decision to Orzubek Nazarov?
Can we know whether the claim is true or false now?
Prediction = Yes, we can know.

Claim = The Swan of Catania was taught by the Italian composer Giovanni Furno.
To validate the above claim, we have asked the following questions: 
Question 1 = What is the nationality of Giovanni Furno?
Answer 1 = Italian
Can we know whether the claim is true or false now?
Prediction = No, we cannot know.

Claim = Lars Onsager won the Nobel prize when he was 30 years old.
To validate the above claim, we have asked the following questions:  
Question 1 = When Lars Onsager won the Nobel prize?
Answer 1 = 1968
Can we know whether the claim is true or false now?
Prediction = No, we cannot know.

Claim = Smith worked on the series The Handmaid's Tale that is based on a novel by Margaret Atwood. 
To validate the above claim, we have asked the following questions:
Question 1 = Which novel The Handmaid's Tale is based on?
Answer 1 = Margaret Atwood
Can we know whether the claim is true or false now?
Prediction = No, we cannot know.

Claim = Smith worked on the series The Handmaid's Tale that is based on a novel by Margaret Atwood. 
To validate the above claim, we have asked the following questions:
Question 1 = Which novel The Handmaid's Tale is based on?
Answer 1 = Margaret Atwood
Question 2 = Did Smith work on the series The Handmaid's Tale?
Answer 2 = Yes
Can we know whether the claim is true or false now?
Prediction = Yes, we can know.

Claim = The first season of the series The Handmaid's Tale was released in 2017.
To validate the above claim, we have asked the following questions:
Question 1 = When was the first season of the series The Handmaid's Tale released?
Answer 1 = 2017
Can we know whether the claim is true or false now?
Prediction = Yes, we can know.

Claim = [[CLAIM]]
To validate the above claim, we have asked the following questions:
[[QA_CONTEXTS]]
Can we know whether the claim is true or false now?
Prediction = '''
CODE_DEMO_SUBSEQUENT = '''Task: to verify a claim, we need to ask a series of simple questions. Here the task is given a claim and previous questions generate the following question to ask. 
This question should be:

- Simple with a single subject-verb-object structure.
- Specific and directly related to the key aspect of the claim that needs validation.

Claim = Superdrag and Collective Soul are both rock bands.
To validate the above claim, we need to ask the following simple questions sequentially: 
Question 1 = Is Superdrag a rock band?
Answer 1 = Yes
Question 2 = Is Collective Soul a rock band?

Claim = Jimmy Garcia lost by unanimous decision to a professional boxer that challenged for the WBO lightweight title in 1995. 
To validate the above claim, we need to ask the following simple questions sequentially: 
Question 1 = Who is the professional boxer that challenged for the WBO lightweight title in 1995? 
Answer 1 = Orzubek Nazarov
Question 2 = Did Jimmy Garcia lose by unanimous decision to Orzubek Nazarov?

Claim = The Swan of Catania was taught by the Italian composer Giovanni Furno.
To validate the above claim, we need to ask the following simple questions sequentially: 
Question 1 = What is the nationality of Giovanni Furno?
Answer 1 = Italian
Question 2 = Who was taught by Giovanni Furno?

Claim = Smith worked on the series The Handmaid's Tale that is based on a novel by Margaret Atwood.
To validate the above claim, we need to ask the following simple questions sequentially:
Question 1 = Which novel The Handmaid's Tale is based on?
Answer 1 = Margaret Atwood
Question 2 = Who worked on the series The Handmaid's Tale?

Claim = The Potomac River runs along the neighborhood where Ashley Estates Kavanaugh's wedding was held.
To validate the above claim, we need to ask the following simple questions sequentially:
Question 1 = Where was Ashley Estates Kavanaugh's wedding held?
Answer 1 = Christ Church in Georgetown
Question 2 = Which river runs along the Christ Church in Georgetown?

Claim = Ulrich Walter's employer is headquartered in Cologne.
To validate the above claim, we need to ask the following simple questions sequentially:
Question 1 = Who is Ulrich Walter's employer?
Answer 1 = University of Cologne
Question 2 = Where is the University of Cologne headquartered?

Claim = Lars Onsager won the Nobel prize when he was 30 years old.
To validate the above claim, we need to ask the following simple questions sequentially: 
Question 1 = When Lars Onsager won the Nobel prize?
Answer 1 = 1968
Question 2 = When was Lars Onsager born?

Claim = [[CLAIM]]
To validate the above claim, we need to ask the following simple questions sequentially: 
[[QA_CONTEXTS]]'''

# generating labels for each summary
# keeping it all in one row after each summaries 


def check_label(claim, question, summary):
    PR_template = CHECK_PROMPT
    example_input = PR_template.replace('[[CLAIM]]', claim.strip())
    example_input = example_input.replace('[[QUESTION]]', question.strip())
    example_input = example_input.replace('[[SUMMARY]]', summary.strip())
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert annotator who assists in determining the relationship between a claim and a summary in the context of a given question. Your task is to label the summary as either Refuted, Supported, or Not Enough Evidence based on how it answers the question in relation to the claim. Just provide final label."},
            {"role": "user", "content": example_input}
        ]
    )
    return completion.choices[0].message.content

# Process each row in the DataFrame and add new columns for labels
for index, row in df.iterrows():
    print("GET THE LABEL: " + str(index))
    claim = row['claim']
    question = row['generated_question_1']
    summary = row['summary_1']
    label_column = 'label'
    label = check_label(claim, question, summary)
    df.at[index, label_column] = label
    
#     for i in range(1, 6):
#         summary_column = f'summary{i}'
#         label_column = f'label{i}'
        
#         summary = row[summary_column]
#         label = check_label(claim, question, summary)
        
#         df.at[index, label_column] = label

df.at[0, 'label'] =1

import pandas as pd

# Function to process grouped data and aggregate summaries and URLs based on labels
def process_group(group):
    supported_summaries = group[group['label'] == 'Supported']['summary_1'].tolist()
    supported_urls = group[group['label'] == 'Supported']['document_url_question_1'].tolist()
    refuted_summaries = group[group['label'] == 'Refuted']['summary_1'].tolist()
    refuted_urls = group[group['label'] == 'Refuted']['document_url_question_1'].tolist()
    not_enough_evidence_summaries = group[group['label'] == 'Not Enough Evidence']['summary_1'].tolist()
    not_enough_evidence_urls = group[group['label'] == 'Not Enough Evidence']['document_url_question_1'].tolist()
    
    # Assuming other relevant information is consistent within the same claim_id
    claim = group['claim'].iloc[0]
    claim_id = group['claim_id'].iloc[0]
    question = group['generated_question_1'].iloc[0]  # Replace with the correct column if needed
    claim_date = group['claim_date'].iloc[0]
    speaker = group['speaker'].iloc[0]
    reporting_source = group['reporting_source'].iloc[0]
    
    return pd.Series([claim, question, claim_date, speaker, reporting_source, 
                      supported_summaries, supported_urls, 
                      refuted_summaries, refuted_urls, 
                      not_enough_evidence_summaries, not_enough_evidence_urls])

# Group by 'claim_id' and apply the processing function
processed_df = df.groupby('claim_id').apply(process_group).reset_index()

# Rename columns for clarity
processed_df.columns = ['claim_id', 'claim', 'question', 'claim_date', 'speaker', 'reporting_source', 
                        'supported_summaries', 'supported_urls', 
                        'refuted_summaries', 'refuted_urls', 
                        'not_enough_evidence_summaries', 'not_enough_evidence_urls']

import pandas as pd
import re

# Imitate GPT call to always return the first index
def choose_best_summary(claim, question, summaries):
    if not summaries:
        return None, None

    summaries_text = "\n".join([f"[[{i}]] - {summary}" for i, summary in enumerate(summaries)])
    prompt = CHOOSING_BEST_PROMPT.replace("[[CLAIM]]", claim).replace("[[QUESTION]]", question).replace("[[SUMMARIES]]", summaries_text)
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert annotator who assists in determining best and most informative summary based on the provided claim, question and text."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = completion.choices[0].message.content.strip()
    
    # Extract the index using regex
    match = re.search(r'\[\[(\d+)\]\]', content)
    if not match:
        match = re.search(r'(\d+)', content)
    
    if match:
        best_summary_index = int(match.group(1))
        if best_summary_index < len(summaries):
            return summaries[best_summary_index], best_summary_index
        else:
            return summaries[0], 0
    else:
        return summaries[0], 0

# Function to apply the filtering logic and choose the best summary
def process_row(row):
    supported_summaries = row['supported_summaries'] if isinstance(row['supported_summaries'], list) else eval(row['supported_summaries'])
    supported_urls = row['supported_urls'] if isinstance(row['supported_urls'], list) else eval(row['supported_urls'])
    refuted_summaries = row['refuted_summaries'] if isinstance(row['refuted_summaries'], list) else eval(row['refuted_summaries'])
    refuted_urls = row['refuted_urls'] if isinstance(row['refuted_urls'], list) else eval(row['refuted_urls'])
    not_enough_evidence_summaries = row['not_enough_evidence_summaries'] if isinstance(row['not_enough_evidence_summaries'], list) else eval(row['not_enough_evidence_summaries'])
    not_enough_evidence_urls = row['not_enough_evidence_urls'] if isinstance(row['not_enough_evidence_urls'], list) else eval(row['not_enough_evidence_urls'])
    claim_id = row['claim_id']
    claim = row['claim']
    question = row['question']
    claim_date = row['claim_date']
    speaker = row['speaker']
    reporting_source = row['reporting_source']
    
    results = []

    if supported_summaries:
        if len(supported_summaries) == 1:
            best_supported_summary = supported_summaries[0]
            best_supported_url = supported_urls[0]
        else:
            best_supported_summary, best_supported_index = choose_best_summary(claim, question, supported_summaries)
            best_supported_url = supported_urls[best_supported_index]
        results.append({
            'claim_id' : claim_id,
            'claim': claim,
            'question': question,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'best_summary': best_supported_summary,
            'best_url': best_supported_url,
            'label': 'Supported'
        })
        
    if refuted_summaries:
        if len(refuted_summaries) == 1:
            best_refuted_summary = refuted_summaries[0]
            best_refuted_url = refuted_urls[0]
        else:
            best_refuted_summary, best_refuted_index = choose_best_summary(claim, question, refuted_summaries)
            best_refuted_url = refuted_urls[best_refuted_index]
        results.append({
            'claim_id' : claim_id,
            'claim': claim,
            'question': question,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'best_summary': best_refuted_summary,
            'best_url': best_refuted_url,
            'label': 'Refuted'
        })
        
    if not results and not_enough_evidence_summaries:
        if len(not_enough_evidence_summaries) == 1:
            best_not_enough_evidence_summary = not_enough_evidence_summaries[0]
            best_not_enough_evidence_url = not_enough_evidence_urls[0]
        else:
            best_not_enough_evidence_summary, best_not_enough_evidence_index = choose_best_summary(claim, question, not_enough_evidence_summaries)
            best_not_enough_evidence_url = not_enough_evidence_urls[best_not_enough_evidence_index]
        results.append({
            'claim_id' : claim_id,
            'claim': claim,
            'question': question,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'best_summary': best_not_enough_evidence_summary,
            'best_url': best_not_enough_evidence_url,
            'label': 'Not Enough Evidence'
        })

    return pd.DataFrame(results)

# Create an empty DataFrame to store the results
final_df = pd.DataFrame(columns=['claim_id','claim', 'question', 'claim_date', 'speaker', 'reporting_source', 'best_summary', 'best_url', 'label'])

# Apply the processing function to each row
for index, row in processed_df.iterrows():
    print("CHOOSE BEST SUMMARY: " + str(index))
    processed_row_df = process_row(row)
    final_df = pd.concat([final_df, processed_row_df], ignore_index=True)

# Display the final DataFrame



#getting judgement to the summary and question based on the summary

def get_judgement(claim, question, summary):
    judgement_input = JUDGEMENT_PROMPT.replace('[[CLAIM]]', claim.strip())
    judgement_input = judgement_input.replace('[[QUESTION]]', question.strip())
    judgement_input = judgement_input.replace('[[SUMMARY]]', summary.strip())
    
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "You are an expert annotator who assist in providing answer based on the provided information of claim, question and extracted summary of the information."},
        {"role": "user", "content":judgement_input}
      ]
    )
    
    answer = completion.choices[0].message.content
    return answer

# Add a new column for judgements
final_df['judgement'] = final_df.apply(lambda row: get_judgement(row['claim'], row['question'], row['best_summary']), axis=1)


def get_verification_status(claim, question, answer):
    qa_contexts_txt = f'Question 1 = {question}\nAnswer 1 = {answer}\n'
    example = CODE_DEMO_STOP.replace('[[CLAIM]]', claim.strip())
    example = example.replace('[[QA_CONTEXTS]]', qa_contexts_txt.strip())
    
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "You are an expert annotator who assist in telling are we having enough information to verify claim or not based on the provided claim, questions and answers. Answer only yes or no"},
        {"role": "user", "content":example}
      ]
    )
    can_we_continue = completion.choices[0].message.content
    return can_we_continue

# Add a new column for verification status
final_df['verification_status'] = final_df.apply(lambda row: get_verification_status(row['claim'], row['question'], row['judgement']), axis=1)

import re
import pandas as pd

# Function to generate a follow-up question
def generate_followup_question(claim, context):
    qa_contexts_txt = '\n'.join([f'Question {i+1} = {context[i][0]}\nAnswer {i+1} = {context[i][1]}' for i in range(len(context))])
    example_followup = CODE_DEMO_SUBSEQUENT.replace('[[CLAIM]]', claim.strip())
    example_followup = example_followup.replace('[[QA_CONTEXTS]]', qa_contexts_txt.strip())
    
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "You are an expert annotator who assists in generating new questions that would be required to validate the claim, based on the claim, previous questions, and answers. Return only the question."},
        {"role": "user", "content": example_followup}
      ] 
    )
    followup_question = completion.choices[0].message.content.strip()
    return followup_question

# Create a new DataFrame to store the results
new_rows = []

for index, row in final_df.iterrows():
    print("GENERATE FOLLOW UP QUESTION: " + str(index))
    if not re.search(r'\byes\b', row['verification_status'], re.IGNORECASE):
        claim = row['claim']
        
        context = {
            "generated question 1": row['question'],
            "judgement 1": row['judgement'],
            "summary 1": row['best_summary'],
            "label 1": row['label'],
            "url 1": row['best_url']
        }
        context_list = [(f"{key}", f"{value}") for key, value in context.items()]
        new_question = generate_followup_question(claim, context_list)
        
        # Clean the new question
        new_question = re.sub(r'^Question\s*\d+\s*=\s*', '', new_question).strip()
        
        new_row = {
            'claim_id': row['claim_id'],
            'claim': claim,
            'claim_date': row['claim_date'],
            'speaker': row['speaker'],
            'reporting_source': row['reporting_source'],
            'generated_question_1': row['question'],
            'judgement_1': row['judgement'],
            'summary_1': row['best_summary'],
            'label_1': row['label'],
            'url_1': row['best_url'],
            'context': context,
            'generated_question_2': new_question
        }
        
        new_rows.append(new_row)

# Convert the list of new rows to a DataFrame
new_df = pd.DataFrame(new_rows)


new_df.to_csv('../final/q2_samples_dev.csv', index = False)

remaining_rows = []

for index, row in final_df.iterrows():
    if re.search(r'\byes\b', row['verification_status'], re.IGNORECASE):
        remaining_row = {
            'claim_id': row['claim_id'],
            'claim': row['claim'],
            'claim_date': row['claim_date'],
            'speaker': row['speaker'],
            'reporting_source': row['reporting_source'],
            'generated_question_1': row['question'],
            'summary_1': row['best_summary'],
            'label_1': row['label'],
            'document_url_question_1': row['best_url'],
            'judgement': row['judgement'].replace("\n", " ").strip(),
        }
        
        remaining_rows.append(remaining_row)
remaining_df = pd.DataFrame(remaining_rows)

remaining_df.to_csv('../final/out_csvs/q1_enough_generated.csv', index  = False)