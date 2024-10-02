import pandas as pd
from openai import OpenAI

client = OpenAI(api_key="API_KEY")
df = pd.read_csv('../final/q0_with_summaries.csv')

import re

# Assume merged_df is your DataFrame
df_columns = df.columns

def get_max_suffix(columns):
    max_suffix = 0
    for column in columns:
        match = re.search(r'(\d+)$', column)
        if match:
            suffix = int(match.group(1))
            if suffix > max_suffix:
                max_suffix = suffix
    return max_suffix

MAX_SUFFIX = get_max_suffix(df_columns)
df[f'label_{MAX_SUFFIX}'] = ""
df = df[df['claim_id'] != 'claim_id']
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
Based on the provided claim, questions and summaries, answer the questions in a single answer.
Claim: [[CLAIM]]
Question 1: [[QUESTION_1]]
Summary 1: [[SUMMARY_1]]
Question 2: [[QUESTION_2]]
Summary 2: [[SUMMARY_2]]
Question 3: [[QUESTION_3]]
Summary 3: [[SUMMARY_3]]
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
MODEL="gpt-4o"
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


for index, row in df.iterrows():
    print("GET THE LABEL: " + str(index))
    claim = row['claim']
    question = row[f'generated_question_{MAX_SUFFIX}']
    summary = row[f'summary_{MAX_SUFFIX}']
    label_column = f'label_{MAX_SUFFIX}'
    label = check_label(claim, question, summary)
    df.at[index, label_column] = label

unique_pairs = df[['claim_id', 'claim', 'claim_date', 'speaker', 'reporting_source', 'generated_question_1', 'document_url_question_1', 'summary_1', 'label_1', 'generated_question_2', 'document_url_question_2', 'summary_2', 'label_2', 'generated_question_3']].drop_duplicates()

unique_pairs['supported_summaries'] = None
unique_pairs['refuted_summaries'] = None
unique_pairs['not_enough_evidence_summaries'] = None
unique_pairs['supported_urls'] = None
unique_pairs['refuted_urls'] = None
unique_pairs['not_enough_evidence_urls'] = None

for idx, unique_pair in unique_pairs.iterrows():
    filtered_rows = df[
        (df['claim_id'] == unique_pair['claim_id']) &
        (df['claim'] == unique_pair['claim']) &
        (df['generated_question_1'] == unique_pair['generated_question_1']) &
        (df['document_url_question_1'] == unique_pair['document_url_question_1']) &
        (df['summary_1'] == unique_pair['summary_1']) &
        (df['label_1'] == unique_pair['label_1']) &
        (df['generated_question_2'] == unique_pair['generated_question_2']) &
        (df['document_url_question_2'] == unique_pair['document_url_question_2']) &
        (df['summary_2'] == unique_pair['summary_2']) &
        (df['label_2'] == unique_pair['label_2']) &
        (df['generated_question_3'] == unique_pair['generated_question_3'])
    ]
    # print(f'Rows for unique pair {idx}:')
    label_supported = filtered_rows[filtered_rows["label_3"] == "Supported"]["summary_3"]
    label_refuted = filtered_rows[filtered_rows["label_3"] == "Refuted"]["summary_3"]
    label_nee = filtered_rows[filtered_rows["label_3"] == "Not Enough Evidence"]["summary_3"]

    supported_url = filtered_rows[filtered_rows["label_3"] == "Supported"]["document_url_question_3"]
    refuted_url = filtered_rows[filtered_rows["label_3"] == "Refuted"]["document_url_question_3"]
    nee_url = filtered_rows[filtered_rows["label_3"] == "Not Enough Evidence"]["document_url_question_3"]

    unique_pairs = unique_pairs.copy()  # Make a copy to avoid SettingWithCopyWarning
    unique_pairs.at[idx, 'supported_summaries'] = list(label_supported)
    unique_pairs.at[idx, 'refuted_summaries'] = list(label_refuted)
    unique_pairs.at[idx, 'not_enough_evidence_summaries'] = list(label_nee)
    unique_pairs.at[idx, 'supported_urls'] = list(label_supported)
    unique_pairs.at[idx, 'refuted_urls'] = list(label_refuted)
    unique_pairs.at[idx, 'not_enough_evidence_urls'] = list(label_nee)

processed_df = unique_pairs

#finding best summary and saving question and answer pair
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
#             raise ValueError(f"Extracted index {best_summary_index} is out of range for summaries: {content}")
            return summaries[0],0
    else:
#         raise ValueError(f"Invalid index format in GPT response: {content}")
        return summaries[0],0


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
    question = row['generated_question_2']
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
            'claim_id': claim_id,
            'claim': claim,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'generated_question_1': row["generated_question_1"], 
            'document_url_question_1': row["document_url_question_1"], 
            'summary_1': row["summary_1"], 
            'label_1': row["label_1"],
            'generated_question_2': row["generated_question_2"], 
            'document_url_question_2': row["document_url_question_2"], 
            'summary_2': row["summary_2"], 
            'label_2': row["label_2"],
            'generated_question_3': question,
            'document_url_question_3': best_supported_url, 
            'summary_3': best_supported_summary,
            'label_3': 'Supported'
        })
        
    if refuted_summaries:
        if len(refuted_summaries) == 1:
            best_refuted_summary = refuted_summaries[0]
            best_refuted_url = refuted_urls[0]
        else:
            best_refuted_summary, best_refuted_index = choose_best_summary(claim, question, refuted_summaries)
            best_refuted_url = refuted_urls[best_refuted_index]
        results.append({
            'claim_id': claim_id,
            'claim': claim,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'generated_question_1': row["generated_question_1"], 
            'document_url_question_1': row["document_url_question_1"], 
            'summary_1': row["summary_1"], 
            'label_1': row["label_1"],
            'generated_question_2': row["generated_question_2"], 
            'document_url_question_2': row["document_url_question_2"], 
            'summary_2': row["summary_2"], 
            'label_2': row["label_2"],
            'generated_question_3': question,
            'document_url_question_3': best_refuted_url, 
            'summary_3': best_refuted_summary,
            'label_3': 'Refuted'
        })
        
    if not results and not_enough_evidence_summaries:
        if len(not_enough_evidence_summaries) == 1:
            best_not_enough_evidence_summary = not_enough_evidence_summaries[0]
            best_not_enough_evidence_url = not_enough_evidence_urls[0]
        else:
            best_not_enough_evidence_summary, best_not_enough_evidence_index = choose_best_summary(claim, question, not_enough_evidence_summaries)
            best_not_enough_evidence_url = not_enough_evidence_urls[best_not_enough_evidence_index]
        results.append({

            'claim_id': claim_id,
            'claim': claim,
            'claim_date': claim_date,
            'speaker': speaker,
            'reporting_source': reporting_source,
            'generated_question_1': row["generated_question_1"], 
            'document_url_question_1': row["document_url_question_1"], 
            'summary_1': row["summary_1"], 
            'label_1': row["label_1"],
            'generated_question_2': row["generated_question_2"], 
            'document_url_question_2': row["document_url_question_2"], 
            'summary_2': row["summary_2"], 
            'label_2': row["label_2"],
            'generated_question_3': question,
            'document_url_question_3': best_not_enough_evidence_url, 
            'summary_3': best_not_enough_evidence_summary,
            'label_3': 'Not Enough Evidence'
        })

    return pd.DataFrame(results)

# Create an empty DataFrame to store the results
final_df = pd.DataFrame(columns=['claim_id', 'claim', 'generated_question_1', 'document_url_question_1', 'summary_1', 'label_1', 'document_url_question_2', 'generated_question_2', 'summary_2', 'label_2', 'generated_question_3', 'summary_3', 'label_3'])

# Apply the processing function to each row
for index, row in processed_df.iterrows():
    print("CHOOSE BEST SUMMARY: " + str(index))
    processed_row_df = process_row(row)
    final_df = pd.concat([final_df, processed_row_df], ignore_index=True)



#getting judgement to the summary and question based on the summary

def get_judgement(claim, question_1, summary_1, question_2, summary_2, question_3, summary_3):
    judgement_input = JUDGEMENT_PROMPT.replace('[[CLAIM]]', claim.strip())
    judgement_input = judgement_input.replace('[[QUESTION_1]]', question_1.strip())
    judgement_input = judgement_input.replace('[[SUMMARY_2]]', summary_1.strip())
    judgement_input = judgement_input.replace('[[QUESTION_1]]', question_2.strip())
    judgement_input = judgement_input.replace('[[SUMMARY_2]]', summary_2.strip())
    judgement_input = judgement_input.replace('[[QUESTION_3]]', question_3.strip())
    judgement_input = judgement_input.replace('[[SUMMARY_3]]', summary_3.strip())
    
    completion = client.chat.completions.create(
      model=MODEL,
      messages=[
        {"role": "system", "content": "You are an expert annotator who assist in providing answer based on the provided information of claim, two questions and corresponding extracted summaries of the information."},
        {"role": "user", "content":judgement_input}
      ]
    )
    
    answer = completion.choices[0].message.content
    return answer

# Add a new column for judgements
final_df['judgement'] = final_df.apply(lambda row: get_judgement(row['claim'], row['generated_question_1'], row['summary_1'], row['generated_question_2'], row['summary_2'], row['generated_question_3'], row['summary_3']), axis=1)

remaining_rows = []

for index, row in final_df.iterrows():
    remaining_row = {
        'claim_id': row['claim_id'],
        'claim': row['claim'],
        'claim_date': row['claim_date'],
        'speaker': row['speaker'],
        'reporting_source': row['reporting_source'],
        'generated_question_1': row['generated_question_1'],
        'summary_1': row['summary_1'],
        'label_1': row['label_1'],
        'document_url_question_1': row['document_url_question_1'],
        'generated_question_2': row['generated_question_2'],
        'summary_2': row['summary_2'],
        'label_2': row['label_2'],
        'document_url_question_2': row['document_url_question_2'],
        'generated_question_3': row['generated_question_3'],
        'summary_3': row['summary_3'],
        'label_3': row['label_3'],
        'document_url_question_3': row['document_url_question_3'],
        'judgement': row['judgement'].replace("\n", " ").strip(),
    }
        
    remaining_rows.append(remaining_row)
remaining_df = pd.DataFrame(remaining_rows)

remaining_df.to_csv('../final/out_csvs/q3_enough_generated.csv', index  = False)