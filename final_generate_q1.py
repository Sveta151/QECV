from huggingface_hub import login
huggingface_token = 'huggingface_token'
login(token=huggingface_token)
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
import sys

# Load the train and dev datasets using pandas
# train_df = pd.read_json('https://huggingface.co/chenxwh/AVeriTeC/raw/main/data/train.json')
dev_df = pd.read_json('../data_dev.json')

from openai import OpenAI
import pandas as pd
## Set the API key
client = OpenAI(api_key="API_KEY")

MODEL="gpt-4o"

CODE_DEMO_FIRST_NEW = '''Task: to verify a claim, we need to ask a series of simple questions. Here the task is given a claim, generate the first question to ask. 
This question should be:

- Simple with a single subject-verb-object structure.
- Specific and directly related to the key aspect of the claim that needs validation.


Claim = Superdrag and Collective Soul are both rock bands.
To validate the above claim, the first simple question we need to ask is:
Question = Is Superdrag a rock band?

Claim = Jimmy Garcia lost by unanimous decision to a professional boxer that challenged for the WBO lightweight title in 1995. 
To validate the above claim, the first simple question we need to ask is: 
Question = Who is the professional boxer that challenged for the WBO lightweight title in 1995? 

Claim = The Swan of Catania was taught by the Italian composer Giovanni Furno.
To validate the above claim, the first simple question we need to ask is:
Question = What is the nationality of Giovanni Furno?

Claim = Smith worked on the series The Handmaid's Tale that is based on a novel by Margaret Atwood.
To validate the above claim, the first simple question we need to ask is:
Question = Which novel The Handmaid's Tale is based on?

Claim = The Potomac River runs along the neighborhood where Ashley Estates Kavanaugh's wedding was held.
To validate the above claim, the first simple question we need to ask is:
Question = Where was Ashley Estates Kavanaugh's wedding held?

Claim = Ulrich Walter's employer is headquartered in Cologne.
To validate the above claim, the first simple question we need to ask is:
Question = Who is Ulrich Walter's employer?

Claim = Lars Onsager won the Nobel prize when he was 30 years old.
To validate the above claim, the first simple question we need to ask is:
Question = When Lars Onsager won the Nobel prize?

Claim = [[CLAIM]]
To validate the above claim, the first simple question we need to ask is:
Question = '''

def generate_question_new(claim):
    QG_template_start = CODE_DEMO_FIRST_NEW
    input_claim = QG_template_start.replace('[[CLAIM]]', claim.strip())
    completion = client.chat.completions.create(
    model = MODEL,
    messages=[
    {"role": "system", "content": "You are a helpful assistant tasked with generating specific and simple questions to verify provided claims. Each question should have a single subject-verb-object structure and directly address the key aspect of the claim that needs validation."},
    {"role": "user", "content":input_claim}
  ]
    )
    output_q = completion.choices[0].message.content
    return output_q
    
dev_df['claim_id'] = dev_df.index

N_generate_new = []

if sys.argv[1] == "all":
    N = len(dev_df)
else:
    N = int(sys.argv[1])
for i in range(N):
    claim = dev_df['claim'][i]
    first_question = generate_question_new(claim)
    N_generate_new.append({
        'claim_id' : dev_df['claim_id'][i],
        'claim' : claim,
        'claim_date' : dev_df['claim_date'][i],
        'speaker' : dev_df['speaker'][i],
        'reporting_source' : dev_df['reporting_source'][i],
        'generated_question_1': first_question
    })

questions_dev_first_full_df = pd.DataFrame(N_generate_new)

questions_dev_first_full_df.to_csv('../final/q1_' + sys.argv[1] +'_samples_dev.csv', index = False)