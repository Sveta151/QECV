# QueenVer
AVeriTeC shared task pipeline 

# Pipeline

## Install Dependencies
`pip install -r requirements.txt`

In addition to this, you also need to install OpenAI, sentence-transformers and FAISS.

## Run pipeline
Notes: This is dummy pipeline. It requires files under `data/<claim_id>.json` to obtain relevant documents for `<claim_id>`

Notes: Provide openai `API_KEY` under `run.py`. It will be moved to using env later.

Run: `./script.py`
