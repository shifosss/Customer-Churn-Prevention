from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    max_length=200,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    device=0 if torch.cuda.is_available() else -1,
    use_auth_token='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'
)

# Load the dataset
df = pd.read_csv('data/sms_spam_TEST_REFINED.csv')

total_correct = 0

for i in range(len(df['prompt'])):
    input_ = {'role': 'user', 'content': df['prompt'][i]}
    response = pipe(input_)
    result = response[0]['generated_text']
    if 'spam' in result.lower() and df['label'][i] == 1 or 'ham' in result.lower() and df['label'][i] == 0:
        total_correct += 1

print(total_correct / len(df['prompt']))