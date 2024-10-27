import pandas as pd

train_df = pd.read_csv('')

for i in range(len(train_df['prompt'])):
    input_ = {'role': 'user', 'content': train_df['prompt'][i]}
    train_df['prompt'][i] = input_

print(train_df)