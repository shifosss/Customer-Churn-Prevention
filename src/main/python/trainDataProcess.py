import pandas as pd

train_df = pd.read_csv('data/sms_spam_TRAIN_ORIGIN.csv')
test_df = pd.read_csv('data/sms_spam_TEST_ORIGIN.csv')

for i in range(len(train_df['prompt'])):
    input_ = {'role': 'user', 'content': train_df['prompt'][i]}
    train_df['prompt'][i] = input_

for i in range(len(test_df['prompt'])):
    input_ = {'role': 'user', 'content': test_df['prompt'][i]}
    test_df['prompt'][i] = input_

test_df.to_csv('data/sms_spam_TEST_REFINED.csv', index=False)