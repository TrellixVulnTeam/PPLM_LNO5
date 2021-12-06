import pandas as pd
import numpy as np

from transformers import GPT2LMHeadModel
# This downloads GPT-2 Medium, it takes a little while
_ = GPT2LMHeadModel.from_pretrained("gpt2-medium")

test = pd.read_csv('test_final_v3.csv')

from run_pplm import run_pplm_example

topics = ['world', 'sports', 'business', 'sci_tech']

for index, row in test.iterrows():
    print(index)
    for topic in topics:
        output_text = \
          run_pplm_example(
              cond_text=row['Phrase'],
              num_samples=1,
              bag_of_words=topic,
              length=100,
              stepsize=0.03,
              sample=True,
              num_iterations=3,
              window_length=5,
              gamma=1.5,
              gm_scale=0.95,
              kl_scale=0.01,
              verbosity='quiet'
          )
        output_text = output_text.replace('<|endoftext|>', '')
        test.loc[index, topic + '_pplm'] = output_text
    test.to_csv('pplm_out.csv', index=False)