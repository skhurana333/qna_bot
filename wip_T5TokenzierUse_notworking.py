# https://www.youtube.com/watch?v=6W-HMbgRJRM <- does only question over existing wiki data and summarization task
# https://www.youtube.com/watch?v=wZBPJpDdiEM <-  commercial basten , needs basten login

from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

############################################################################################################
# distilbert model 
''' 
import pandas as pd
from transformers import DistilBertTokenizerFast

df = apts = pd.read_csv("/Users/skhurana/prjs/qna_bot/data/apt_info.csv", sep=',', names=["apt_name", "apt_info"])

x = list(df['apt_name'])
y = list(df['apt_info'])

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(x, truncation=True, padding=True) # has input ids etc
print(train_encodings)

import tensorflow as tf
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y))
print(train_dataset)
'''

############################################################################################################
# t5 model
# pip install accelerate
# notes
#   "google/flan-t5-xl takes time to respond
#   "google/flan-t5-small responds immediately

''' to see how to make below work  '''
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl",padding=True, truncation=True)
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

mydata = open("/Users/skhurana/prjs/qna_bot/data/apt_info.txt", "r")
text_data = mydata.readlines()
#print(text_data)

input_ids = tokenizer(text_data, return_tensors="pt")
#.inputs_ids.to("cuda")

outputs = model.generate(input_ids, min_length=20, max_new_tokens=600,padding=True, truncation=True)


print(tokenizer.decode(outputs[0], skip_special_tokens=True))


