# https://www.youtube.com/watch?v=qgaM0weJHpA
# https://www.youtube.com/watch?v=l8ZYCvgGu0o
# https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset


# notes
#   "google/flan-t5-xl takes time to respond
#   "google/flan-t5-small responds immediately
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl") 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

######## expmnt1
ipfile = open("data/hs_info.txt", "r")
#encode_data =
########

question = "what are vegetables rich in protien"
inputs = tokenizer(question, return_tensors="pt")

###

outputs = model.generate(**inputs, max_new_tokens=1600)  # max_length = decide o/p length
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))