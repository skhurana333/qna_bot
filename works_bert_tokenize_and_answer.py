# https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt
# https://discuss.huggingface.co/t/how-can-i-put-multiple-questions-in-the-same-context-at-once-using-question-answering-technique-im-using-bert/10416
# https://towardsdatascience.com/3-types-of-contextualized-word-embeddings-from-bert-using-transfer-learning-81fcefe3fe6d


from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from datasets import load_dataset

from transformers import BertTokenizer, BertForQuestionAnswering, AutoTokenizer, pipeline

import torch

from transformers import pipeline

model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
# Gopalan Habitat Splendour: #
Gopalan Habitat Splendour is a residential complex in Brookefields,Bangalore. It was constructed on 2006. 
It has around 500 apartments. It has 2 bhk, 3bhk , 4bhk and penthouse apartments.  Postal zip code is 560037. 
It is in east part of Bangalore. 
The areas nearby Gopalan Habitat Splendour are Whitefied, Marathahalli, Varthur"

# Sobha Dream Acres: #
Sobha Dream Acres is a residential complex in Panathur, outer ring road, Bangalore. It was constructed on 2020. It has around 5100 apartments. 
It has 2 bhk and studio apartments.  
Postal zip code is 560035. The areas nearby Sobha Dream Acres are  Marathahalli, TheLake.

# Prestige City: #
Prestige city is a residential complex in Bangalore. It is still under construction. 
It is supposed to be completed in 2025. has around 15000 apartments. It has 2 bhks, villas and 3 bhks. 
It is located on Sarjapur road. Postal zip code is 560040. 
The areas nearby Prestige City  are  Varthur, Sarajapur Village. Good things about this apartment are 
a) good location b) near greenery c) lots of good people around 4) away from pollution 
"""


question = "What are good things about Prestige city, tell me all?"
qar = question_answerer(question=question, context=context)
print("Prestige city nearby areas -> " + qar['answer'] + ":::score-> " + str(qar['score']))


# construction year
question = "When was Gopalan Habitat Splendour constructed?"
qar = question_answerer(question=question, context=context)
print("Gopalan Habitat Splendour construction year -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "When was Sobha Dream Acres constructed?"
qar = question_answerer(question=question, context=context)
print("Sobha Dream Acres construction year -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "When was Prestige city constructed?"
qar = question_answerer(question=question, context=context)
print("Prestige city construction year -> " + qar['answer'] + ":::score-> " + str(qar['score']))


# nearby
question = "What are areas nearby Gopalan Habitat Splendour?" # getting wrong answer with low score
qar = question_answerer(question=question, context=context)
print("Gopalan Habitat Splendour  nearby areas-> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "What are areas nearby Sobha Dream Acres?"
qar = question_answerer(question=question, context=context)
print("Sobha Dream Acres nearby areas -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "What are areas nearby Prestige city?"
qar = question_answerer(question=question, context=context)
print("Prestige city nearby areas -> " + qar['answer'] + ":::score-> " + str(qar['score']))


# location
question = "Where is Gopalan Habitat Splendour located?"
qar = question_answerer(question=question, context=context)
print("Gopalan Habitat Splendour location -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "Where is Sobha Dream Acres located?"
qar = question_answerer(question=question, context=context)
print("Sobha Dream Acres location -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "Where is Prestige City located?"
qar = question_answerer(question=question, context=context)
print("Prestige City location -> " + qar['answer'] + ":::score-> " + str(qar['score']))

# postal zip codes
question = "What is postal zip code of Gopalan Habitat Splendour?"
qar = question_answerer(question=question, context=context)
print("Gopalan Habitat Splendour location -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "what is postal zip code of Sobha Dream Acres?"
qar = question_answerer(question=question, context=context)
print("Sobha Dream Acres location -> " + qar['answer'] + ":::score-> " + str(qar['score']))

question = "what is postal zip code of Prestige City?"
qar = question_answerer(question=question, context=context)
print("Prestige City location -> " + qar['answer'] + ":::score-> " + str(qar['score']))



''' wip below
yet to make below work
################################## custom fine tune ##################################
# from -  https://huggingface.co/learn/nlp-course/chapter7/7?fw=pt
raw_datasets = load_dataset("json",
                            data_files={"train": ["data/apt_info_for_bert1.json",
                                                  "data/apt_info_for_bert2.json"],
                                        "test": "data/apt_info_for_bert3.json"},
                            )
print(raw_datasets)
#raw_datasets = load_dataset("squad")

print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# We can pass to our tokenizer the question and the context together,
# and it will properly insert the special tokens to form a sentence like this
# [CLS] question [SEP] context [SEP]

context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

#  we will deal with long contexts by creating several training features from one sample of our dataset,
#  with a sliding window between them.
#
# max_length to set the maximum length (here 100)
# truncation="only_second" to truncate the context (which is in the second position) when the question with its context is too long
# stride to set the number of overlapping tokens between two successive chunks (here 50)
# return_overflowing_tokens=True to let the tokenizer know we want the overflowing tokens
#



inputs = tokenizer(question,
                   context,
                   max_length=100,
                   truncation="only_second",
                   stride=50,
                   return_overflowing_tokens=True,
                   return_offsets_mapping=True,
                   )

print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")
'''


''' yet to make below work
################################## bert ##################################
tokenizer = Tokenizer.from_file("data/apt_info_for_bert1.json")
normalizer = normalizers.Sequence([NFD(), StripAccents()])
tokenizer.normalizer = normalizer

## BERT preprocesses texts by removing accents and lowercasing. We also use a unicode normalizer:
bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# The pre-tokenizer is just splitting on whitespace and punctuation
bert_tokenizer.pre_tokenizer = Whitespace()

# And the post-processing uses the template we saw in the previous section:
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

# We can use this tokenizer and train on it on wikitext like in the quicktour
trainer = WordPieceTrainer(vocab_size=30522, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
bert_tokenizer.train(files, trainer)
bert_tokenizer.save("data/bert-wiki.json")


################################## bert ends ##################################
'''




