# https://www.youtube.com/watch?v=trpVNau7iKY
# https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/qdrant.html
#   reuse qdrant indexes instead of re-indexing everytime
# yet to try
#   - run in offline mode
#   - check "Offline mode" in https://huggingface.co/docs/transformers/main/installation
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader, TextLoader
from langchain.vectorstores import Qdrant, DeepLake
from langchain.indexes import VectorstoreIndexCreator

from langchain.vectorstores import Chroma
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.prompts import PromptTemplate


loader = UnstructuredFileLoader("data/apt_info.txt")
docs = loader.load()
print(docs[0].page_content)

llm = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temperature": 0.1,"max_new_tokens":256})
#llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0.1,"max_new_tokens":256})

text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
texts = text_splitter.split_text(docs[0].page_content)  ### <---- pay attention here


# we are passing metadata also. It can be any metadata .
# This metadata in our case is telling which source/doc the results come from
embeddings = HuggingFaceEmbeddings()

################################################################################################
# doc store
#1. pip install qdrant-client
docsearch = Qdrant.from_texts(texts, embeddings, path="/tmp/local_qdrant",
    collection_name="apt_documents")

#2. Chroma
#docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])

################################################################################################

question = "when was Gopalan Habitat Splendour constructed?"

################################################################################################
# chain vs qna
# 1. chain - below works
#chain = load_qa_with_sources_chain(llm, chain_type="refine") # refine does not work
#docs = docsearch.similarity_search(question)  # <- works
 #chain({"input_documents" : docs, "question" : question}, return_only_outputs=True)  #<- gives error, to check
#print(docs[0].page_content)

# 2. qna - hangs, does not work

#qna = RetrievalQA.from_chain_type(llm= llm, chain_type="stuff", retriever=docsearch.as_retriever())
#question = "when was Gopalan Habitat Splendour constructed?"
#print(qna.run(question))

# 3. https://python.langchain.com/en/latest/use_cases/question_answering.html
# or https://python.langchain.com/en/latest/use_cases/question_answering/semantic-search-over-chat.html
# pip install tiktoken
# this too hangs - Model google/flan-t5-xl time out, google/flan-ul2
db = DeepLake.from_texts(texts, embeddings, dataset_path="/tmp/deeplake_path", overwrite=True)
db = DeepLake(dataset_path="/tmp/deeplake_path", read_only=True, embedding_function=embeddings)
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)
ans = qa({"query": question})
print("ANSWER IS " + ans)



################################################################################################



'''
template = """Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
{summaries} are unordered and come from OCR

Respond in English and Order and fix typos

QUESTION : {question}
========
{summaries}
=========
FINAL ANSWER IN English:"""

#### 3. using template
'''

