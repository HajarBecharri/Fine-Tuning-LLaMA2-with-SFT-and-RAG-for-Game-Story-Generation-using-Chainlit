#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
from langchain_community.document_loaders import PyPDFLoader

# Path to the folder containing the PDF files
folder_path = 'Documents'

# List to hold all documents
all_docs = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        # Create a PyPDFLoader instance for each PDF file
        loader = PyPDFLoader(os.path.join(folder_path, filename))
        # Load the documents from the PDF file
        docs = loader.load()
        # Append the loaded documents to the all_docs list
        all_docs.extend(docs)


# In[22]:


print(all_docs[:3])


# In[23]:


from langchain.schema import Document
# Verify all_docs is not empty and contains Document instances
if not all_docs:
    raise ValueError("No documents loaded. Please check the folder path and PDF files.")

for doc in all_docs:
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc)}")
    if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
        raise AttributeError("Document instance must have 'page_content' and 'metadata' attributes")


# In[24]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(all_docs)
documents[:3]


# In[25]:


# Verify documents is a list of Document instances
if not documents:
    raise ValueError("No split documents created.")
for doc in documents:
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc)}")


# ## Vector Embedding And Vector Store
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(documents, embedding_function)


# In[28]:


# query it
query = "Qu'est ce que le jeux  Super Mario Bros "
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)


# In[29]:


from langchain_community.llms import Ollama
# ## Load Ollama LAMA2 LLM model
llm=Ollama(model="llama3")
llm


# In[30]:


## Design ChatPrompt Template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
You are a bot that answers questions about game storys in french, using only the context provided.
If you don't know the answer, simply state that you don't know.
<context>
{context}
</context>
Question: {input}""")


# In[31]:


## Chain Introduction
## Create Stuff Docment Chain

from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain=create_stuff_documents_chain(llm,prompt)


# In[32]:


retriever = db.as_retriever(language="fr")
retriever


# In[33]:


from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)


# In[35]:


response=retrieval_chain.invoke({"input":"donner moi une histoire pour crrer un jeux  "})


# In[36]:


print(response['answer'])

