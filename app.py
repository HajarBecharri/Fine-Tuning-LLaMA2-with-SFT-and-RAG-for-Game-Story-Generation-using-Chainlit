from operator import itemgetter
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from chainlit.types import ThreadDict
import chainlit as cl
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

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
    
if not all_docs:
    raise ValueError("No documents loaded. Please check the folder path and PDF files.")

for doc in all_docs:
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc)}")
    if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
        raise AttributeError("Document instance must have 'page_content' and 'metadata' attributes")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(all_docs)
    # Verify documents is a list of Document instances
if not documents:
    raise ValueError("No split documents created.")
for doc in documents:
    if not isinstance(doc, Document):
        raise TypeError(f"Expected Document instance, got {type(doc)}")
    # create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#     # load it into Chroma
db = Chroma.from_documents(documents, embedding_function)
# notre llm
llm = Ollama(model="llama3")



custom_prompt_template = """You are a helpful, respectful and honest assistant.
 Always answer as helpfully as possible, while being safe. 
   Your answers should not include any harmful, unethical,
     racist, sexist, toxic, dangerous, or illegal content. 
     Please ensure that your responses are socially unbiased and positive in nature .
     using only the context provided.
If a question does not make any sense, or is not factually coherent,
 explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 
 all your response should be in French

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'] )
    return prompt



@cl.password_auth_callback
def auth():
    return cl.User(identifier="Gamer")


@cl.on_chat_start
async def on_chat_start():
    elements = [
        cl.Image(name="image1", display="inline", path="falcon.jpeg")
    ]
    await cl.Message(content="Bonjour, je suis Falcon. Je peux vous aider à générer des histoires basées sur vos descriptions. Veuillez fournir une description pour commencer.", elements=elements).send()
    
    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    prompt = set_custom_prompt()

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    cl.user_session.set("chain", chain)




@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain") 
    #call backs happens asynchronously/parallel 
    cb = cl.AsyncLangchainCallbackHandler()
    
    # call the chain with user's message content
    res = await chain.ainvoke(message.content, callbacks=[cb])
    print(res)
    answer = res["answer"]

    await cl.Message(content=answer).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Restaurer la mémoire de la conversation
    memory = ConversationBufferMemory(return_messages=True , memory_key="chat_history")
    root_messages = [m for m in thread["steps"] if m["parentId"] is None]
    for message in root_messages:
        if message["type"] == "user_message":
            # Récupérer la question à partir du message restauré
            question = message["output"]
            memory.chat_memory.add_user_message(question)
        else:
            memory.chat_memory.add_ai_message(message["output"])

    # Récupérer ou créer la chaîne en utilisant la mémoire restaurée
    chain = cl.user_session.get("chain")
    if chain is None:
        print("chain est null")
        # Créer une nouvelle chaîne si elle n'existe pas (utile pour les tests)
        prompt = set_custom_prompt()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        cl.user_session.set("chain", chain)
    else :
        print("la chain est non null")





    

