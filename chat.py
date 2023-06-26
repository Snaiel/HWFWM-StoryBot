
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import UnstructuredEPubLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI

from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
import os

from pprint import pprint

import_data = False

if import_data:
    loaders = []

    for file in os.listdir("data"):
        loaders.append(UnstructuredEPubLoader(f"data/{file}"))

    def put_metadata_into_sub_dict(docs: list[Document]) -> list[Document]:
        for doc in docs:
            doc.metadata = {
                'metadata': doc.metadata
            }
        return docs

    docs = []
    for loader in loaders:
        new_docs = loader.load()
        new_docs = put_metadata_into_sub_dict(new_docs)
        docs.extend(new_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

collection_name = "HWFWM_StoryBot"

if import_data:
    vectorstore = Milvus.from_documents(documents, embeddings, collection_name=collection_name, drop_old=True)
else:
    vectorstore = Milvus(embeddings, collection_name)


question = "How does Shako kill Jason"

# giving a number higher than the amount of results gives an error
context = vectorstore.similarity_search(question)

context = " ".join([i.page_content for i in context])

chat = ChatOpenAI()

prompt=PromptTemplate(
    template="You are a helpful assistant that answers question based on the given context. \n\n\n CONTEXT: \n {context}",
    input_variables=["context"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

human_template="{question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
output = chat(chat_prompt.format_prompt(context=context, question=question).to_messages())
pprint(output.content)