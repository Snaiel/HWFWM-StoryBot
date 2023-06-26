
from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import UnstructuredEPubLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI

from pprint import pprint

import_data = False

if import_data:
    loaders = [
        UnstructuredEPubLoader("data/He Who Fights with Monsters (Book 01) (Shirtaloon).epub")
    ]

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    documents = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

collection_name = "HWFWM_StoryBot"

if import_data:
    vectorstore = Milvus.from_documents(documents, embeddings, collection_name=collection_name, drop_old=True)
else:
    vectorstore = Milvus(embeddings, collection_name)

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True)

question = "How does Shako kill Jason"

output = qa({"query": question})

# pprint(output)
print(output['result'].strip())

