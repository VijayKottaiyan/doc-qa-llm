from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA

directory = "/Users/vijayk/Programming/AI/RAG/Document_QA/data"  # keep multiple files (.txt, .pdf) in data folder.


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


documents = load_docs(directory)
print(len(documents))

docs = split_docs(documents)
print(len(docs))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

collection_name = "langchain-demo"

db = Chroma.from_documents(docs, embeddings, collection_name=collection_name)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(
    model="phi3:mini",
    temperature=0.2,
    num_ctx=4096,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
)

query = "What is the tentative date of handing over the building"

answer = qa_chain.invoke(query)

print(answer["result"])
