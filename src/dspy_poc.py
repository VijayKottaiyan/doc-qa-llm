# dsp_rag.py

import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb


def load_and_split_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"No. of documents loaded - {len(documents)}")
    print(f"No. of Chunks created - {len(docs)}")

    return docs


class GenerateAnswer(dspy.Signature):
    """
    This RAG Question and Answer Tool, Which utilizes custom documents as its knowledge base.
    Model takes the QUESTION along with the CONTEXT as the input.
    Produces a Concise answer which is factual and based on the given context.
    """

    context = dspy.InputField(
        desc="Contains relevant information's and facts regarding QUESTION"
    )
    question = dspy.InputField(
        desc="Question provided by the user regarding their document"
    )
    answer = dspy.OutputField(
        desc="A concise answer to the QUESTION based on the CONTEXT provided"
    )


class RAG(dspy.Module):
    def __init__(self, k):
        super().__init__()
        self.retriever = dspy.Retrieve(k=k)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retriever(question).passage

        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


if __name__ == "__main__":

    client = dspy.LM(
        model="phi3:mini", model_type="chat", max_tokens=4096, temperature=0.3
    )
    dspy.configure(lm=client)

    collection_name = "dspy-rag"

    chroma_client = chromadb.Client()

    retriever_model = ChromadbRM(
        collection_name=collection_name,
        embedding_function=dspy.LM(model="nomic-embed-text"),
        persist_directory=":memory:",
        chroma_client=chroma_client,
    )

    chunks = load_and_split_docs(
        directory="/Users/vijayk/Programming/AI/RAG/Document_QA/data"
    )

    dspy_docs = [
        dspy.Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in chunks
    ]

    retriever_model.add(dspy_docs)

    rag_program = RAG()

    query = "What is the tentative date of handing over the building?"

    prediction = rag_program(question=query)

    print(prediction.answer)
