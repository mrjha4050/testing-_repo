from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()

llm= GooglePalm(google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.6)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path= "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='new_testing_db.csv', source_column="prompt", encoding='iso-8859-1')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path , instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """
        Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context and make changes acccording to user requirement.
        If the answer is not found in the context, search it and make it refine and present the answer , and try to make every answer more refined and correct .

        CONTEXT: {context}

        QUESTION: {question}
        """
    PROMPT=PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT})
    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("what is investment banking?"))
