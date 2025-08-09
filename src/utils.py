from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def pdf_to_vector_store(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embedding_model
):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)

    return vectorstore


def assign_score_with_llm_as_a_judge(
        query,
        result,
        llm
):

    prompt = PromptTemplate.from_template(
        """
        You are a judge who evaluates the relevance of a document to a query.

        Query: {query}
        Document: {result}

        Provide a score from 1 to 5, where 1 means "not relevant at all" and 5 means "highly relevant".
        Provide your score as a single word: "1", "2", "3", "4", or "5".
        Output only the score without any additional text.

        Provide output in json format with the following keys:
        - "score": the score you give to the document based on its relevance to the query
        """
    )

    chain = (
        prompt |
        llm |
        JsonOutputParser()
    )

    response = chain.invoke({
        "query": query,
        "result": result
    })
    return response

def filter_retrieved_results_by_score(results, query, llm, threshold):
    scored = [
        (result, assign_score_with_llm_as_a_judge(query, result.page_content, llm))
        for result in results
    ]

    filtered = [
        result.page_content for result, score in scored if int(score['score']) >= threshold
    ]

    return filtered