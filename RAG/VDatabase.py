import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def PreProcessData(lsData):
    docs = []
    for siteData in lsData:
        text = ""
        text += "PageId: " + str(siteData['id']) + "\n"
        text += "Title: " + siteData['title'] + "\n"
        text += f"Content : {siteData['content']}\n"

        metadata = dict(
            source=siteData['id'],
            title=siteData['title']
        )
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)
    updateFaissIndex(docs)
    return True

def updateFaissIndex(new_docs):
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    faiss_index_path = r"AITask/faiss_index"

    # Load the existing FAISS index if it exists
    if os.path.exists(faiss_index_path):
        faiss_index = FAISS.load_local(faiss_index_path, embeddings_model,allow_dangerous_deserialization=True)
        faiss_index = faiss_index.from_documents(new_docs,embedding=embeddings_model)
    else:
        # Determine the dimension of the embeddings
        sample_embedding = embeddings_model.embed_documents(["sample text"])[0]

        faiss_index = FAISS.from_documents(new_docs,embedding=embeddings_model)

    # Save the updated FAISS index
    faiss_index.save_local(faiss_index_path)

def load_faiss_index():
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    faiss_index_path = r"AITask/faiss_index"
    
    if os.path.exists(faiss_index_path):
        faiss_index = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
        return faiss_index
    else:
        raise ValueError("FAISS index not found. Please ensure the index is created and saved at the specified path.")


def answer_query(query):
    faiss_index = load_faiss_index()
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Embed the query
    query_embedding = embeddings_model.embed_documents([query])[0]
    
    # Search the FAISS index
    search_results = faiss_index.similarity_search_by_vector(query_embedding, k=1)  # Get top 5 similar results
    
    # Retrieve the corresponding documents
    # Retrieve the corresponding documents
    documents = []
    for result in search_results:
        doc = result.metadata
        # Retrieve full content from metadata
        documents.append({
            'source': doc['source'],
            'title': doc['title'],
            'content': result.page_content
        })
    return documents

# Example usage
if __name__ == "__main__":
    # Load the FAISS index and answer a query

    
    query = "Who is donald trump"
    results = answer_query(query)
    print("Query Results:", results)
