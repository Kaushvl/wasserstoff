from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

movies = []

def PreProcessData(lsData):
    docs=[]
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
    embeddings_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    faiss_index_path = r"AITask\faiss_index"
    # Load the existing FAISS index if it exists
    if os.path.exists(faiss_index_path):
        faiss_index = FAISS.load(faiss_index_path)
    else:
        faiss_index = FAISS(embeddings_model)
    
    # Embed the new documents
    new_embeddings = embeddings_model.embed_documents([doc.page_content for doc in new_docs])
    
    # Add the new embeddings to the FAISS index
    faiss_index.add_embeddings(new_embeddings, new_docs)
    
    # Save the updated FAISS index
    faiss_index.save(faiss_index_path)