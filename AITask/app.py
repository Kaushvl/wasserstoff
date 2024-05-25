import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from ollama import chat
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import requests

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def GetWebText(strUrl: str):
    objLoader = RecursiveUrlLoader(
        url=strUrl, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    objData = objLoader.load()
    strWebcontent = objData[0].page_content

    with open('strData.txt', 'w') as file:
        file.write(strWebcontent)

    return strWebcontent

def GetTextChunks(strText: str):
    objTextSplitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    lsChunks = objTextSplitter.split_text(strText)
    return lsChunks

def GetVectorStore(lsTextChunks: list):
    objEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(lsTextChunks, embedding=objEmbeddings)
    vector_store.save_local("faiss_index")

def UserInput(strUserquestion: str):
    objEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", objEmbeddings, allow_dangerous_deserialization=True)
    strContext = new_db.similarity_search(strUserquestion)
    context = strContext
    question = strUserquestion

    prompt_template = f"""
    Answer the question only from the provided context in simple words, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question:\n {question}
    """
    objCompletion = chat(model='llama3:8b-instruct-q8_0', messages=[{"role":"assistant","content":prompt_template}])
    st.write("Reply: ", objCompletion['message']['content'])

def fetch_wordpress_content(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch content from WordPress site.")
        return None

def generate_embeddings(text):
    objEmbeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return objEmbeddings.embed(text)

def update_vector_database(post_id, embeddings):
    vector_store = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    vector_store.add_texts([embeddings])
    vector_store.save_local("faiss_index")

def main():
    st.set_page_config("Chat Website")
    st.header("Chat With URL using Local LLAMA3")

    strUserquestion = st.text_input("Ask a Question from the PDF Files")

    if strUserquestion:
        UserInput(strUserquestion)

    with st.sidebar:
        st.title("Menu:")
        strWebUrl = st.text_input("Enter the website URL")
        wp_api_url = st.text_input("Enter the WordPress API URL")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = GetWebText(strWebUrl)
                lsTextChunks = GetTextChunks(raw_text)
                GetVectorStore(lsTextChunks)
                st.success("Done")
        
        if st.button("Fetch WordPress Content"):
            with st.spinner("Fetching content..."):
                content = fetch_wordpress_content(wp_api_url)
                if content:
                    text = extract_text_from_wp_content(content)
                    embeddings = generate_embeddings(text)
                    update_vector_database(content['id'], embeddings)
                    st.success("Content processed and embeddings updated.")

def extract_text_from_wp_content(content):
    return " ".join([item['content'] for item in content['posts']])

if __name__ == "__main__":
    main()
