import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


def get_conversational_chain(question, context):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context". Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)

    response = chain.run({"context": context, "question": question})

    return response