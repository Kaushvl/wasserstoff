from flask import Flask, request, jsonify, render_template
import sys
from RAG.DataRetrieval import fetchUrlData
from RAG.VDatabase import answer_query
from RAG.LLMProcessing import get_conversational_chain
import time

app = Flask(__name__,template_folder="./templates")

@app.route('/')
def home():
    return render_template(r"index.html")

@app.route('/process_url', methods=['POST'])
def process_url():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        fetchUrlData(url)
        return jsonify({'message': 'URL processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    try:
        # Get the context from the vector database
        context_results = answer_query(user_query)
        if not context_results:
            return jsonify({'answer': 'No relevant information found'}), 200
        
        # Extract the relevant context
        context = context_results[0]['content']
        print(context)
        
        # Get the conversational chain
        answer = get_conversational_chain(question=user_query,context= context)

        print("44 dfd Done")
        
        # Run the chain to get the answer
        # answer = chain.invoke({"question":user_query,"input_documents":context_results})

        print(answer)
        
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
