from flask import Flask, render_template, request, jsonify
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_community.document_loaders import TextLoader

# Load environment variables
load_dotenv()
api_key = "You Mistral Ai API goes here"
client = Mistral(api_key=api_key)

# Load data
loader = TextLoader(r"youfile_name.txt", encoding="utf-8")
docs = loader.load()
text = docs[0].page_content

# Chunk text data
chunk_size = 6500
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get text embedding
def get_text_embedding(input_text):
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    return embeddings_batch_response.data[0].embedding

# Add a delay between API calls to avoid rate limiting
delay_seconds = 2
text_embeddings = []
for chunk in chunks:
    embedding = get_text_embedding(chunk)
    text_embeddings.append(embedding)
    time.sleep(delay_seconds)

# Convert embeddings to a NumPy array and index with Faiss
text_embeddings = np.array(text_embeddings)
d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

# Function to generate a response with Mistral
def run_mistral(prompt, model="open-mistral-nemo"):
    messages = [{"role": "user", "content": prompt}]
    time.sleep(delay_seconds)
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content

# Flask setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    question = request.args.get('msg')
    
    # Get embedding for the question
    question_embedding = np.array([get_text_embedding(question)])
    
    # Find the closest matching chunk
    D, I = index.search(question_embedding, k=2)
    retrieved_chunk = [chunks[i] for i in I[0]]

    # Construct the prompt for Mistral
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, always give answer in bangla.
    Query: {question}
    Answer:
    """

    # Get response from Mistral
    answer = run_mistral(prompt)
    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True)
