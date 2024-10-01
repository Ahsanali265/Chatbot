from flask import Flask, request, jsonify, send_from_directory
import faiss
import openai
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


openai.api_key = "Your Api Key"

retrieval_model = SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder="model")
df = pd.read_csv("updated_faq.csv", encoding='latin-1')

embeddings = retrieval_model.encode(df['Question'].tolist())
spc = df['Answer'].tolist()
embeddings = np.array(embeddings).astype('float32')
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) 
faiss_index.add(embeddings)

def search(query_vector, faiss_index, spc, k=1):
    try:
        if not isinstance(faiss_index, faiss.Index):
            raise ValueError("Index is not a faiss.Index object.")
        
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vector)
        similarity, indices = faiss_index.search(query_vector, k)
        
        results = []
        for j, i in enumerate(indices[0]):
            score = round(similarity[0][j], 2)
            if score < 0.75:
                results.append((None, score))  
            else:
                results.append((spc[i], score)) 
        
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return [("Sorry, I couldn't retrieve relevant information at this time.", "")]

def generate_response_with_openai(messages):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."

def get_response(query):
    query_vector = retrieval_model.encode([query])
    search_results = search(query_vector, faiss_index, spc)
    
    top_result = search_results[0][0] if search_results and search_results[0][0] else "I couldn't find specific information about your query. If you have any further questions, please let me know."
    context = top_result if top_result != "I couldn't find specific information about your query. If you have any further questions, please let me know." else "I couldn't find specific information about your query. If you have any further questions, please let me know."

    prompt = f"""
    You are a helpful assistant. Here are some instructions to handle user queries:

        1.  If the user query is a common greeting or farewell or express gratitude, respond appropriately. For example:
           - "Hi there! How can I assist you today?"
           - "Goodbye! Feel free to reach out if you have any more questions."
           -  "You're welcome! Let me know if there's anything else I can help with."

        2.  If the query is incomplete or contains common typos,try to correct or understand the type, try to interpret the intended question it can be basically anything ,example of Patterns include:
           - Variations like "sps", "what, is sps", "what sps", "wht sps", and "waht sps".

        3.  If the query matches any standard context in csv file, provide the relevant answer from the provided or pre defined data. For example, if the user asks about "SPS", provide a detailed answer about the company and extract that information from the file. Use the following context to provide the answer:
           - Context: {context}

        4.  If the query does not match any known patterns or standard queries in the database, respond with a message indicating that specific information could not be found and encourage the user to provide more details or ask another question. For example: "I couldn't find specific information about your query. If you have any further questions, please let me know."

        5.  If the user input query or asked about something, fetch that from the file and make slight changes or modifications but keep the main idea and context intact.
        
        6. Just give the answer to the user's query and dont tell what you done in the background.
        
        7. If the user asks for help or guidance, respond with a friendly message offering assistance and asking how you can help.
                
        8. Keep responses concise and ensure smooth, helpful interactions,and guide the user to provide more context or ask specific questions if their queries are unclear.
        
        9. If the user expresses that they have more questions or queries, respond with an encouraging message to continue assisting. For example: "Sure! I'm here to help. What else would you like to know?"
        
        10. If the user indicates that their query was a joke or they do not have a serious question, respond politely and redirect to ask if they need further assistance or information. For example: "Got it! If you have any other questions or need information about SPS, feel free to ask."
        
        11. If the user says something casual or similar informal greetings, gratitude,slangs respond with a friendly acknowledgment and offer assistance.
        
        12. If the user uses inappropriate language or offensive comments, respond with: "Please avoid using inappropriate language. I'm here to help with information about SPS services. Let me know if you have any relevant questions."
        
        13. If the user asks for personal information or sensitive data, respond with: "I'm here to provide information about SPS services and products. Please avoid sharing personal information. Let me know if you have any questions about our services."
        
        14. For Handling General conversations like "ok" ,"hm" etc,Ensure responses are friendly, helpful, and guide the user to provide more context if needed.
        
        15. If the user asks for a joke or fun fact, respond with a light-hearted joke or interesting fact about SPS or the services provided. For example: "Why did the computer go to the doctor? It had a virus! 😄"
        
        16. If the user says "nothing" or "no", respond with: "Understood. If you have any questions or need assistance in the future, feel free to ask."
        
        17. If a query is not related to SPS or is too general or you can't find it in database or csv, respond with: "I’m here to assist with information about SPS. If you have any questions related to our services, please let me know."
        The user's query is: '{query}'
        """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    response = generate_response_with_openai(messages)
    return response

@app.route('/')
def index():
    return send_from_directory('.', 'bot.html')

@app.route('/<filename>')
def serve_files(filename):
    return send_from_directory('.', filename)


@app.route('/api/get_response', methods=['POST'])
def get_response_route():
    try:
        data = request.json
        query = data.get('query', '')
        response = get_response(query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /api/get_response route: {e}")
        return jsonify({"response": "Sorry, there was an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)
