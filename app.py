#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import openai
from openai import OpenAI

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to convert natural language to SQL using OpenAI
def generate_sql_from_nl(nl_query: str) -> str:
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": f"Convert this natural language request into a SQL query:\n\n{nl_query}"}],
        temperature=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

# Function to explain SQL using OpenAI
def explain_sql_from_query(sql_query: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Explain the following SQL query in simple English:\n\n{sql_query}"}],
        temperature=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

# Set up the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/generate_sql', methods=['POST'])
def generate_sql():
    data = request.get_json()
    nl_query = data.get('query', '')
    if not nl_query:
        return jsonify({'error': 'No query provided'}), 400
    sql_query = generate_sql_from_nl(nl_query)
    return jsonify({'sql': sql_query})

@app.route('/explain_sql', methods=['POST'])
def explain_sql():
    data = request.get_json()
    sql_query = data.get('query', '')
    if not sql_query:
        return jsonify({'error': 'No SQL query provided'}), 400
    explanation = explain_sql_from_query(sql_query)
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)
