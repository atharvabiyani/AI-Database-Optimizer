# AI Database Optimizer

**AI Database Optimizer** is an intelligent assistant for SQL productivity and database optimization. It combines Natural Language Processing with Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to help users:

- Generate SQL queries from natural language
- Explain complex SQL queries in plain English
- Analyze Snowflake database structures
- Detect redundant tables, views, and naming issues
- Provide interactive optimization suggestions

# Poster
[📄 View Project Poster (PDF)](./poster_39.USAA%20-%20DatabAIse.pdf)

---

## 🧠 Features

### 🔄 Natural Language ↔ SQL
- **Generate SQL** from plain English queries
- **Explain SQL** using GPT-3.5
- Components: `app.py` + `frontend.html`

### 🧠 RAG + LLM Database Optimization
- Connects to Snowflake to extract schema metadata
- Embeds structure using HuggingFace + ChromaDB
- Accepts user queries and generates optimization suggestions
- Components: `final_connection_with_rag_and_llm.py` + `frontend_rag.html`

## 🚀 Getting Started

### Requirements

- Python 3.9+
- OpenAI API Key
- Snowflake account & credentials
- HuggingFace model access (`sentence-transformers/all-MiniLM-L6-v2`)

### Installation


## 🌐 Interfaces

- `/` → Natural Language to SQL + SQL Explanation (`frontend.html`)
- `/rag_query` → Ask optimization questions (`frontend_rag.html`)

---

## 💼 Example Use Cases

- Analysts asking: “Show top 10 products by sales”
- Developers explaining legacy SQL queries
- DBAs identifying duplicate tables
- Engineers cleaning up schema structure

---

## 📌 Notes

- Uses `gpt-3.5-turbo` for SQL generation and explanation
- Embeds Snowflake schema/table metadata into `ChromaDB`
- Utilizes vector search for high-relevance schema and query analysis

---


## 👥 Contributors
Technical Team:
- Atharva Biyani 
- Dharshini Mahesh 
- Neha Kandula 
- Sneha Elangovan 
- Suvel Sunilnath 

Faculty Advisor:
- Dr. Gopal Gupta

Sponsors & Support Staff:
- Steven Kirtzic, PhD
- Jeffrey Gordon, MS
