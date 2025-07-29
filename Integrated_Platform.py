import streamlit as st
import langchain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import ansible_runner
from kafka import KafkaConsumer
import requests
import json
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load Hugging Face dataset and model
dataset = load_dataset("imdb", split="test")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Streamlit UI Header
st.title("Integrated Platform Engineering Assistant")

# Step 1: Knowledge Base Retrieval using LangChain & FAISS
def retrieve_kb_answer(query):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("./kb_faiss_index", embeddings)
    docs = vectorstore.similarity_search(query, k=3)
    return [doc.page_content for doc in docs]

query = st.text_input("Enter your troubleshooting query:")
if st.button("Search KB"):
    st.write("Knowledge Base Results:")
    st.write(retrieve_kb_answer(query))

# Step 2: Automated Troubleshooting with Ansible
def run_ansible_playbook(playbook_path, extra_vars={}):
    result = ansible_runner.run(private_data_dir='.', playbook=playbook_path, extravars=extra_vars)
    return result.stdout.read()

playbook_path = st.text_input("⚡ Enter Ansible Playbook Path:")
if st.button("Run Ansible Playbook"):
    st.write("Running Playbook...")
    st.write(run_ansible_playbook(playbook_path))

# Step 3: Real-time Log Analysis with Kafka
def consume_logs():
    consumer = KafkaConsumer('logs', bootstrap_servers=['localhost:9092'], auto_offset_reset='latest')
    for message in consumer:
        log_data = message.value.decode('utf-8')
        st.write("[Log Analysis] Processing log:", log_data)
        # AI-based log anomaly detection logic can be integrated here

if st.button("Start Log Consumption"):
    st.write("Listening to logs...")
    consume_logs()

# Step 4: ServiceNow CMDB Querying
def query_cmdb(ci_name):
    url = "https://your-instance.service-now.com/api/now/table/cmdb_ci?sysparm_query=name=" + ci_name
    headers = {"Authorization": "Bearer YOUR_SERVICENOW_TOKEN", "Content-Type": "application/json"}
    response = requests.get(url, headers=headers)
    return json.loads(response.text)

ci_name = st.text_input("Enter CI Name for CMDB Lookup:")
if st.button("Query CMDB"):
    st.write("CMDB Query Result:")
    st.write(query_cmdb(ci_name))

# Example Sentiment Analysis using Hugging Face Model
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    sentiment = "Positive" if outputs.logits.argmax().item() == 1 else "Negative"
    return sentiment

text_input = st.text_area("Enter text for sentiment analysis:")
if st.button("Analyze Sentiment"):
    st.write("Sentiment:", analyze_sentiment(text_input))
