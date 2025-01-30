import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Streamlit UI
st.title("💬 Chat with DeepSeek R1 Model")

# Inisialisasi model (pastikan Ollama sudah berjalan)
llm = Ollama(model="deepseek-r1:8b")

# Template prompt untuk chat
chat_prompt = """
You are a helpful assistant. Respond to the following question based on the context:
Question: {question}
Answer:
"""

QA_PROMPT = PromptTemplate.from_template(chat_prompt)

# Membuat chain untuk berkomunikasi dengan model
llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)

# Variabel session untuk menyimpan percakapan
if "history" not in st.session_state:
    st.session_state.history = []

# Antarmuka untuk mengirim pesan
user_input = st.text_input("Ask a question:", "")

if user_input:
    # Menambahkan input pengguna ke history
    st.session_state.history.append(f"You: {user_input}")

    # Mendapatkan respon dari model
    response = llm_chain.run(question=user_input)
    st.session_state.history.append(f"Model: {response}")

# Menampilkan history percakapan
if st.session_state.history:
    for message in st.session_state.history:
        st.write(message)
