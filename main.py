import streamlit as st
from testing import get_qa_chain, create_vector_db

st.title("Investment banker GPT 💵")
btn = st.button("Create knowledge")
if btn:
    create_vector_db()

question= st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response= chain(question)

    st.header("Answer")
    st.write(response["result"])
