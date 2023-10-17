import streamlit as st
from io import StringIO
from utils import *

st.title("DOC2MCQ🙋")
st.divider()
st.subheader("CITE3 DEMO version")
st.caption("Upload your document and get multiple choice questions!")
model, tokenizer = load_Summarization_model_tokenizer()
uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

if uploaded_file is not None:
    text_list = doc2Chunk(uploaded_file)
    # st.write(StringIO(uploaded_file.getvalue().decode("utf-8")).read()) 

    for text in text_list:
        summarized_text = summarizer(text, model=model, tokenizer=tokenizer)
        keywordList = keywordExtraction(summarized_text)
        st.write(f'summarized_text: {summarized_text}')
        st.write(f'keywords: {keywordList}')



    # for uploaded_file in uploaded_files:
    #     bytes_data = uploaded_file.read()
    #     st.write("filename:", uploaded_file.name)
    #     st.write(bytes_data)
