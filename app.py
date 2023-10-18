import streamlit as st
from io import StringIO
from utils import *

st.title("DOC2MCQ🙋")
st.divider()
st.subheader("CITE3 DEMO version")
st.caption("Upload your document and get multiple choice questions!")
model, tokenizer = load_Summarization_model_tokenizer()
model_gen,tokenizer_gen = load_QuestionGeneration_model_tokenizer()
uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)

if uploaded_file is not None:
    text_list = doc2Chunk(uploaded_file)
    for text in text_list:
        summarized_text = summarizer(text, model=model, tokenizer=tokenizer)
        keywordList = keywordExtraction(summarized_text)
        for keyword in keywordList:
            question = generate_Question(model=model_gen, tokenizer= tokenizer_gen, text=summarized_text, keyword=keyword)
            st.write(f'question: {question}\n Answer: {keyword}\n')

        st.write(f'summarized_text: {summarized_text}')
        st.write(f'keywords: {keywordList}')
