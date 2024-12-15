import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from hybrid_llm import upload, query


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def handle_userinput(user_question):
    history = st.session_state.chat_history
    st.session_state.response = query(user_question, history)
    st.session_state.chat_history = {user_question: st.session_state.response, **history}

    if len(st.session_state.chat_history) > 7:
        st.session_state.chat_history = dict(list(st.session_state.chat_history.items())[:7])


    for key, value in st.session_state.chat_history.items():
        st.write(user_template.replace("{{MSG}}", key), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", value), unsafe_allow_html=True)


def main():
    load_dotenv()


    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "response" not in st.session_state:
        st.session_state.response = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                upload(text_chunks)


if __name__ == '__main__':
    main()