import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# Load environment variables from API.env
load_dotenv('API.env')

def summarize_pdf(pdf_file_path, custom_prompt_text):
    """
    Summarizes a PDF using a user-provided prompt with Gemini.
    """
    # 1. Initialize the LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your API.env file.")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )

    # 2. Load and split the PDF document
    loader = PyPDFLoader(pdf_file_path)
    docs_chunks = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    )

    # 3. Create a prompt template from the user's input
    prompt_template_string = custom_prompt_text + """
    
    {text}
    
    """
    prompt = PromptTemplate.from_template(prompt_template_string)

    # 4. Set up and run the summarization chain
    summarize_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    result = summarize_chain.invoke({"input_documents": docs_chunks})

    return result['output_text']

def main():
    st.set_page_config(
        page_title="Custom PDF Summarizer", 
        layout="wide"
    )

    st.title("Custom PDF Summarizer with Gemini")
    st.markdown("This application uses Google's Gemini model to create a tailored summary from a PDF file based on your specific instructions.")

    uploaded_file = st.file_uploader(
        "*1. Upload your PDF file*", 
        type="pdf"
    )

    if uploaded_file:
        temp_file_path = os.path.join(".", "temp_uploaded_file.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Successfully uploaded {uploaded_file.name}")

        custom_prompt = st.text_area(
            "*2. Enter your custom prompt*", 
            height=150,
            placeholder="For example: 'Summarize the key findings of this research paper for a non-technical audience in five bullet points.'"
        )

        if st.button("*Generate Summary*", type="primary"):
            if not custom_prompt:
                st.warning("Please provide a prompt to generate the summary.")
            else:
                with st.spinner("Gemini is working on your summary..."):
                    try:
                        summary = summarize_pdf(temp_file_path, custom_prompt)
                        if summary:
                            st.subheader("Your Custom Summary")
                            st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if _name_ == "_main_":
    main()