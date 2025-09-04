import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# Load environment variables from API.env file
load_dotenv('API.env')

def get_llm():
    """
    Instantiates and returns the LLM model.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your API.env file.")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )
    return llm

def process_pdf(pdf_file_path, custom_prompt_text, chain_type, chunk_size, chunk_overlap):
    """
    Loads, splits, and summarizes a PDF based on the provided parameters.
    """
    llm = get_llm()
    if not llm:
        return None

    # Load and split the PDF using dynamic chunk parameters
    loader = PyPDFLoader(pdf_file_path)
    docs_chunks = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    )

    # Create and run the summarization chain based on the selected type
    if chain_type == "stuff":
        # The stuff method uses a single prompt for the entire document
        prompt_template = custom_prompt_text + """

        {text}

        """
        prompt = PromptTemplate.from_template(prompt_template)
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    elif chain_type == "map_reduce":
        # The map_reduce method summarizes chunks first, then combines them
        map_prompt_template = "Summarize the following text chunk concisely:\n\n{text}"
        map_prompt = PromptTemplate.from_template(map_prompt_template)

        combine_prompt_template = custom_prompt_text + """

        {text}
        """
        combine_prompt = PromptTemplate.from_template(combine_prompt_template)
        
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )
    else:
        st.error("Invalid chain type selected.")
        return None

    result = chain.invoke({"input_documents": docs_chunks})
    return result['output_text']

def main():
    """
    The main function to run the Streamlit application with a refined UI.
    """
    st.set_page_config(page_title="PDF Summarizer", page_icon="📝", layout="wide")

    # --- Main Page Content ---
    st.title("📄 PDF Summarizer with Gemini")
    st.markdown("### Get custom summaries from your PDF documents instantly!")
    
    # Placeholder for the summary output
    summary_placeholder = st.container()

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Use a form to group all the inputs and have a single submit button
        with st.form("input_form"):
            st.subheader("1. Upload Your PDF")
            uploaded_file = st.file_uploader(
                "Choose a PDF file", type="pdf", label_visibility="collapsed"
            )

            st.subheader("2. Enter Your Prompt")
            custom_prompt = st.text_area(
                "What should the summary focus on?",
                height=150,
                placeholder="e.g., 'Summarize the key findings for a non-technical audience in five bullet points.'"
            )
            
            st.subheader("3. Advanced Settings")
            chain_type = st.selectbox(
                "Summarization Method",
                ("stuff", "map_reduce"),
                help="**Stuff**: Faster, single-step. Best for smaller docs. **MapReduce**: Slower, multi-step. Handles very large documents."
            )

            # Use columns for a cleaner layout of sliders
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider(
                    "Chunk Size", 500, 20000, 4000, 500,
                    help="The maximum number of characters in each text chunk."
                )
            with col2:
                chunk_overlap = st.slider(
                    "Chunk Overlap", 0, 5000, 200, 50,
                    help="The number of characters to overlap between adjacent chunks."
                )
                
            # The submit button for the form
            submitted = st.form_submit_button("Generate Summary", type="primary", use_container_width=True)

    # --- Processing and Display Logic ---
    if submitted:
        if uploaded_file is None:
            st.warning("Please upload a PDF file first.")
        elif not custom_prompt.strip():
            st.warning("Please enter a custom prompt.")
        else:
            with st.spinner("Processing document and generating summary... 🧠"):
                try:
                    # Save the uploaded file to a temporary location
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Call the processing function
                    summary = process_pdf(
                        temp_file_path,
                        custom_prompt,
                        chain_type,
                        chunk_size,
                        chunk_overlap
                    )
                    
                    # Display the summary in the placeholder on the main page
                    if summary:
                        with summary_placeholder:
                            st.success("Summary Generated!")
                            st.markdown("---")
                            st.subheader(f"Summary of `{uploaded_file.name}`")
                            st.markdown(summary)
                    else:
                         with summary_placeholder:
                            st.error("Failed to generate summary. Please check the PDF or API key.")

                except Exception as e:
                    with summary_placeholder:
                        st.error(f"An error occurred: {e}")
                finally:
                    # Clean up the temporary file
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
    else:
        with summary_placeholder:
            st.info("Upload a PDF and provide a prompt in the sidebar to get started.")

if __name__ == "__main__":
    main()
