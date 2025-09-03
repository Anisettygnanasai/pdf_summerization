# PDF Summarization App 📄

This is a Streamlit application that allows you to upload a PDF file and get a concise summary using the power of Google's Generative AI.

## ✨ Features

  * **PDF Upload:** Easily upload your PDF document.
  * **Intelligent Summarization:** The app processes the text from the PDF and generates a summary using the Gemini family of models.
  * **User-Friendly Interface:** Built with Streamlit for a simple and intuitive user experience.

## ⚙️ How It Works

The application uses the `pypdf` library to extract text from the PDF. This text is then passed to the Google Generative AI API, which uses a large language model to produce a summary. The entire process is orchestrated using the `langchain` framework to handle the document loading, splitting, and communication with the AI model.

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * A Google API key for Generative AI. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Installation

1.  Clone the repository:

    ```sh
    git clone https://github.com/Anisettygnanasai/pdf_summerization.git
    cd pdf_summerization
    ```

2.  Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the root directory of the project.
2.  Add your Google API key to the file:
    ```
    GOOGLE_API_KEY="your_api_key_here"
    ```

### Running the App

Run the Streamlit application from your terminal:

```sh
streamlit run app.py
```

Your web browser will open a new tab with the application.
