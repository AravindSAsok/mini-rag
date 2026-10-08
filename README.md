# Mini RAG Assistant

A beginner-friendly document retrieval demo. Ask a question in the browser and see which passages from the knowledge base are most relevant.

The Streamlit page demonstrates the retrieval part of RAG only. It displays passages and similarity scores; it does not generate an LLM answer.

## What is RAG?

RAG means retrieval-augmented generation. A full RAG system first retrieves relevant information from documents, then gives those passages to a language model to draft an answer. This project currently demonstrates retrieval. No LLM, Ollama, or paid API is needed for the web app.

## Project structure

~~~text
mini-rag/
├── data/
│   └── rag.txt              # Knowledge base
├── src/
│   └── main.py              # Existing chunking, embedding, and search logic
├── streamlit_app.py         # Browser interface using the existing search logic
├── requirements.txt         # Dependencies for the CLI and web UI
├── .gitignore               # Excludes local environments, caches, and backups
└── README.md
~~~

The local data/rag.txt.backup file is a backup, not a supported document type. It is excluded from Git and is not loaded by the app.

## Run the website locally

Open a terminal in D:\mini-rag, then run:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
~~~

Streamlit prints a local URL, usually http://localhost:8501. Open it in your browser, enter a question, and press Search.

If the virtual environment is activated first, you can use the shorter command:

~~~powershell
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
~~~

The first start downloads the all-MiniLM-L6-v2 embedding model if it is not already cached. This requires internet access. The app builds document embeddings at startup and caches the loaded assistant while that app process is running.

### Run the original command-line version

The CLI remains available and uses the same RAG code:

~~~powershell
.\.venv\Scripts\python.exe src\main.py
~~~

For a single retrieval query:

~~~powershell
.\.venv\Scripts\python.exe src\main.py search "Which members were in EXO-M?"
~~~

## How retrieval works

1. The existing code reads .txt and .md documents from the data folder.
2. It splits the text into chunks. Blank lines separate chunks; long paragraphs are split with overlap.
3. sentence-transformers and all-MiniLM-L6-v2 convert each chunk into a numeric embedding.
4. The app converts the user's question into an embedding with the same model.
5. The existing search function compares normalized embeddings using cosine similarity and returns the top three chunks.
6. The page shows each passage, its source file, chunk number, and similarity score.

A similarity score is a ranking signal, not a probability that a passage is correct. Read the displayed passages to check whether they support the question.

The Streamlit app uses Streamlit's resource cache so the model and document embeddings are loaded once per running app process, not rebuilt on every button click. The index is still in memory and is rebuilt after the app restarts.

## Deploy free with Streamlit Community Cloud

The simple public-app workflow uses a GitHub repository. Only publish documents and code that you are comfortable making public.

1. Create a public GitHub repository.
2. Upload streamlit_app.py, the src folder, data/rag.txt, requirements.txt, .gitignore, and this README.
3. Do not upload .venv, data/rag.txt.backup, Python cache files, or model caches. The .gitignore file excludes these local files.
4. Sign in to https://share.streamlit.io/ with GitHub.
5. Choose Create app, select your repository and branch, and set the app entry file to streamlit_app.py.
6. Click Deploy. Streamlit will install the dependencies and give you a shareable URL.

Community Cloud apps sleep after 12 hours without traffic, so the first visit after sleep may take longer while the app starts and rebuilds the index. Resource limits can vary. If deployment reports a dependency or memory error, check the app logs.

Official guides: [deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [manage app resources](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app).

No Dockerfile is needed for this setup. Nothing is committed, uploaded, or deployed automatically by this project.

## Current limits

- The web UI retrieves passages but does not generate answers.
- The web UI does not use Ollama or a paid API.
- The loader supports .txt and .md files, not PDF or Word documents.
- The embeddings are kept in memory rather than saved to a persistent vector database.
