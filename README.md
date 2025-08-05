# LLM Query Engine Project

This project is a simple Query Engine built using Large Language Models (LLMs) like OpenAI and Cohere. It enables users to query a local document and receive intelligent answers using a vector-based search and LLM response mechanism.

## 🚀 Features

- **PDF Loader**: Loads your PDF document as input.
- **Text Splitter**: Efficiently splits your document into chunks for better indexing.
- **Embeddings**: Uses LLM-based embeddings (Cohere/OpenAI) to convert text into vector format.
- **Vector Store (FAISS)**: Stores the vectorized data and allows similarity search.
- **LLM Integration**: Integrates OpenAI/Cohere to answer your queries based on context.

---

## 📁 Project Structure

```
LLM_Query_Project/
│
├── main.py                  # Main execution script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🔧 Requirements

- Python 3.8+
- Install required packages:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

You must set the following environment variables before running the code:

```bash
export COHERE_API_KEY=your_cohere_api_key
export OPENAI_API_KEY=your_openai_api_key
```

> Replace `your_cohere_api_key` and `your_openai_api_key` with your actual API keys.

---

## 📌 How It Works

1. Upload a PDF document.
2. The document is split and embedded using an LLM.
3. The embeddings are stored using FAISS.
4. When you enter a query, the most relevant chunks are retrieved.
5. An answer is generated using OpenAI or Cohere based on the context.

---

## 🛠️ Run the App

```bash
python main.py
```

You will be prompted to enter a query based on the loaded PDF document.

---

## 💡 Example Use Case

Use this tool to:
- Summarize academic papers
- Query large PDFs efficiently
- Create knowledge engines for documentation

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

[MIT](https://choosealicense.com/licenses/mit/)
