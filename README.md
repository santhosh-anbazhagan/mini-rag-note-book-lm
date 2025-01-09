# Mini Notebook LM 
### A RAG System with OpenAI, LangChain, and Streamlit

Mini Notebook LM is a sophisticated Retrieval-Augmented Generation (RAG) system that transforms document interaction through advanced natural language processing. By combining OpenAI's powerful GPT models with LangChain's document processing capabilities, users can easily extract insights from their documents through a streamlined Streamlit interface.


## 🖥️ User Interface
<img src="/images/src-1.png" alt="Document Upload Interface" />
<img src="/images/src-2.png" alt="Question and Answer Interface" />

## 🎯 Key Features

Our system offers a comprehensive suite of document analysis tools:

- Document Processing: Upload and analyze PDF or TXT files up to 200MB
- Smart Chunking: Automatic document segmentation for optimal token utilization
- Visual Analytics: Interactive visualization of token distribution across document segments
- Intelligent Q&A: Natural language question answering powered by GPT models
- User-Friendly Interface: Clean, intuitive design built with Streamlit

## 🛠️ Technology Stack

The application leverages cutting-edge technologies:

- Python: Powers the core application logic and data processing
- Streamlit: Provides the interactive user interface framework
- LangChain: Enables sophisticated document handling and query processing
- OpenAI GPT Models: Delivers advanced natural language understanding

## 📋 Installation Guide

Follow these steps to set up Mini Notebook LM locally:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/mini-notebook-lm.git
cd mini-notebook-lm
```

### 2. Install Dependencies

Ensure Python 3.8+ is installed, then set up the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure OpenAI API

Set up your OpenAI credentials:

1. Obtain an API key from OpenAI
2. Create a `.env` file in the project root
3. Add your API key:
   ```
   OPENAI_API_KEY=your-api-key
   ```

### 4. Launch the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

## 📂 Project Structure

The project maintains a clear, modular organization:

```
mini-notebook-lm/
│
├── app.py                 # Main Streamlit application
├── utils/
│   ├── document_parser.py # Document processing logic
│   └── query_handler.py   # Query management system
├── requirements.txt       # Project dependencies
├── .env                  # Environment configuration
└── README.md            # Project documentation
```

## 🔄 System Architecture

Our RAG system operates through a streamlined process:

1. Document Intake: Users upload their PDF or TXT documents
2. Processing Pipeline: Documents undergo smart chunking for efficient processing
3. Visual Feedback: Token distribution visualization provides system transparency
4. Interactive Q&A: Users engage with their documents through natural language queries

## 🚀 Future Development

We have an exciting roadmap for future enhancements:

- Extended Format Support: Integration of DOCX and other document formats
- Advanced Storage: Implementation of vector databases for improved retrieval
- Global Reach: Addition of multi-language processing capabilities
- Cloud Deployment: Migration to cloud platforms (AWS, Azure) for scalability

## 🌐 Deployment

Access the live application through our GitHub deployment [link].

## 🤝 Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Submit a pull request

## 📧 Contact Information

For support or inquiries, reach out to:

**Santhosh Kumar A**
- Email: santhoshanbazhagan1910@gmail.com
- GitHub: [Santhosh-Anbazhagn](https://github.com/Santhosh-Anbazhagan)

## 📜 License

This project is distributed under the MIT License. See the LICENSE file for complete details.

---

*Built with ❤️ by Santhosh Anbazhgan*
