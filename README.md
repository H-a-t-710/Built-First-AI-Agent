# PDF Summary Pipeline

This project provides a powerful PDF processing pipeline that combines multiple AI models to generate comprehensive document summaries. The pipeline uses both Groq and Gemini AI models in sequence to provide detailed and well-structured summaries of PDF documents.

## Features

- **Multi-Model Pipeline**:
  - Groq LLM for initial chunk processing
  - Gemini Pro for final summary synthesis
  - Combines strengths of both models

- **PDF Processing**:
  - Extract text from PDF files
  - OCR support for scanned documents
  - Smart text chunking for large documents
  - Batch processing of multiple PDFs

- **Robust Processing**:
  - Error handling and retry mechanisms
  - Progress logging and status updates
  - Token limit management
  - Structured output organization

## Prerequisites

1. Python 3.8 or higher
2. Required Python packages (install using `pip install -r requirements.txt`):
   ```
   google-generativeai
   groq
   PyPDF2
   pytesseract
   pdf2image
   ```

3. For OCR support:
   - Install Tesseract OCR:
     - Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
     - Linux: `sudo apt-get install tesseract-ocr`
     - macOS: `brew install tesseract`

## Setup

1. Clone the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. Ensure the following directory structure:
   ```
   Built-First-AI-Agent/
   ├── Inputs/          # Place PDF files here
   ├── output/          # Results will be saved here
   ├── PdfSummaryPipeline.py
   ├── requirements.txt
   └── .env
   ```

## Usage

1. Place your PDF files in the `Inputs` directory

2. Run the pipeline:
   ```bash
   python PdfSummaryPipeline.py
   ```

3. Results will be saved in the `output` directory:
   - Each PDF gets its own subdirectory
   - Three files are generated for each PDF:
     - `extracted_text.txt`: Raw text from the PDF
     - `groq_output.txt`: Groq's chunk-by-chunk summary
     - `gemini_output.txt`: Gemini's final comprehensive summary

## How It Works

1. **Text Extraction**:
   - Extracts text from PDF
   - Falls back to OCR if needed
   - Splits text into manageable chunks

2. **Groq Processing**:
   - Processes each chunk individually
   - Generates bullet-point summaries
   - Maintains technical accuracy

3. **Gemini Processing**:
   - Takes Groq's summaries as input
   - Synthesizes a comprehensive final summary
   - Adds structure and organization

## Configuration

The pipeline can be configured through several parameters:

- `chunk_size`: Words per chunk (default: 2000)
- `max_output_tokens`: Token limit for AI responses (default: 8192)
- `temperature`: AI creativity level (default: 0.7)

## Error Handling

The pipeline includes robust error handling for:
- Missing or corrupted PDF files
- Text extraction failures
- OCR processing issues
- API rate limits and quotas
- Token limit management

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 