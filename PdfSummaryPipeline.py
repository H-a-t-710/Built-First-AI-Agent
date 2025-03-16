import os
import google.generativeai as genai
from groq import Groq
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_env_variable(key):
    """Load environment variables from .env file"""
    try:
        with open(".env", "r") as file:
            for line in file:
                if line.startswith(key + "="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        print(".env file not found.")
    return None

# Initialize API clients
GROQ_API_KEY = load_env_variable("GROQ_API_KEY")
GEMINI_API_KEY = load_env_variable("GEMINI_API_KEY")

if not GROQ_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Both GROQ_API_KEY and GEMINI_API_KEY must be set in .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with OCR fallback"""
    if not Path(pdf_path).exists():
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    
    try:
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"Extracting text from page {page_num}/{num_pages}")
                text += page.extract_text() + "\n"
        
        if not text.strip():  # If no text was extracted, use OCR
            print("No text extracted, attempting OCR...")
            images = convert_from_path(pdf_path)
            text = "".join(pytesseract.image_to_string(image) for image in images)
        
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chunk_text(text, chunk_size=2000):
    """Split text into manageable chunks"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para.split())
        
        if current_size + para_size <= chunk_size:
            current_chunk.append(para)
            current_size += para_size
        else:
            if para_size > chunk_size:
                sentences = para.split('. ')
                for sentence in sentences:
                    sentence_size = len(sentence.split())
                    if current_size + sentence_size > chunk_size:
                        chunks.append('\n\n'.join(current_chunk))
                        current_chunk = [sentence]
                        current_size = sentence_size
                    else:
                        current_chunk.append(sentence)
                        current_size += sentence_size
            else:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def get_groq_summary(text):
    """Get summary using Groq API"""
    if not text:
        return None
    
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        system_prompt = """You are an expert at summarizing text chunks. Please provide a concise summary of the key points and important information in this text section. Use clear and simple language with bullet points for readability."""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
            ],
            temperature=0.7,
            max_tokens=1024,
            stream=True
        )
        
        summary = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            summary += content
        
        return summary.strip()
    except Exception as e:
        print(f"Error generating Groq summary: {e}")
        return None

def get_gemini_summary(text):
    """Get final summary using Gemini API"""
    if not text:
        return None
    
    try:
        system_prompt = """You are an expert at creating comprehensive summaries. Review the following collection of summaries and create a final, well-structured summary that:
1. Identifies the main themes and key points
2. Highlights the most important findings
3. Presents any significant conclusions
4. Uses clear sections and bullet points for readability
5. Maintains academic/technical accuracy while being accessible

Format your response in markdown."""

        chat = gemini_model.start_chat(history=[])
        response = chat.send_message(
            f"{system_prompt}\n\nPlease analyze and summarize the following text:\n\n{text}"
        )
        return response.text
    except Exception as e:
        print(f"Error generating Gemini summary: {e}")
        return None

def process_pdf(pdf_path):
    """Process PDF through both Groq and Gemini models"""
    print(f"\nProcessing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Failed to extract text from PDF")
        return None
    
    print(f"\nExtracted {len(text)} characters of text")
    chunks = chunk_text(text)
    print(f"\nSplit text into {len(chunks)} chunks")
    
    # Process chunks with Groq
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)} with Groq:")
        print("-" * 80)
        summary = get_groq_summary(chunk)
        if summary:
            chunk_summaries.append(f"# Chunk {i} Summary\n\n{summary}\n")
        print("-" * 80)
    
    # Combine chunk summaries
    if not chunk_summaries:
        print("Failed to generate any summaries")
        return None
    
    groq_final = "# Complete Document Summary (Groq)\n\n"
    groq_final += "This is a compilation of all chunk summaries from the document:\n\n"
    groq_final += "\n---\n\n".join(chunk_summaries)
    
    # Process with Gemini
    print("\nGenerating final summary with Gemini...")
    gemini_final = get_gemini_summary(groq_final)
    
    return {
        "extracted_text": text,
        "groq_summaries": chunk_summaries,
        "groq_final": groq_final,
        "gemini_final": gemini_final
    }

if __name__ == "__main__":
    input_dir = Path("Inputs")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"\nProcessing file: {pdf_file}")
        result = process_pdf(pdf_file)
        
        if result:
            # Create directory for this PDF's results
            pdf_output_dir = output_dir / pdf_file.stem
            pdf_output_dir.mkdir(exist_ok=True)
            
            # Save extracted text
            with open(pdf_output_dir / "extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(result["extracted_text"])
            
            # Save Groq summaries
            with open(pdf_output_dir / "groq_output.txt", "w", encoding="utf-8") as f:
                f.write(result["groq_final"])
            
            # Save Gemini summary
            with open(pdf_output_dir / "gemini_output.txt", "w", encoding="utf-8") as f:
                f.write(result["gemini_final"])
            
            print(f"\nResults for {pdf_file.name} have been saved to: {pdf_output_dir}")
        else:
            print(f"\nFailed to process {pdf_file.name}") 