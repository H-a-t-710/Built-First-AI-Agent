#Part - 1

from groq import Groq
import PyPDF2
from pathlib import Path

# Load environment variables manually
def load_env_variable(key):
    try:
        with open(".env", "r") as file:
            for line in file:
                if line.startswith(key + "="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        print(".env file not found.")
    return None

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Returns: The extracted text as a string.
    """
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
        
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def chunk_text(text, chunk_size=2000):
    """
    Split text into chunks of approximately chunk_size words.
    Tries to split on paragraph breaks or sentences to maintain context.
    """
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        # Rough word count (split on spaces)
        para_size = len(para.split())
        
        if current_size + para_size <= chunk_size:
            current_chunk.append(para)
            current_size += para_size
        else:
            # If the paragraph itself is too big, split it into sentences
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
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def get_summary_from_llm(text, is_final=False):
    """
    Gets a summary of the text using the Groq LLM API.
    Returns: The summarized text as a string.
    """
    if not text:
        print("Error: No text provided for summarization")
        return None

    # Initialize Groq client with API key
    GROQ_API_KEY = load_env_variable("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in .env file")
        return None
    
    client = Groq(api_key=GROQ_API_KEY)
    
    try:
        # Create the summarization prompt
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
        
        # Process the streaming response
        summary = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            summary += content
        
        return summary.strip()
    
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def process_pdf(pdf_path):
    """
    Main function to process a PDF file and get its summary.
    """
    # Extract text from PDF
    print(f"\nProcessing PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("Failed to extract text from PDF")
        return
    
    # Get length of extracted text
    print(f"\nExtracted {len(text)} characters of text")
    
    # Split text into chunks
    chunks = chunk_text(text)
    print(f"\nSplit text into {len(chunks)} chunks")
    
    # Get summaries for each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing chunk {i}/{len(chunks)}:")
        print("-" * 80)
        summary = get_summary_from_llm(chunk)
        if summary:
            chunk_summaries.append(f"# Chunk {i} Summary\n\n{summary}\n")
        print("-" * 80)
    
    # Combine all chunk summaries into the final summary
    if chunk_summaries:
        print("\nGenerating final summary...")
        final_summary = "# Complete Document Summary\n\n"
        final_summary += "This is a compilation of all chunk summaries from the document:\n\n"
        final_summary += "\n---\n\n".join(chunk_summaries)
    else:
        print("Failed to generate any summaries")
        return None
    
    return {
        "extracted_text": text,
        "chunk_summaries": chunk_summaries,
        "final_summary": final_summary
    }

# Example usage
if __name__ == "__main__":
    input_dir = Path("Inputs")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Process all PDF files in the input directory
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"\nProcessing file: {pdf_file}")
        result = process_pdf(pdf_file)
        
        if result:
            # Create a directory for this PDF's results
            pdf_output_dir = output_dir / pdf_file.stem
            pdf_output_dir.mkdir(exist_ok=True)
            
            # Save extracted text
            with open(pdf_output_dir / "extracted_text.txt", "w", encoding="utf-8") as f:
                f.write(result["extracted_text"])
            
            # Save individual chunk summaries
            with open(pdf_output_dir / "chunk_summaries.txt", "w", encoding="utf-8") as f:
                for i, summary in enumerate(result["chunk_summaries"], 1):
                    f.write(f"Chunk {i} Summary:\n")
                    f.write("-" * 40 + "\n")
                    f.write(summary + "\n\n")
            
            # Save final summary
            with open(pdf_output_dir / "final_summary.txt", "w", encoding="utf-8") as f:
                f.write(result["final_summary"])
            
            print(f"\nResults for {pdf_file.name} have been saved to: {pdf_output_dir}")
        else:
            print(f"\nFailed to process {pdf_file.name}")