import os
import google.generativeai as genai
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the API key is set in the environment variables
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please set it as an environment variable.")

# Configure the Generative AI client with the API key
genai.configure(api_key="AIzaSyDZvSPlKaB7w327fiXJHvNzZGhLYpNIoGA")

# Define the configuration for text generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the Generative Model
try:
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
    )
except Exception as e:
    logging.error(f"Error initializing Generative Model: {e}. Possible reasons include API quota exhaustion, invalid configuration, or network issues. Check your API usage dashboard and ensure your configuration parameters are correct.")
    model = None

# Function to extract text from a PDF file
# Uses OCR for image-based PDFs
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() + "\n" for page in reader.pages if page.extract_text())

        if not text.strip():  # If no text was extracted, use OCR
            logging.info("No text extracted, attempting OCR...")
            images = convert_from_path(pdf_path)
            ocr_text = "".join(pytesseract.image_to_string(image) for image in images)
            return ocr_text

        return text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Function to process extracted text using the generative AI model
def process_text_with_ai(input_text, retries=3, backoff_factor=2):
    if model is None:
        logging.error("Generative model is not initialized. Ensure the initialization was successful and check for quota or configuration issues.")
        return ""

    # Ensure input text does not exceed token limit
    max_input_length = 8192 - 100  # Reserve tokens for response and metadata
    if len(input_text.split()) > max_input_length:
        logging.warning("Input text exceeds token limit. Truncating input.")
        input_text = " ".join(input_text.split()[:max_input_length])

    for attempt in range(retries):
        try:
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(input_text)
            if response.text.strip():
                return response.text
            else:
                logging.warning("AI returned an empty response. Retrying...")
                time.sleep(backoff_factor ** attempt)
        except Exception as e:
            logging.error(f"Error processing text with AI: {e}")
            if "429" in str(e):
                wait_time = backoff_factor ** attempt
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                break
    return ""

# Directory containing PDF files
pdf_directory = "Inputs"

# Ensure the directory exists
if not os.path.exists(pdf_directory):
    logging.error(f"The directory {pdf_directory} does not exist.")
    exit(1)

# Loop through each PDF file in the directory
for pdf_file in os.listdir(pdf_directory):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_directory, pdf_file)
        logging.info(f"Processing: {pdf_file}")

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        if extracted_text.strip():
            # Process the text using the AI model
            ai_response = process_text_with_ai(extracted_text)

            # Handle empty AI responses
            if not ai_response.strip():
                logging.warning(f"AI response for {pdf_file} was empty even after retries.")
                logging.info(f"Saving extracted text for manual processing: {pdf_file}")
                with open(f"{pdf_file}_extracted.txt", "w", encoding="utf-8") as f:
                    f.write(extracted_text)
            else:
                logging.info(f"AI Response for {pdf_file}:")
                print(ai_response)
                print("=" * 80)
        else:
            logging.warning(f"No text extracted from {pdf_file}.")
