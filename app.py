from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import pytesseract
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure Tesseract path (if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # Step 1: Extract text using OpenCV and Tesseract OCR
        image = cv2.imread(file_path)

        # Resize the image to improve OCR accuracy
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply noise reduction using Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11
        )

        # Use Tesseract to extract text
        extracted_text = pytesseract.image_to_string(binary, config='--psm 6')

        # Debugging: Print extracted text
        print("Extracted Text:", extracted_text)

        # Handle missing text
        if not extracted_text.strip():
            return jsonify({
                "error": "No text could be extracted from the image. Please ensure the image is clear and contains legible text."
            }), 400

        # Step 2: Correct spelling errors and improve text using GPT-4
        corrected_text = correct_text_with_gpt(extracted_text)

        # Step 3: Summarize or format the text into structured points
        structured_notes = summarize_text_with_gpt(corrected_text)

        # Return the processed text
        return jsonify({
            "extracted_text": extracted_text,
            "corrected_text": corrected_text,
            "structured_notes": structured_notes
        })

    except Exception as e:
        # Log the full exception details
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

# Helper function to correct text using GPT-4
def correct_text_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a text correction assistant."},
            {"role": "user", "content": f"Correct the following text: {text}"}
        ]
    )
    return response.choices[0].message.content

# Helper function to summarize text using GPT-4
def summarize_text_with_gpt(text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the following text into structured points: {text}"}
        ]
    )
    return response.choices[0].message.content

# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)