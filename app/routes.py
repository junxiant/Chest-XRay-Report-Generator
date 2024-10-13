import os
from flask import render_template, request, jsonify
from app import app
from app.model import CvT2DistilGPT2MIMICXRChenInference
from PIL import Image
import io
import re

# Initialize the model
model = CvT2DistilGPT2MIMICXRChenInference(
    "./checkpoints/epoch=8-val_chen_cider=0.425092.ckpt",
    "./checkpoints/"
)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        generated_report = model.generate(image)
        
        # Format the report
        formatted_report = format_report(generated_report)
        
        return jsonify({'report': formatted_report})

def format_report(report):
    # Split the report into sentences
    sentences = re.split('(?<=[.!?]) +', report)
    
    # Capitalize each sentence and create a list item
    formatted_sentences = [f"• {sentence.capitalize()}" for sentence in sentences if sentence]
    
    # Join the formatted sentences with newlines
    return "\n".join(formatted_sentences)