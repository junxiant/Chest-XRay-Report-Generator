from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from inference import XrayReportGenerator

def init_model():
    if not hasattr(init_model, 'generator'):
        init_model.generator = XrayReportGenerator()
    return init_model.generator

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        app.generator = init_model()
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            try:
                report = app.generator.generate_report(file)
                return jsonify({'report': report})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)