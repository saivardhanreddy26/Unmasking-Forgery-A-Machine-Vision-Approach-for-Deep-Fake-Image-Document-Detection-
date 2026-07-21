
from flask import Flask, render_template, request, jsonify, url_for, flash, redirect
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import base64
from config import Config
from models.forgery_detector import ForgeryDetector

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

# Initialize forgery detector with Config class instance
config = Config()
detector = ForgeryDetector(config)

# Load trained models
try:
    detector.load_models(
        str(config.RESNET_MODEL_PATH),
        str(config.ANN_MODEL_PATH)
    )
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Warning: Pre-trained models not found. Training a new model...")
    # You can optionally trigger the training script here
    # import subprocess
    # subprocess.run(["python", "training/train_model.py"])
    # detector.load_models(str(config.RESNET_MODEL_PATH), str(config.ANN_MODEL_PATH))
except Exception as e:
    print(f"Error: Could not load models - {e}")
    print("Please ensure the model files are correctly placed and not corrupted.")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image_array):
    """Convert numpy array to base64 for display"""
    _, buffer = cv2.imencode('.png', image_array)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/detect')
def detect():
    """Detection page"""
    return render_template('detect.html')

@app.route('/visualization')
def visualization():
    """Visualization page"""
    return render_template('visualization.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


@app.route('/detect-document')
def detect_document():
    """Document detection page"""
    return render_template('detect_document.html')




@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for forgery detection"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect forgery
        results = detector.detect_forgery(filepath)
        
        # Convert forensic maps to base64
        forensic_maps_b64 = {}
        for key, img in results['forensic_maps'].items():
            forensic_maps_b64[key] = image_to_base64(img)
        
        # Prepare response
        response = {
            'success': True,
            'verdict': results['verdict'],
            'is_forged': results['is_forged'],
            'confidence': round(results['confidence'], 4),
            'forgery_probability': round(results['forgery_probability'], 2),
            'authenticity_probability': round(results['authenticity_probability'], 2),
            'forensic_maps': forensic_maps_b64,
            'filename': filename,
            'upload_url': url_for('static', filename=f'uploads/{filename}')
        }
        
        # Clean up uploaded file after processing
        # os.remove(filepath)  # Uncomment if you want to delete after processing
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-detect', methods=['POST'])
def api_batch_detect():
    """API endpoint for batch forgery detection"""
    try:
        # Check if files are present
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        
        results_list = []
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Detect forgery
                result = detector.detect_forgery(filepath)
                
                results_list.append({
                    'filename': filename,
                    'verdict': result['verdict'],
                    'confidence': round(result['confidence'], 4),
                    'forgery_probability': round(result['forgery_probability'], 2)
                })
        
        return jsonify({
            'success': True,
            'results': results_list,
            'total_processed': len(results_list)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/detect-document', methods=['POST'])
def api_detect_document():
    """API endpoint for document forgery detection"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Import document detector
        from models.document_forgery_detector import DocumentForgeryDetector
        
        # Detect document forgery
        doc_detector = DocumentForgeryDetector()
        results = doc_detector.detect_document_forgery(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'is_forged': results['is_forged'],
            'confidence': round(results['confidence'], 4),
            'forgery_type': results['forgery_type'],
            'quality_metrics': results['quality_metrics'],
            'text_analysis': results['text_analysis'],
            'visual_analysis': results['visual_analysis'],
            'filename': filename,
            'upload_url': url_for('static', filename=f'uploads/{filename}')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500







@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
