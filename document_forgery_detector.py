import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from collections import Counter
import Levenshtein
from scipy import stats
import json

class DocumentForgeryDetector:
    """
    Document forgery detection using OCR and computer vision
    Detects tampering in documents like IDs, certificates, invoices, etc.
    """
    
    def __init__(self):
        # Set tesseract path if needed (Windows)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        self.forgery_indicators = []
        self.confidence_score = 0.0
    
    def convert_to_native_types(self, obj):
        """
        Convert NumPy types to native Python types for JSON serialization
        """
        if isinstance(obj, dict):
            return {key: self.convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_native_types(item) for item in obj]
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def detect_document_forgery(self, image_path):
        """
        Main function to detect document forgery
        
        Returns:
            dict: Comprehensive forgery analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize results
        results = {
            'is_forged': False,
            'confidence': 0.0,
            'forgery_type': 'None',
            'issues_found': [],
            'quality_metrics': {},
            'text_analysis': {},
            'visual_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. Quality Checks
            quality_metrics = self.check_document_quality(image)
            results['quality_metrics'] = quality_metrics
            
            # 2. Text Extraction and Analysis
            text_analysis = self.analyze_text_tampering(image)
            results['text_analysis'] = text_analysis
            
            # 3. Visual Tampering Detection
            visual_analysis = self.detect_visual_tampering(image)
            results['visual_analysis'] = visual_analysis
            
            # 4. Font Consistency Check
            font_analysis = self.check_font_consistency(image)
            
            # 5. Calculate Overall Forgery Score
            forgery_score = self.calculate_forgery_score(
                quality_metrics,
                text_analysis,
                visual_analysis,
                font_analysis
            )
            
            results['confidence'] = float(forgery_score)
            results['is_forged'] = bool(forgery_score > 0.5)
            
            # Determine forgery type
            if results['is_forged']:
                results['forgery_type'] = self.determine_forgery_type(
                    quality_metrics, text_analysis, visual_analysis
                )
        
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
        
        # Convert all NumPy types to native Python types
        results = self.convert_to_native_types(results)
        
        return results
    
    def check_document_quality(self, image):
        """
        Check document quality metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur Detection (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = bool(laplacian_var < 100)
        
        # 2. Brightness Check
        mean_brightness = np.mean(gray)
        is_too_dark = bool(mean_brightness < 50)
        is_too_bright = bool(mean_brightness > 200)
        
        # 3. Contrast Check
        contrast = gray.std()
        has_low_contrast = bool(contrast < 30)
        
        # 4. Resolution Check
        height, width = gray.shape
        is_low_resolution = bool((height * width) < (500 * 500))
        
        # 5. JPEG Quality Estimation
        jpeg_quality = self.estimate_jpeg_quality(image)
        has_low_jpeg_quality = bool(jpeg_quality < 75)
        
        # 6. Noise Level
        noise_level = self.estimate_noise_level(gray)
        has_high_noise = bool(noise_level > 15)
        
        return {
            'blur_score': float(laplacian_var),
            'is_blurry': is_blurry,
            'brightness': float(mean_brightness),
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'contrast': float(contrast),
            'has_low_contrast': has_low_contrast,
            'resolution': f"{width}x{height}",
            'is_low_resolution': is_low_resolution,
            'jpeg_quality': int(jpeg_quality),
            'has_low_jpeg_quality': has_low_jpeg_quality,
            'noise_level': float(noise_level),
            'has_high_noise': has_high_noise
        }
    
    def estimate_jpeg_quality(self, image):
        """
        Estimate JPEG compression quality
        """
        try:
            # Re-compress and compare
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            _, encimg = cv2.imencode('.jpg', image, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            
            # Calculate MSE
            mse = np.mean((image - decimg) ** 2)
            
            # Estimate quality based on MSE
            if mse < 100:
                return 95
            elif mse < 500:
                return 85
            elif mse < 1000:
                return 75
            else:
                return 60
        except:
            return 80
    
    def estimate_noise_level(self, gray_image):
        """
        Estimate noise level using median absolute deviation
        """
        try:
            # Apply median filter
            median = cv2.medianBlur(gray_image, 5)
            
            # Calculate noise
            noise = np.abs(gray_image.astype(float) - median.astype(float))
            noise_level = np.median(noise)
            
            return float(noise_level)
        except:
            return 0.0
    
    def analyze_text_tampering(self, image):
        """
        Analyze text for tampering using OCR
        """
        # Extract text with detailed information
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            return {
                'error': str(e),
                'text_extracted': False,
                'num_words': 0
            }
        
        # Filter valid text
        valid_text = []
        confidence_scores = []
        bbox_sizes = []
        
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = int(data['conf'][i]) if str(data['conf'][i]).isdigit() else 0
            
            if conf > 0 and text:
                valid_text.append(text)
                confidence_scores.append(conf)
                
                # Calculate bounding box size
                w = data['width'][i]
                h = data['height'][i]
                bbox_sizes.append(w * h)
        
        if not valid_text:
            return {
                'text_extracted': False,
                'num_words': 0,
                'has_low_confidence_text': False,
                'has_mixed_confidence': False,
                'has_inconsistent_sizing': False,
                'has_unusual_characters': False
            }
        
        # Analysis
        avg_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        min_confidence = float(np.min(confidence_scores)) if confidence_scores else 0.0
        max_confidence = float(np.max(confidence_scores)) if confidence_scores else 0.0
        
        # Check for suspicious patterns
        has_low_confidence_text = bool(avg_confidence < 70)
        has_mixed_confidence = bool((max_confidence - min_confidence) > 50)
        
        # Font size variation (using bbox sizes as proxy)
        bbox_std = float(np.std(bbox_sizes)) if len(bbox_sizes) > 1 else 0.0
        bbox_mean = float(np.mean(bbox_sizes)) if bbox_sizes else 1.0
        has_inconsistent_sizing = bool(bbox_std > bbox_mean * 0.5 if bbox_mean > 0 else False)
        
        # Special character detection (possible manual editing)
        full_text = ' '.join(valid_text)
        special_chars = re.findall(r'[^a-zA-Z0-9\s\.,]', full_text)
        has_unusual_characters = bool(len(special_chars) > len(full_text) * 0.1)
        
        return {
            'text_extracted': True,
            'num_words': int(len(valid_text)),
            'avg_confidence': avg_confidence,
            'min_confidence': min_confidence,
            'has_low_confidence_text': has_low_confidence_text,
            'has_mixed_confidence': has_mixed_confidence,
            'has_inconsistent_sizing': has_inconsistent_sizing,
            'has_unusual_characters': has_unusual_characters,
            'extracted_text': full_text[:200]  # First 200 chars
        }
    
    def detect_visual_tampering(self, image):
        """
        Detect visual tampering using computer vision techniques
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge Analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0) / edges.size)
        
        # 2. Copy-Move Detection (basic)
        copy_move_score = self.detect_copy_move(gray)
        
        # 3. Splicing Detection
        splicing_score = self.detect_splicing(image)
        
        # 4. Pixel Inconsistency
        pixel_inconsistency = self.detect_pixel_inconsistency(image)
        
        # 5. Compression Artifact Analysis
        compression_artifacts = self.detect_compression_artifacts(gray)
        
        return {
            'edge_density': edge_density,
            'has_unusual_edges': bool(edge_density > 0.3 or edge_density < 0.05),
            'copy_move_score': copy_move_score,
            'has_copy_move': bool(copy_move_score > 0.6),
            'splicing_score': splicing_score,
            'has_splicing': bool(splicing_score > 0.5),
            'pixel_inconsistency': pixel_inconsistency,
            'has_pixel_inconsistency': bool(pixel_inconsistency > 0.5),
            'compression_artifacts': compression_artifacts,
            'has_compression_artifacts': bool(compression_artifacts > 0.6)
        }
    
    def detect_copy_move(self, gray_image):
        """
        Detect copy-move forgery using block matching
        """
        try:
            # Divide image into blocks
            block_size = 16
            h, w = gray_image.shape
            
            blocks = []
            positions = []
            
            for i in range(0, h - block_size, block_size // 2):
                for j in range(0, w - block_size, block_size // 2):
                    block = gray_image[i:i+block_size, j:j+block_size]
                    if block.shape == (block_size, block_size):
                        blocks.append(block.flatten())
                        positions.append((i, j))
            
            if len(blocks) < 2:
                return 0.0
            
            # Find similar blocks (sample to avoid long computation)
            blocks = np.array(blocks)
            similar_count = 0
            max_samples = min(100, len(blocks))
            
            sample_indices = np.random.choice(len(blocks), max_samples, replace=False)
            
            for i in sample_indices:
                for j in range(i + 1, len(blocks)):
                    # Skip adjacent blocks
                    if abs(positions[i][0] - positions[j][0]) < block_size * 2 and \
                       abs(positions[i][1] - positions[j][1]) < block_size * 2:
                        continue
                    
                    # Calculate correlation with nan handling
                    corr_matrix = np.corrcoef(blocks[i], blocks[j])
                    if not np.isnan(corr_matrix[0, 1]):
                        correlation = corr_matrix[0, 1]
                        if correlation > 0.95:  # Very similar
                            similar_count += 1
            
            # Normalize score
            total_comparisons = max_samples * (len(blocks) - 1) / 2
            score = similar_count / total_comparisons if total_comparisons > 0 else 0.0
            
            return float(min(score * 10, 1.0))  # Scale up and cap at 1.0
        except Exception as e:
            print(f"Copy-move detection error: {e}")
            return 0.0
    
    def detect_splicing(self, image):
        """
        Detect image splicing using noise analysis
        """
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Divide into regions
            h, w = lab.shape[:2]
            region_h, region_w = h // 3, w // 3
            
            if region_h == 0 or region_w == 0:
                return 0.0
            
            noise_levels = []
            
            for i in range(3):
                for j in range(3):
                    region = lab[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                    if region.size > 0:
                        noise = self.estimate_noise_level(region[:,:,0])
                        noise_levels.append(noise)
            
            if len(noise_levels) < 2:
                return 0.0
            
            # Check variance in noise levels
            noise_variance = float(np.var(noise_levels))
            
            # High variance indicates splicing
            score = min(noise_variance / 50, 1.0)
            return float(score)
        except Exception as e:
            print(f"Splicing detection error: {e}")
            return 0.0
    
    def detect_pixel_inconsistency(self, image):
        """
        Detect pixel-level inconsistencies
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate local statistics
            h, w = hsv.shape[:2]
            block_size = 32
            
            if h < block_size or w < block_size:
                return 0.0
            
            variances = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = hsv[i:i+block_size, j:j+block_size, 2]  # Value channel
                    if block.size > 0:
                        variances.append(float(np.var(block)))
            
            # Check for unusual variance patterns
            if len(variances) > 1:
                variance_of_variances = float(np.var(variances))
                score = min(variance_of_variances / 1000, 1.0)
                return float(score)
            
            return 0.0
        except Exception as e:
            print(f"Pixel inconsistency error: {e}")
            return 0.0
    
    def detect_compression_artifacts(self, gray_image):
        """
        Detect JPEG compression artifacts
        """
        try:
            h, w = gray_image.shape
            
            if h < 16 or w < 16:
                return 0.0
            
            # Resize if too large
            if h > 512 or w > 512:
                scale = 512 / max(h, w)
                gray_image = cv2.resize(gray_image, None, fx=scale, fy=scale)
                h, w = gray_image.shape
            
            # Apply DCT
            dct = cv2.dct(np.float32(gray_image) / 255.0)
            
            # Check for 8x8 block artifacts
            block_diffs = []
            
            for i in range(8, min(h, 256), 8):
                for j in range(8, min(w, 256), 8):
                    if i < h and j < w:
                        diff = abs(dct[i, j] - dct[i-1, j-1])
                        if not np.isnan(diff) and not np.isinf(diff):
                            block_diffs.append(float(diff))
            
            if block_diffs:
                avg_diff = float(np.mean(block_diffs))
                score = min(avg_diff * 10, 1.0)
                return float(score)
            
            return 0.0
        except Exception as e:
            print(f"Compression artifacts error: {e}")
            return 0.0
    
    def check_font_consistency(self, image):
        """
        Check font consistency across document
        """
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            heights = []
            widths = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 60 and data['text'][i].strip():
                    heights.append(data['height'][i])
                    widths.append(data['width'][i])
            
            if len(heights) > 5:
                height_std = float(np.std(heights))
                height_mean = float(np.mean(heights))
                
                # Coefficient of variation
                cv = height_std / height_mean if height_mean > 0 else 0.0
                
                return {
                    'has_inconsistent_fonts': bool(cv > 0.3),
                    'font_variation_score': float(cv)
                }
            
            return {
                'has_inconsistent_fonts': False,
                'font_variation_score': 0.0
            }
        except Exception as e:
            print(f"Font consistency error: {e}")
            return {
                'has_inconsistent_fonts': False,
                'font_variation_score': 0.0
            }
    
    def calculate_forgery_score(self, quality, text, visual, font):
        """
        Calculate overall forgery probability
        """
        score = 0.0
        weight_sum = 0.0
        
        # Quality issues (weight: 0.2)
        quality_score = 0.0
        if quality.get('is_blurry'): quality_score += 0.3
        if quality.get('has_low_jpeg_quality'): quality_score += 0.4
        if quality.get('has_high_noise'): quality_score += 0.3
        score += quality_score * 0.2
        weight_sum += 0.2
        
        # Text issues (weight: 0.3)
        text_score = 0.0
        if text.get('has_low_confidence_text'): text_score += 0.4
        if text.get('has_mixed_confidence'): text_score += 0.3
        if text.get('has_inconsistent_sizing'): text_score += 0.3
        score += text_score * 0.3
        weight_sum += 0.3
        
        # Visual tampering (weight: 0.4)
        visual_score = 0.0
        if visual.get('has_copy_move'): visual_score += 0.4
        if visual.get('has_splicing'): visual_score += 0.3
        if visual.get('has_pixel_inconsistency'): visual_score += 0.3
        score += visual_score * 0.4
        weight_sum += 0.4
        
        # Font issues (weight: 0.1)
        font_score = font.get('font_variation_score', 0.0)
        score += font_score * 0.1
        weight_sum += 0.1
        
        final_score = float(min(score / weight_sum if weight_sum > 0 else 0, 1.0))
        return final_score
    
    def determine_forgery_type(self, quality, text, visual):
        """
        Determine the type of forgery detected
        """
        types = []
        
        if visual.get('has_copy_move'):
            types.append('Copy-Move')
        if visual.get('has_splicing'):
            types.append('Splicing')
        if text.get('has_mixed_confidence'):
            types.append('Text Tampering')
        if quality.get('is_blurry') and quality.get('has_low_jpeg_quality'):
            types.append('Scanned/Photocopied')
        
        return ', '.join(types) if types else 'General Tampering'
