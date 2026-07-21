import cv2
import numpy as np
from PIL import Image
import io

class ELAProcessor:
    """
    Error Level Analysis (ELA) for detecting image forgery.
    ELA highlights areas with different compression levels.
    """
    
    def __init__(self, quality=90):
        self.quality = quality
    
    def convert_to_ela_image(self, image_path, output_size=(256, 256)):
        """
        Convert image to ELA representation
        
        Args:
            image_path: Path to input image
            output_size: Output dimensions (height, width)
        
        Returns:
            ELA image as numpy array
        """
        try:
            # Load original image
            original = Image.open(image_path).convert('RGB')
            original = original.resize(output_size, Image.LANCZOS)
            
            # Save image with specified quality
            temp_buffer = io.BytesIO()
            original.save(temp_buffer, 'JPEG', quality=self.quality)
            temp_buffer.seek(0)
            
            # Reload compressed image
            compressed = Image.open(temp_buffer)
            
            # Calculate difference (ELA)
            ela_image = np.array(original) - np.array(compressed)
            
            # Normalize to 0-255 range
            ela_image = np.abs(ela_image)
            ela_image = (ela_image / ela_image.max() * 255).astype(np.uint8)
            
            return ela_image
            
        except Exception as e:
            raise Exception(f"Error in ELA conversion: {str(e)}")
    
    def apply_enhanced_ela(self, image_path, output_size=(256, 256)):
        """
        Apply enhanced ELA with additional forensic features
        """
        # Get basic ELA
        ela_image = self.convert_to_ela_image(image_path, output_size)
        
        # Convert to grayscale for additional analysis
        ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection to highlight boundaries
        edges = cv2.Canny(ela_gray, 50, 150)
        
        # Combine ELA with edge information
        enhanced_ela = cv2.merge([ela_image[:,:,0], ela_image[:,:,1], edges])
        
        return enhanced_ela
    
    def generate_forensic_maps(self, image_path, output_size=(256, 256)):
        """
        Generate multiple forensic maps for comprehensive analysis
        """
        # Original image
        original = cv2.imread(image_path)
        original = cv2.resize(original, output_size)
        
        # ELA map
        ela = self.convert_to_ela_image(image_path, output_size)
        
        # Noise analysis
        noise = self.extract_noise_pattern(original)
        
        # JPEG ghost analysis
        jpeg_ghost = self.jpeg_ghost_detection(image_path, output_size)
        
        return {
            'original': original,
            'ela': ela,
            'noise': noise,
            'jpeg_ghost': jpeg_ghost
        }
    
    def extract_noise_pattern(self, image):
        """
        Extract noise patterns using median filtering
        """
        denoised = cv2.medianBlur(image, 5)
        noise = cv2.absdiff(image, denoised)
        return noise
    
    def jpeg_ghost_detection(self, image_path, output_size):
        """
        JPEG ghost detection for multiple quality levels
        """
        original = Image.open(image_path).convert('RGB')
        original = original.resize(output_size, Image.LANCZOS)
        
        ghosts = []
        qualities = [70, 80, 90, 95]
        
        for quality in qualities:
            temp_buffer = io.BytesIO()
            original.save(temp_buffer, 'JPEG', quality=quality)
            temp_buffer.seek(0)
            compressed = Image.open(temp_buffer)
            
            ghost = np.abs(np.array(original) - np.array(compressed))
            ghosts.append(ghost)
        
        # Average ghost map
        avg_ghost = np.mean(ghosts, axis=0).astype(np.uint8)
        return avg_ghost
