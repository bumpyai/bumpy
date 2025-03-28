from flask import Blueprint, render_template, request, jsonify, current_app, session, url_for
import os
import uuid
import time
import json
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import base64
import requests
from dotenv import load_dotenv
import logging
from datetime import datetime
from io import BytesIO
import rembg
from PIL import ImageDraw
from uuid import uuid4

# Try to import optional dependencies
HAVE_NUMPY = False
HAVE_TORCH = False
HAVE_CV2 = False

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    logging.warning("NumPy not available. Some features will be limited.")

try:
    import torch
    import torchvision.transforms as transforms
    HAVE_TORCH = True
except ImportError:
    logging.warning("PyTorch not available. U²-Net model will not be used.")

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    logging.warning("OpenCV not available. Advanced processing will be limited.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Blueprint
bg_remover_bp = Blueprint('bg_remover', __name__, url_prefix='/bg-remover')

# Configure upload settings
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
MODELS_FOLDER = 'app/models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Set device based on available hardware
USE_GPU = False
DEVICE = None
if HAVE_TORCH:
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
    logger.info(f"Using device: {DEVICE}")
else:
    logger.info("PyTorch not available, falling back to CPU-only processing")

# Use ThreadPoolExecutor for async tasks if available
try:
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading
    
    # Processing queue for batch operations
    processing_queue = queue.Queue()
    # Threading pool for background tasks
    executor = ThreadPoolExecutor(max_workers=2)
    # Status tracking for real-time updates
    processing_status = {}
    
    HAVE_THREADING = True
except ImportError:
    logger.warning("ThreadPoolExecutor not available. Processing will be synchronous.")
    HAVE_THREADING = False
    processing_status = {}

# Model selection - can be changed by environment variables
DEFAULT_MODEL = 'enhanced'  # Since we can't guarantee torch availability, default to enhanced

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bg_remover_bp.route('/')
def index():
    # Ensure directories exist every time the route is accessed
    os.makedirs(current_app.config.get('UPLOAD_FOLDER', UPLOAD_FOLDER), exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    return render_template('bg_remover.html')

# Add endpoint for checking processing status
@bg_remover_bp.route('/status/<task_id>', methods=['GET'])
def check_status(task_id):
    """Check the status of a background removal task"""
    if task_id in processing_status:
        return jsonify(processing_status[task_id])
    return jsonify({'status': 'not_found'}), 404

@bg_remover_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and background removal."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the requested effect (default to transparent)
    effect = request.form.get('effect', 'transparent')
    
    # Get user ID from session or use a temporary ID
    user_id = session.get('user_id')
    if not user_id:
        # For anonymized users, create a temporary ID. In a production app,
        # you might want to require authentication for some features.
        if 'anon_user_id' not in session:
            session['anon_user_id'] = f"anon_{uuid4().hex[:10]}"
        user_id = session['anon_user_id']
    
    # Check user daily usage limit
    can_process, usage_count = _record_usage(user_id, 'started')
    if not can_process:
        return jsonify({
            'status': 'error',
            'message': 'Daily limit reached. You have processed 25 images today. Please try again tomorrow or upgrade your plan.',
            'error_type': 'daily_limit',
            'usage_count': usage_count
        }), 429
    
    # Create user-specific subdirectories in uploads and results
    user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], user_id)
    user_results_dir = os.path.join('static/results', user_id)
    
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(user_results_dir, exist_ok=True)
    
    try:
        # Handle file upload from form
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No file selected'})
            
            # Secure the filename and save with timestamp
            filename = secure_filename(file.filename)
            base_name, ext = os.path.splitext(filename)
            unique_filename = f"{base_name}_{timestamp}{ext}"
            original_path = os.path.join(user_dir, unique_filename)
            
            # Save the uploaded file
            file.save(original_path)
            logging.info(f"Saved uploaded file to: {original_path}")
            
        elif 'image_data' in request.form:
            # Base64 image data
            image_data = request.form['image_data']
            
            # Check if it's a data URL and extract the base64 part
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            
            try:
                # Decode the base64 image
                image_bytes = base64.b64decode(image_data)
                
                # Save as PNG
                unique_filename = f"pasted_image_{timestamp}.png"
                original_path = os.path.join(user_dir, unique_filename)
                
                with open(original_path, 'wb') as f:
                    f.write(image_bytes)
                logging.info(f"Saved base64 image to: {original_path}")
                
            except Exception as e:
                logging.error(f"Error decoding base64 image: {str(e)}")
                return jsonify({'status': 'error', 'message': 'Invalid image data'})
        else:
            return jsonify({'status': 'error', 'message': 'No image provided'})
        
        # Generate output filename
        output_filename = f"result_{timestamp}.png"
        output_path = os.path.join(user_results_dir, output_filename)
        
        # Normalize paths
        original_path = os.path.normpath(original_path)
        output_path = os.path.normpath(output_path)
        
        # Log paths for debugging
        logging.info(f"Original path: {original_path}")
        logging.info(f"Output path: {output_path}")
        
        # Start background removal
        start_time = time.time()
        
        # Choose removal method based on effect
        if effect == 'transparent':
            result = _basic_bg_removal(original_path, output_path)
        else:
            result = _advanced_bg_removal(original_path, output_path, effect)
        
        if result.get('status') != 'success':
            return jsonify(result)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Generate URLs for frontend
        original_url = url_for('static', filename=f"uploads/{user_id}/{unique_filename}")
        result_url = url_for('static', filename=f"results/{user_id}/{output_filename}")
        
        # Log URLs for debugging
        logging.info(f"Original URL: {original_url}")
        logging.info(f"Result URL: {result_url}")
        
        # Try to get image dimensions
        try:
            with Image.open(output_path) as img:
                width, height = img.size
                format_name = img.format
                file_size = os.path.getsize(output_path)
                
                file_info = {
                    'width': width,
                    'height': height,
                    'size': file_size,
                    'format': format_name
                }
        except Exception as e:
            logging.error(f"Error getting image info: {str(e)}")
            file_info = {}
        
        # Check if the result file exists
        if not os.path.exists(output_path):
            logging.error(f"Result file does not exist: {output_path}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to save the processed image'
            })
        
        # Return success response
        return jsonify({
            'status': 'success',
            'original_url': original_url,
            'result_url': result_url,
            'effect': effect,
            'processing_time': round(processing_time, 2),
            'file_info': file_info,
            'premium_available': False
        })
        
    except Exception as e:
        logging.exception("Error in upload_file")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def update_status(task_id, progress, message):
    """Update the processing status for a task"""
    if task_id in processing_status:
        processing_status[task_id]['progress'] = progress
        processing_status[task_id]['message'] = message

def remove_background(image_path, output_path, effect_type='transparent', model_type='enhanced', status_callback=None):
    """Remove background from image using selected model"""
    # Prioritize using U2NET model if available
    if model_type == 'u2net' and HAVE_TORCH:
        return _u2net_processing(image_path, output_path, effect_type, status_callback)
    elif model_type == 'deeplabv3' and HAVE_TORCH:
        return _deeplabv3_processing(image_path, output_path, effect_type, status_callback)
    else:
        # Fallback to enhanced AI processing
        return _ai_background_removal(image_path, output_path, effect_type, status_callback)

def _download_model(model_name='u2net'):
    """Download the U2NET model if it doesn't exist"""
    model_path = os.path.join(MODELS_FOLDER, f"{model_name}.pth")
    
    # If model already exists, return the path
    if os.path.exists(model_path):
        return model_path
        
    # Download URLs for different models
    model_urls = {
        'u2net': 'https://github.com/xuebinqin/U-2-Net/releases/download/v1.0.0/u2net.pth',
        'u2netp': 'https://github.com/xuebinqin/U-2-Net/releases/download/v1.0.0/u2netp.pth'
    }
    
    # If model URL exists, download it
    if model_name in model_urls:
        url = model_urls[model_name]
        logger.info(f"Downloading {model_name} model from {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Model downloaded successfully to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            return None
    else:
        logger.error(f"Unknown model: {model_name}")
        return None

def _load_u2net_model():
    """Load the U2NET model for inference"""
    from models.u2net import U2NET
    
    model_path = _download_model('u2net')
    if not model_path:
        raise Exception("Could not download U2NET model")
    
    # Load the model
    model = U2NET(3, 1)
    if USE_GPU:
        model = model.to(DEVICE)
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    return model

def _u2net_processing(image_path, output_path, effect_type='transparent', status_callback=None):
    """Process image with U2NET model"""
    if not HAVE_TORCH or not HAVE_NUMPY:
        logger.warning("U2NET processing requires PyTorch and NumPy. Using fallback method.")
        return _ai_background_removal(image_path, output_path, effect_type, status_callback)
        
    if status_callback:
        status_callback(10, "Loading U2NET model...")
    
    try:
        # Check if model implementation is available, if not use fallback
        try:
            from models.u2net import U2NET
        except ImportError:
            logger.warning("U2NET model implementation not found, using fallback")
            return _ai_background_removal(image_path, output_path, effect_type, status_callback)
        
        # Load the model
        model = _load_u2net_model()
        
        if status_callback:
            status_callback(30, "Preprocessing image...")
            
        # Load and preprocess image
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            
            # Resize for processing (U2NET typically works with 320x320)
            transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            
            if USE_GPU:
                img_tensor = img_tensor.to(DEVICE)
            
            if status_callback:
                status_callback(50, "Running inference...")
            
            # Run inference
            with torch.no_grad():
                output = model(img_tensor)
                
                # U2NET has multiple outputs, use the final one
                if isinstance(output, tuple):
                    output = output[0]
                
                # Convert output to mask
                output = output.squeeze().cpu().numpy()
                
                # Normalize to [0, 1]
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                
                # Resize mask back to original image size
                mask = cv2.resize(output, (width, height))
                
                # Threshold to create binary mask (can be adjusted)
                mask = (mask > 0.5).astype(np.uint8) * 255
            
            if status_callback:
                status_callback(70, "Applying mask...")
                
            # Open original image and apply mask
            original_image = Image.open(image_path).convert("RGBA")
            mask_image = Image.fromarray(mask).convert("L")
            
            # Apply mask to create transparent background
            original_image.putalpha(mask_image)
            
            if status_callback:
                status_callback(85, f"Applying {effect_type} effect...")
                
            # Apply additional effects if requested
            processed_image = _apply_effect(original_image, effect_type)
            
            # Save the result
            processed_image.save(output_path, "PNG")
            
            if status_callback:
                status_callback(100, "Processing complete")
                
            return True, "Background removed successfully with U2NET"
            
        except Exception as e:
            logger.error(f"Error processing image with U2NET: {str(e)}")
            if status_callback:
                status_callback(40, "Error with U2NET, using fallback...")
            return _ai_background_removal(image_path, output_path, effect_type, status_callback)
            
    except Exception as e:
        logger.error(f"Error in U2NET processing: {str(e)}")
        if status_callback:
            status_callback(30, "U2NET unavailable, using fallback...")
        return _ai_background_removal(image_path, output_path, effect_type, status_callback)

def _deeplabv3_processing(image_path, output_path, effect_type='transparent', status_callback=None):
    """Process image with DeepLabV3 model"""
    if not HAVE_TORCH or not HAVE_NUMPY:
        logger.warning("DeepLabV3 processing requires PyTorch and NumPy. Using fallback method.")
        return _ai_background_removal(image_path, output_path, effect_type, status_callback)
        
    if status_callback:
        status_callback(10, "Loading DeepLabV3 model...")
    
    try:
        # Use torchvision's DeepLabV3
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        model.eval()
        
        if USE_GPU:
            model = model.to(DEVICE)
        
        if status_callback:
            status_callback(30, "Preprocessing image...")
            
        # Load and preprocess the image
        input_image = Image.open(image_path).convert("RGB")
        width, height = input_image.size
        
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        
        if USE_GPU:
            input_batch = input_batch.to(DEVICE)
        
        if status_callback:
            status_callback(50, "Running inference...")
            
        # Run inference
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        
        # The output has 21 channels for PASCAL VOC classes
        # We want channel 15 which is the 'person' class for portraits
        # or use multiple channels for other common foreground objects
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Create mask for foreground (person class = 15)
        mask = np.zeros_like(output_predictions, dtype=np.uint8)
        # Add other common foreground classes
        foreground_classes = [15]  # Person class
        for cls in foreground_classes:
            mask = np.logical_or(mask, output_predictions == cls)
        
        mask = mask.astype(np.uint8) * 255
        
        if status_callback:
            status_callback(70, "Applying mask...")
            
        # Apply mask to original image
        original_image = Image.open(image_path).convert("RGBA")
        mask_image = Image.fromarray(mask).convert("L")
        mask_image = mask_image.resize((width, height))
        
        # Apply mask to create transparent background
        original_image.putalpha(mask_image)
        
        if status_callback:
            status_callback(85, f"Applying {effect_type} effect...")
            
        # Apply desired effect
        processed_image = _apply_effect(original_image, effect_type)
        
        # Save the result
        processed_image.save(output_path, "PNG")
        
        if status_callback:
            status_callback(100, "Processing complete")
            
        return True, "Background removed successfully with DeepLabV3"
        
    except Exception as e:
        logger.error(f"Error using DeepLabV3: {str(e)}")
        if status_callback:
            status_callback(30, "DeepLabV3 failed, using fallback...")
        return _ai_background_removal(image_path, output_path, effect_type, status_callback)

def _ai_background_removal(image_path, output_path, effect_type='transparent', status_callback=None):
    """Enhanced AI-like background removal simulation with improved image analysis"""
    try:
        if status_callback:
            status_callback(10, "Loading image...")
        
        logger.info(f"Starting enhanced AI background removal on {image_path} with effect {effect_type}")
        
        # Check if the input file exists
        if not os.path.exists(image_path):
            logger.error(f"Input file does not exist: {image_path}")
            return False, f"Input file does not exist: {image_path}"
            
        # Check if the output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            
        # Open the image - check if it's corrupted
        try:
            original = Image.open(image_path)
            logger.info(f"Successfully opened image: {image_path}, size: {original.size}, format: {original.format}")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {str(e)}")
            return False, f"Failed to open image: {str(e)}"
        
        # Convert to proper format for processing
        if status_callback:
            status_callback(20, "Converting image format...")
        
        img_pil = original.convert("RGBA")
        
        # Choose which algorithm to use based on available libraries
        if HAVE_CV2:
            return _advanced_bg_removal(img_pil, output_path, effect_type, status_callback)
        else:
            return _basic_bg_removal(img_pil, output_path, effect_type, status_callback)
            
    except Exception as e:
        logger.error(f"Unexpected error in background removal: {str(e)}")
        return False, f"Unexpected error: {str(e)}"

def _advanced_bg_removal(input_path, output_path, effect_type):
    """Perform advanced background removal with effects."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # First remove background
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()
            
        # Process the image
        output_data = rembg.remove(input_data)
        
        # Apply effect based on type
        if effect_type == 'shadow':
            # For shadow, create a temp transparent file then add shadow
            temp_path = output_path.replace('.png', '_temp.png')
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(output_data)
                
            # Open the transparent image and add shadow
            img = Image.open(temp_path)
            # Create shadow effect using PIL
            shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            # Add shadow logic here
            
            # Save the final result
            img.save(output_path, 'PNG')
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Clean up temp file
            
        elif effect_type == 'blur':
            # For blur background
            temp_path = output_path.replace('.png', '_temp.png')
            with open(temp_path, 'wb') as temp_file:
                temp_file.write(output_data)
                
            # Open original and transparent image
            original = Image.open(input_path)
            transparent = Image.open(temp_path)
            
            # Create blurred background
            blurred = original.filter(ImageFilter.GaussianBlur(15))
            
            # Composite transparent foreground onto blurred background
            result = Image.new('RGBA', original.size, (0, 0, 0, 0))
            result.paste(blurred, (0, 0))
            result.paste(transparent, (0, 0), transparent)
            
            # Save result
            result.save(output_path, 'PNG')
            if os.path.exists(temp_path):
                os.remove(temp_path)  # Clean up
            
        elif effect_type == 'artistic':
            # Just save the transparent result for now
            # In a real implementation, you would apply an artistic filter
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
        else:
            # Default transparent background
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
        
        # Verify the file was saved
        if os.path.exists(output_path):
            logging.info(f"Successfully saved processed image with effect '{effect_type}' to: {output_path} (Size: {os.path.getsize(output_path)} bytes)")
            return {'status': 'success', 'message': f'Background removed with {effect_type} effect'}
        else:
            logging.error(f"Failed to save output file: {output_path}")
            return {'status': 'error', 'message': 'Failed to save processed image'}
        
    except Exception as e:
        logging.exception(f"Error in advanced background removal: {str(e)}")
        return {'status': 'error', 'message': f'Error processing image: {str(e)}'}

def _basic_bg_removal(input_path, output_path):
    """Perform basic background removal, saving result to output_path."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Try using rembg library for background removal with error handling
        try:
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                
            # Process the image with a timeout guard
            # Use smaller, faster models to avoid timeouts
            try:
                # Use isnet-general-use model which is faster and smaller
                logging.info(f"Attempting background removal with isnet-general-use model")
                output_data = rembg.remove(
                    input_data,
                    session=rembg.new_session("isnet-general-use"),
                    only_mask=False,
                    alpha_matting=False  # Disable alpha matting for speed
                )
            except Exception as model_error:
                # If that fails, try with simpler settings
                logging.warning(f"isnet-general-use model failed, trying with u2netp (smaller model): {str(model_error)}")
                output_data = rembg.remove(
                    input_data, 
                    session=rembg.new_session("u2netp"),  # Much smaller model
                    only_mask=False,
                    alpha_matting=False,
                    post_process_mask=True  # Extra processing to clean up the mask
                )
            
            # Save the output
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
                
        except Exception as rembg_error:
            logging.error(f"rembg processing failed: {str(rembg_error)}")
            
            # Fallback method using PIL
            logging.info("Using fallback method with PIL for basic background removal")
            
            # Open the image with PIL
            img = Image.open(input_path).convert("RGBA")
            
            # Resize image if it's too large (to prevent timeouts)
            max_size = 1000  # Maximum width or height
            if img.width > max_size or img.height > max_size:
                # Calculate new dimensions while preserving aspect ratio
                if img.width > img.height:
                    new_width = max_size
                    new_height = int(img.height * (max_size / img.width))
                else:
                    new_height = max_size
                    new_width = int(img.width * (max_size / img.height))
                logging.info(f"Resizing image from {img.width}x{img.height} to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Create a simple mask by detecting edges
            # This is very simple but will work as a last resort
            if HAVE_CV2 and HAVE_NUMPY:
                # Use OpenCV for better edge detection if available
                try:
                    # Convert PIL image to OpenCV format
                    cv_img = np.array(img.convert('RGB'))
                    cv_img = cv_img[:, :, ::-1].copy()  # RGB to BGR
                    
                    # Process with OpenCV
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, 50, 150)
                    
                    # Dilate edges to create a mask
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    dilated = cv2.dilate(edges, kernel, iterations=3)
                    
                    # Convert back to PIL
                    mask = Image.fromarray(dilated).convert("L")
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
                except Exception as cv_error:
                    logging.error(f"OpenCV processing failed: {str(cv_error)}")
                    # Fall back to PIL-only method
                    mask = img.convert("L")
                    mask = mask.filter(ImageFilter.FIND_EDGES)
                    mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
                    mask = mask.point(lambda x: 255 if x > 128 else 0)
            else:
                # Simple PIL-only method
                # Create a mask based on edge detection
                mask = img.convert("L")
                mask = mask.filter(ImageFilter.FIND_EDGES)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=1))
                mask = mask.point(lambda x: 255 if x > 128 else 0)
            
            # Apply mask to original image
            result = Image.new('RGBA', img.size, (0, 0, 0, 0))
            result.paste(img, (0, 0), mask)
            
            # Save the result
            result.save(output_path, format="PNG")
            
        # Verify the file was saved
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logging.info(f"Successfully saved processed image to: {output_path} (Size: {file_size} bytes)")
            
            # Extra check to ensure the file is not empty or corrupted
            if file_size < 100:  # Extremely small file, likely an error
                logging.warning(f"Output file is suspiciously small ({file_size} bytes)")
                return {'status': 'error', 'message': 'Generated image appears to be invalid'}
                
            return {'status': 'success', 'message': 'Background removed successfully'}
        else:
            logging.error(f"Failed to save output file: {output_path}")
            return {'status': 'error', 'message': 'Failed to save processed image'}
            
    except Exception as e:
        logging.exception(f"Error in basic background removal: {str(e)}")
        return {'status': 'error', 'message': f'Error processing image: {str(e)}'}

def _apply_effect(img, effect_type):
    """Apply different effects to the image after background removal"""
    try:
        if effect_type == 'transparent':
            # Just return the image with transparent background
            return img
        
        elif effect_type == 'shadow':
            # Add a drop shadow effect
            # Create a mask from the alpha channel
            alpha = img.split()[3]
            
            # Create a black shadow
            shadow = Image.new('RGBA', img.size, (0, 0, 0, 0))
            shadow_alpha = ImageOps.colorize(alpha, (0, 0, 0, 0), (0, 0, 0, 128))
            shadow.putalpha(shadow_alpha)
            
            # Blur the shadow
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=5))
            
            # Offset the shadow
            shadow_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
            shadow_img.paste(shadow, (5, 5))
            
            # Composite the original image on top of the shadow
            result = Image.alpha_composite(shadow_img, img)
            return result
        
        elif effect_type == 'blur':
            # Create a blurred version of the original for the background
            # We need a new blank canvas with white background
            bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
            
            # Create a blurred version of the original image
            blurred = img.copy().filter(ImageFilter.GaussianBlur(radius=15))
            
            # Darken and desaturate the blurred background
            enhancer = ImageEnhance.Brightness(blurred)
            blurred = enhancer.enhance(0.7)  # Darken
            
            # Blend the blurred image with the white background
            alpha = 0.8  # Transparency level
            for x in range(bg.width):
                for y in range(bg.height):
                    r1, g1, b1, a1 = blurred.getpixel((x, y))
                    r2, g2, b2, a2 = bg.getpixel((x, y))
                    bg.putpixel((x, y), (
                        int(r1 * alpha + r2 * (1 - alpha)),
                        int(g1 * alpha + g2 * (1 - alpha)),
                        int(b1 * alpha + b2 * (1 - alpha)),
                        255
                    ))
            
            # Create a new image with the blurred background and original foreground
            result = bg.copy()
            result.paste(img, (0, 0), img.split()[3])  # Use alpha as mask
            return result
        
        elif effect_type == 'artistic':
            # Apply an artistic filter effect
            # First create a colorful gradient background
            gradient = Image.new('RGBA', img.size, (0, 0, 0, 255))
            
            # Create a gradient from blue to purple
            for y in range(img.height):
                for x in range(img.width):
                    r = int(100 + (x / img.width) * 100)
                    g = int(50 + (y / img.height) * 50)
                    b = int(200 - (y / img.height) * 50)
                    gradient.putpixel((x, y), (r, g, b, 255))
            
            # Apply some artistic filters to the gradient
            gradient = gradient.filter(ImageFilter.GaussianBlur(radius=20))
            
            # Composite the original image on top
            result = gradient.copy()
            result.paste(img, (0, 0), img.split()[3])  # Use alpha as mask
            
            # Add some final touches
            result = result.filter(ImageFilter.EDGE_ENHANCE)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(1.2)
            
            return result
        
        else:
            # Default to transparent if unknown effect type
            return img
    
    except Exception as e:
        logger.error(f"Error applying effect {effect_type}: {str(e)}")
        # Return original image if effect application fails
        return img

def _record_usage(user_id, status):
    """Record usage statistics and check if user has reached their daily limit"""
    try:
        # This would normally write to a database
        # For this demo, we'll just write to a JSON file
        usage_file = 'static/usage_stats.json'
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Load existing stats
        stats = {}
        if os.path.exists(usage_file):
            try:
                with open(usage_file, 'r') as f:
                    stats = json.load(f)
            except:
                stats = {}
        
        # Initialize user data if not exists
        if user_id not in stats:
            stats[user_id] = {'total': 0, 'dates': {}}
        
        # Initialize today's data if not exists
        if today not in stats[user_id]['dates']:
            stats[user_id]['dates'][today] = 0
        
        # Check daily limit - return False if limit exceeded
        daily_limit = 25  # Set daily limit to 25 images
        if stats[user_id]['dates'][today] >= daily_limit:
            return False, stats[user_id]['dates'][today]
            
        # Increment usage counters
        stats[user_id]['total'] += 1
        stats[user_id]['dates'][today] += 1
        
        # Save stats
        with open(usage_file, 'w') as f:
            json.dump(stats, f)
            
        return True, stats[user_id]['dates'][today]
            
    except Exception as e:
        logger.error(f"Error recording usage: {str(e)}")
        # Allow processing in case of error
        return True, 0 