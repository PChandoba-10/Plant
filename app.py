from flask import Flask, request, jsonify, render_template
import os, cv2, torch, shutil, tempfile
from PIL import Image
from collections import Counter
from transformers import ViTImageProcessor, ViTForImageClassification
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

class SimpleSegmentationModel():
    def __init__(self, min_area=500):
        self.min_area = min_area
        # Create a dedicated temporary directory for segmentation outputs for this instance
        self.temp_dir = tempfile.mkdtemp(prefix='plant_seg_')
    
    def predict(self, image_path):
        output = []
        files = []
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image file")
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                    
                x, y, w, h = cv2.boundingRect(cnt)
                if w == 0 or h == 0:
                    continue
                    
                crop = image[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                    
                temp_path = os.path.join(self.temp_dir, f"segment_{i}.jpg")
                saved = cv2.imwrite(temp_path, crop)
                # If cv2.imwrite fails or the file doesn't exist, try using PIL as a fallback.
                if not saved or not os.path.exists(temp_path):
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        pil_img.save(temp_path, "JPEG")
                    except Exception as e:
                        print(f"Segment save warning: Failed to save segment {i} using both methods: {str(e)}")
                        continue

                if os.path.exists(temp_path):
                    output.append([x, y, x+w, y+h])
                    files.append(temp_path)
                else:
                    print("Segment save warning: Failed to save segment")
                    
            return output, files
            
        except Exception as e:
            self.cleanup()
            raise

    def cleanup(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Cleanup warning: {str(e)}")

class ClassificationModel():
    def __init__(self, model_path='ar5entum/vit-base-patch16-224-leaf-classification', device=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if device:
            self.device = device
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(self.device)
        self.labels = self.model.config.id2label

    def predict(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            # Use context manager to ensure the file is closed after reading.
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.logits.argmax(-1).item()
                
        except Exception as e:
            print(f"Classification error: {str(e)}")
            raise

# Create a single global instance for classification (its weights are loaded once)
cls_model = ClassificationModel()

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/homepage.html', methods=['GET'])
def homes():
    return render_template("homepage.html")

@app.route('/camera.html', methods=['GET'])
def homess():
    return render_template("camera.html")

@app.route('/jamun.html', methods=['GET'])
def jamun():
    return render_template("jamun.html")

@app.route('/aloevera.html', methods=['GET'])
def aloevera_page():
    return render_template("aloevera.html")

@app.route('/amla.html', methods=['GET'])
def amla():
    return render_template("amla.html")

@app.route('/banyan.html', methods=['GET'])
def banyan():
    return render_template("banyan.html")

@app.route('/belpatra.html', methods=['GET'])
def belpatra():
    return render_template("belpatra.html")

@app.route('/coconut.html', methods=['GET'])
def coconut():
    return render_template("coconut.html")

@app.route('/help.html', methods=['GET'])
def help_page():
    return render_template("help.html")

@app.route('/mango.html', methods=['GET'])
def mango():
    return render_template("mango.html")

@app.route('/neem.html', methods=['GET'])
def neem():
    return render_template("neem.html")

@app.route('/rose.html', methods=['GET'])
def rose():
    return render_template("rose.html")

@app.route('/setting.html', methods=['GET'])
def setting():
    return render_template("setting.html")

@app.route('/shatavari.html', methods=['GET'])
def shatavari():
    return render_template("shatavari.html")

@app.route('/snakeplant.html', methods=['GET'])
def snakeplant():
    return render_template("snakeplant.html")

@app.route('/tulsi.html', methods=['GET'])
def tulsi():
    return render_template("tulsi.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    temp_files = []
    seg_model = SimpleSegmentationModel()  # Fresh instance per request
    
    try:
        file = request.files['image']
        fd, temp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)
        temp_files.append(temp_path)
        file.save(temp_path)

        seg_boxes, seg_paths = seg_model.predict(temp_path)
        temp_files.extend(seg_paths)
        
        if not seg_paths:
            return jsonify({
                'error': 'No plant segments detected',
                'suggestion': 'Try a clearer image with better contrast'
            }), 400

        predictions = []
        for path in seg_paths:
            try:
                predictions.append(cls_model.predict(path))
            except Exception as e:
                print(f"Skipping invalid segment: {str(e)}")
                continue

        if not predictions:
            return jsonify({'error': 'No valid predictions from segments'}), 400

        counts = Counter(predictions)
        confidence_scores = {cls: count / len(predictions) for cls, count in counts.items()}
        max_pred = max(confidence_scores, key=confidence_scores.get)
        plant_type = cls_model.labels.get(max_pred, "Unknown")

        # DEBUG: Print out what the model predicted
        print("DEBUG: Raw predictions (class IDs) =", predictions)
        print("DEBUG: class->label map =", cls_model.labels)
        print("DEBUG: Final chosen class ID =", max_pred)
        print("DEBUG: plant_type (label) =", plant_type)

        # 🔹 Extra Details for Aloe Vera
        benefits = None

        # Normalize plant_type by removing spaces before comparing
        normalized_label = plant_type.lower().replace(" ", "")
        print("DEBUG: normalized_label =", normalized_label)  # debug

        if normalized_label == "aloevera":
            benefits = [
                "Helps heal burns and wounds",
                "Aids digestion and gut health",
                "Boosts skin hydration and reduces acne",
                "Supports immune function",
                "Has anti-inflammatory properties"
            ]
        
        response = {
            'plant_type': plant_type,
            'confidence': round(confidence_scores[max_pred] * 100, 1),
            'confidence_scores': confidence_scores,
            'segments_found': len(seg_boxes),
            'id2class': cls_model.labels,
            'benefits': benefits
        }

        print("DEBUG: Response JSON =", response)  # debug
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e),
            'suggestion': 'Try a different image or check server logs'
        }), 500
         
    finally:
        for path in temp_files:
            try:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            except Exception as e:
                print(f"Cleanup warning for {path}: {str(e)}")
        seg_model.cleanup()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
