import numpy as np
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from PIL import Image
import pickle

class ImageClassifier:
    def __init__(self, model_path="model_classification.h5", dataset_path="dataset.pkl"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.class_names = None
        self.image_size = (128, 128)
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
    
    def load_class_names(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        with open(self.dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.class_names = dataset['class_names']
    
    def preprocess_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize(self.image_size, Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                return img_array
        except Exception as e:
            raise ValueError(f"Error during image preprocessing: {e}")
    
    def predict(self, image_path, show_probabilities=True, show_image=True):
        img_array = self.preprocess_image(image_path)
        predictions = self.model.predict(img_array, verbose=0)
        probabilities = predictions[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

        results = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                self.class_names[i]: float(probabilities[i]) 
                for i in range(len(self.class_names))
            }
        }
        return results