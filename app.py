from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from tensorflow.keras.applications.vgg16 import preprocess_input
from asgiref.wsgi import WsgiToAsgi  # ASGI adapter for Flask

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
api = Api(app)

try:
    model = load_model('models/soil_health_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class Home(Resource):
    def get(self):
        return jsonify({"message": "Welcome to Home"})

class SoilImageClassifier(Resource):
    def post(self):
        try:
            image_file = request.files['image']
            image = Image.open(image_file).convert('RGB')
            image = image.resize((150, 150)) 
            image_array = np.array(image) / 255.0  
            image_array = preprocess_input(image_array)  
            image_array = np.expand_dims(image_array, axis=0)  

            predictions = model.predict(image_array)

            healthy_percentage = float(predictions[0][0] * 100)
            unhealthy_percentage = float((1 - predictions[0][0]) * 100)

            if predictions[0][0] > 0.5:
                result = "Healthy Soil"
                percentage = {'healthy_percentage': healthy_percentage, 'unhealthy_percentage': unhealthy_percentage}
            else:
                result = "Unhealthy Soil"
                percentage = {'healthy_percentage': healthy_percentage, 'unhealthy_percentage': unhealthy_percentage}

            return {'result': result, 'percentage': percentage}, 200

        except Exception as e:
            return {'error': str(e)}, 500

api.add_resource(Home, '/')
api.add_resource(SoilImageClassifier, '/classify')


asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:asgi_app", host="127.0.0.1", port=8000, reload=True)
