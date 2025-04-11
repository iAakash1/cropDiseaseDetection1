import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import DepthwiseConv2D
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --- Monkey-patch DepthwiseConv2D to remove 'groups' keyword ---
_original_from_config = DepthwiseConv2D.from_config


def patched_from_config(config):
    config.pop('groups', None)
    return _original_from_config(config)


DepthwiseConv2D.from_config = patched_from_config

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model_path = 'best_model.h5'
model = load_model(model_path)

# Class labels
class_indices = {
    0: "Aphids",
    1: "Bacterial_Blight",
    2: "Leaf_Curl_Disease",
    3: "Powdery_Mildew",
    4: "Target_spot",
    5: "boll_rot",
    6: "healthy",
    7: "wilt"
}

# Multilingual control measures
control_measures = {
    "Aphids": {
        "English": "Cultural Control: Use crop rotation, maintain healthy plants, and remove weeds.\nBiological Control: Introduce natural predators such as lady beetles, lacewings, and parasitic wasps.\nChemical Control: Use insecticidal soaps or systemic insecticides for severe infestations. Rotate insecticides to prevent resistance.",
        "Marathi": "सांस्कृतिक नियंत्रण: पिकांचा फेरफटका घ्या, निरोगी वनस्पती ठेवा आणि तण काढून टाका.\nजैविक नियंत्रण: लेडी बीटल्स, लेसविंग्स आणि पॅरासिटिक वास्प्स सारख्या नैसर्गिक शिकारी सादर करा.\nरासायनिक नियंत्रण: गंभीर उपद्रवांसाठी कीटकनाशक साबण किंवा प्रणालीगत कीटकनाशकांचा वापर करा. प्रतिकार टाळण्यासाठी कीटकनाशकांची अदलाबदल करा.",
        "Hindi": "सांस्कृतिक नियंत्रण: फसल चक्र अपनाएं, स्वस्थ पौधों को बनाए रखें और खरपतवार को हटा दें।\nजैविक नियंत्रण: लेडी बीटल्स, लेसविंग्स और परजीवी ततैया जैसे प्राकृतिक शिकारी पेश करें।\nरासायनिक नियंत्रण: गंभीर संक्रमण के लिए कीटनाशक साबुन या प्रणालीगत कीटनाशकों का उपयोग करें। प्रतिरोध को रोकने के लिए कीटनाशकों को घुमाएं।"
    },
    "Bacterial_Blight": {
        "English": "Resistant Varieties: Plant cotton varieties resistant to bacterial blight.\nSanitation: Remove and destroy infected plant debris from the field.\nCrop Rotation: Avoid continuous cotton planting in the same field.\nSeed Treatment: Use certified disease-free seeds or treat seeds with appropriate fungicides.",
        "Marathi": "प्रतिरोधक प्रकार: बॅक्टेरियल ब्लाइट-प्रतिरोधक कापूस प्रकार लावा.\nस्वच्छता: शेतातील संक्रमित वनस्पतींचे अवशेष काढून टाका आणि नष्ट करा.\nपिकांची फेरफटका: त्याच शेतात सलग कापूस लागवड टाळा.\nबियाणे प्रक्रिया: प्रमाणित रोगमुक्त बियाण्यांचा वापर करा किंवा योग्य फंगिसाइड्ससह बियाण्यांवर प्रक्रिया करा.",
        "Hindi": "सांस्कृतिक नियंत्रण: फसल चक्र अपनाएं, स्वस्थ पौधों को बनाए रखें और खरपतवार को हटा दें।\nजैविक नियंत्रण: लेडी बीटल्स, लेसविंग्स और परजीवी ततैया जैसे प्राकृतिक शिकारी पेश करें।\nरासायनिक नियंत्रण: गंभीर संक्रमण के लिए कीटनाशक साबुन या प्रणालीगत कीटनाशकों का उपयोग करें। प्रतिरोध को रोकने के लिए कीटनाशकों को घुमाएं।",
    },
    "Leaf_Curl_Disease": {
        "English": "Resistant Varieties: Use leaf curl virus-resistant cotton varieties.\nVector Control: Control the whitefly population, which transmits the virus, using insecticides or biological methods.\nCultural Practices: Practice good field sanitation by removing infected plants.",
        "Marathi": "प्रतिरोधक प्रकार: पानांच्या वळणाच्या विषाणू-प्रतिरोधक कापूस प्रकारांचा वापर करा.\nवाहक नियंत्रण: व्हाईटफ्लायच्या लोकसंख्येवर नियंत्रण ठेवा, जी हा विषाणू प्रसारित करते. कीटकनाशके किंवा जैविक पद्धतींचा वापर करा.\nसांस्कृतिक पद्धती: संक्रमित वनस्पती काढून टाकून चांगली शेत स्वच्छता राखा.",
        "Hindi": "प्रतिरोधी किस्में: पत्तियों के मुड़ने वाले वायरस प्रतिरोधी कपास किस्मों का उपयोग करें।\nवाहक नियंत्रण: सफेद मक्खियों की आबादी को नियंत्रित करें, जो इस वायरस को फैलाती हैं, इसके लिए कीटनाशकों या जैविक विधियों का उपयोग करें।\nसांस्कृतिक प्रथाएं: संक्रमित पौधों को हटा कर खेत की स्वच्छता बनाए रखें।",
    },
    "Powdery_Mildew": {
        "English": "Fungicide Application: Use sulfur-based or other fungicides to manage outbreaks.\nResistant Varieties: Plant varieties that show resistance to powdery mildew.\nCultural Control: Increase air circulation by spacing plants properly.",
        "Marathi": "फंगिसाइड्सचा वापर: गंधक-आधारित किंवा इतर फंगिसाइड्स वापरा.\nप्रतिरोधक प्रकार: पावडरी मिल्ड्यू प्रतिरोधक कापूस प्रकार निवडा.\nसांस्कृतिक नियंत्रण: रोपे व्यवस्थित अंतरावर लावून हवेचा वायुप्रवाह वाढवा.",
        "Hindi": "फफूंदनाशकों का प्रयोग: सल्फर-आधारित या अन्य फफूंदनाशकों का उपयोग करें।\nप्रतिरोधी किस्में: पाउडरी मिल्ड्यू के प्रति प्रतिरोधी किस्में लगाएं।\nसांस्कृतिक नियंत्रण: पौधों को उचित दूरी पर लगाकर हवा के संचार को बढ़ाएं।",
    },
    "Target_spot": {
        "English": "Use Fungicides: Apply fungicides at the early stages of infection.\nCrop Rotation: Rotate with non-host crops to reduce disease carryover.\nRemove Infected Debris: Clear out plant residues after harvesting.",
        "Marathi": "फंगिसाइड्सचा वापर: संसर्गाच्या सुरुवातीच्या टप्प्यावर फंगिसाइड्स वापरा.\nपिकांची फेरफटका: पिकांमध्ये फेरबदल करून रोगाचा प्रसार टाळा.\nसंक्रमित अवशेष काढून टाका: कापणी झाल्यावर शेतातील अवशेष साफ करा.",
        "Hindi": "फफूंदनाशकों का प्रयोग: शुरुआती चरण में फफूंदनाशक का प्रयोग करें।\nफसल चक्र: गैर-होस्ट फसलों के साथ फसल चक्र अपनाएं।\nसंक्रमित अवशेष हटाएं: कटाई के बाद खेत के अवशेष हटा दें।",
    },
    "boll_rot": {
        "English": "Cultural Practices: Improve field drainage to reduce moisture. Avoid dense planting to ensure good airflow.\nSanitation: Remove infected bolls from the field.\nChemical Control: Apply fungicides during wet conditions to reduce boll rot risks.",
        "Marathi": "सांस्कृतिक पद्धती: ओलसरपणा कमी करण्यासाठी शेतातील निचरा सुधारित करा. चांगली वायुप्रवाह सुनिश्चित करण्यासाठी दाट लागवड टाळा.\nस्वच्छता: संक्रमित बोंड शेतातून काढून टाका.\nरासायनिक नियंत्रण: बोंड सडण्याच्या जोखमी कमी करण्यासाठी ओल्या परिस्थितीत फंगिसाइड्स लावा.",
        "Hindi": "सांस्कृतिक प्रथाएं: नमी को कम करने के लिए खेत की जल निकासी में सुधार करें। अच्छा वायु प्रवाह सुनिश्चित करने के लिए घनी खेती से बचें।\nसफाई: संक्रमित बॉल्स को खेत से हटा दें।\nरासायनिक नियंत्रण: गीली परिस्थितियों में बॉल सड़न के जोखिम को कम करने के लिए फफूंदनाशकों का उपयोग करें।",
    },
    "wilt": {
        "English": "Resistant Varieties: Plant cotton varieties resistant to wilt.\nCrop Rotation: Rotate crops to break the disease cycle.\nSoil Treatment: Use soil fumigants or solarization techniques to reduce pathogen levels.",
        "Marathi": "प्रतिरोधक प्रकार: व्रण रोग-प्रतिरोधक कापूस प्रकार लावा.\nपिकांची फेरफटका: रोगाचा प्रसार कमी करण्यासाठी वेगवेगळ्या पिकांची लागवड करा.\nमाती प्रक्रिया: रोगजनकांची पातळी कमी करण्यासाठी मातीचे निर्जंतुकीकरण करा किंवा सौरायन तंत्राचा वापर करा.",
        "Hindi": "प्रतिरोधी किस्में: उकठा रोग प्रतिरोधी कपास किस्में लगाएं।\nफसल चक्र: रोग के चक्र को तोड़ने के लिए फसल चक्र अपनाएं।\nमृदा उपचार: रोगजनकों को कम करने के लिए मृदा का उपचार करें।",
    },
    "healthy": {
        "English": "Your plant is healthy. No action needed!",
        "Marathi": "तुमचे पीक निरोगी आहे. काहीही करायची गरज नाही!",
        "Hindi": "आपका पौधा स्वस्थ है। कोई उपाय आवश्यक नहीं!"
    }
}


def allowed_file(filename):
    """Check if file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_and_preprocess_image(img_path, target_size=(128, 128)):
    """Preprocess image for model prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


def predict_disease(img_path, language):
    """Perform prediction and return disease and control measures."""
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    disease_name = class_indices[predicted_class]

    # Confidence check
    if confidence < 0.7:
        return {"status": "error", "message": "Prediction is uncertain."}

    description = control_measures.get(disease_name, {}).get(language, "No information available.")

    return {
        "status": "success",
        "disease": disease_name,
        "confidence": round(confidence * 100, 2),
        "description": description
    }


@app.route("/", methods=["GET", "POST"])
def upload_image():
    """Render the file upload page and process image uploads."""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]
        if file.filename == "" or not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file format.")

        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Get language choice
        language = request.form.get("language", "English")

        # Predict disease
        result = predict_disease(filepath, language)

        return render_template("result.html", image_path=filepath, **result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
