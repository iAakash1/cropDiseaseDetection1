import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, ttk
from tensorflow.keras.layers import DepthwiseConv2D

# --- Monkey-patch DepthwiseConv2D to remove the 'groups' keyword if present ---
_original_from_config = DepthwiseConv2D.from_config

def patched_from_config(config):
    # Remove 'groups' if it exists, because it's not recognized by the current Keras version.
    config.pop('groups', None)
    return _original_from_config(config)

DepthwiseConv2D.from_config = patched_from_config
# --------------------------------------------------------------

# Load the trained model from the folder
model_path = r'best_model.h5'
model = load_model(model_path)

# Class indices (automatic mapping based on your training)
class_indices = {
    'Aphids': 0,
    'Bacterial_Blight': 1,
    'Leaf_Curl_Disease': 2,
    'Powdery_Mildew': 3,
    'Target_spot': 4,
    'boll_rot': 5,
    'healthy': 6,
    'wilt': 7
}

# Control measures for diseases in English
control_measures_en = {
    "Aphids": "Cultural Control: Use crop rotation, maintain healthy plants, and remove weeds.\nBiological Control: Introduce natural predators such as lady beetles, lacewings, and parasitic wasps.\nChemical Control: Use insecticidal soaps or systemic insecticides for severe infestations. Rotate insecticides to prevent resistance.",
    "Bacterial_Blight": "Resistant Varieties: Plant cotton varieties resistant to bacterial blight.\nSanitation: Remove and destroy infected plant debris from the field.\nCrop Rotation: Avoid continuous cotton planting in the same field.\nSeed Treatment: Use certified disease-free seeds or treat seeds with appropriate fungicides.",
    "boll_rot": "Cultural Practices: Improve field drainage to reduce moisture. Avoid dense planting to ensure good airflow.\nSanitation: Remove infected bolls from the field.\nChemical Control: Apply fungicides during wet conditions to reduce boll rot risks.",
    "Leaf_Curl_Disease": "Resistant Varieties: Use leaf curl virus-resistant cotton varieties.\nVector Control: Control the whitefly population, which transmits the virus, using insecticides or biological methods.\nCultural Practices: Practice good field sanitation by removing infected plants.",
    "Powdery_Mildew": "Fungicide Application: Use sulfur-based or other fungicides to manage outbreaks.\nResistant Varieties: Plant varieties that show resistance to powdery mildew.\nCultural Control: Increase air circulation by spacing plants properly.",
    "Target_spot": "Use Fungicides: Apply fungicides at the early stages of infection.\nCrop Rotation: Rotate with non-host crops to reduce disease carryover.\nRemove Infected Debris: Clear out plant residues after harvesting.",
    "wilt": "Resistant Varieties: Plant cotton varieties resistant to wilt.\nCrop Rotation: Rotate crops to break the disease cycle.\nSoil Treatment: Use soil fumigants or solarization techniques to reduce pathogen levels.",
}

# Control measures for diseases in Marathi
control_measures_mr = {
    "Aphids": "सांस्कृतिक नियंत्रण: पिकांचा फेरफटका घ्या, निरोगी वनस्पती ठेवा आणि तण काढून टाका.\nजैविक नियंत्रण: लेडी बीटल्स, लेसविंग्स आणि पॅरासिटिक वास्प्स सारख्या नैसर्गिक शिकारी सादर करा.\nरासायनिक नियंत्रण: गंभीर उपद्रवांसाठी कीटकनाशक साबण किंवा प्रणालीगत कीटकनाशकांचा वापर करा. प्रतिकार टाळण्यासाठी कीटकनाशकांची अदलाबदल करा.",
    "Bacterial_Blight": "प्रतिरोधक प्रकार: बॅक्टेरियल ब्लाइट-प्रतिरोधक कापूस प्रकार लावा.\nस्वच्छता: शेतातील संक्रमित वनस्पतींचे अवशेष काढून टाका आणि नष्ट करा.\nपिकांची फेरफटका: त्याच शेतात सलग कापूस लागवड टाळा.\nबियाणे प्रक्रिया: प्रमाणित रोगमुक्त बियाण्यांचा वापर करा किंवा योग्य फंगिसाइड्ससह बियाण्यांवर प्रक्रिया करा.",
    "boll_rot": "सांस्कृतिक पद्धती: ओलसरपणा कमी करण्यासाठी शेतातील निचरा सुधारित करा. चांगली वायुप्रवाह सुनिश्चित करण्यासाठी दाट लागवड टाळा.\nस्वच्छता: संक्रमित बोंड शेतातून काढून टाका.\nरासायनिक नियंत्रण: बोंड सडण्याच्या जोखमी कमी करण्यासाठी ओल्या परिस्थितीत फंगिसाइड्स लावा.",
    "Leaf_Curl_Disease": "प्रतिरोधक प्रकार: पानांच्या वळणाच्या विषाणू-प्रतिरोधक कापूस प्रकारांचा वापर करा.\nवाहक नियंत्रण: व्हाईटफ्लायच्या लोकसंख्येवर नियंत्रण ठेवा, जी हा विषाणू प्रसारित करते. कीटकनाशके किंवा जैविक पद्धतींचा वापर करा.\nसांस्कृतिक पद्धती: संक्रमित वनस्पती काढून टाकून चांगली शेत स्वच्छता राखा.",
    "Powdery_Mildew": "फंगिसाइड्सचा वापर: गंधक-आधारित किंवा इतर फंगिसाइड्स वापरा.\nप्रतिरोधक प्रकार: पावडरी मिल्ड्यू प्रतिरोधक कापूस प्रकार निवडा.\nसांस्कृतिक नियंत्रण: रोपे व्यवस्थित अंतरावर लावून हवेचा वायुप्रवाह वाढवा.",
    "Target_spot": "फंगिसाइड्सचा वापर: संसर्गाच्या सुरुवातीच्या टप्प्यावर फंगिसाइड्स वापरा.\nपिकांची फेरफटका: पिकांमध्ये फेरबदल करून रोगाचा प्रसार टाळा.\nसंक्रमित अवशेष काढून टाका: कापणी झाल्यावर शेतातील अवशेष साफ करा.",
    "wilt": "प्रतिरोधक प्रकार: व्रण रोग-प्रतिरोधक कापूस प्रकार लावा.\nपिकांची फेरफटका: रोगाचा प्रसार कमी करण्यासाठी वेगवेगळ्या पिकांची लागवड करा.\nमाती प्रक्रिया: रोगजनकांची पातळी कमी करण्यासाठी मातीचे निर्जंतुकीकरण करा किंवा सौरायन तंत्राचा वापर करा.",
}

# Control measures for diseases in Hindi
control_measures_hi = {
    "Aphids": "सांस्कृतिक नियंत्रण: फसल चक्र अपनाएं, स्वस्थ पौधों को बनाए रखें और खरपतवार को हटा दें।\nजैविक नियंत्रण: लेडी बीटल्स, लेसविंग्स और परजीवी ततैया जैसे प्राकृतिक शिकारी पेश करें।\nरासायनिक नियंत्रण: गंभीर संक्रमण के लिए कीटनाशक साबुन या प्रणालीगत कीटनाशकों का उपयोग करें। प्रतिरोध को रोकने के लिए कीटनाशकों को घुमाएं।",
    "Bacterial_Blight": "प्रतिरोधी किस्में: बैक्टीरियल ब्लाइट प्रतिरोधी कपास की किस्में लगाएं।\nसफाई: संक्रमित पौधों के मलबे को हटाकर नष्ट कर दें।\nफसल चक्र: एक ही खेत में लगातार कपास की खेती से बचें।\nबीज उपचार: प्रमाणित रोग मुक्त बीजों का उपयोग करें या उपयुक्त फफूंदनाशकों से बीजों का उपचार करें।",
    "boll_rot": "सांस्कृतिक प्रथाएं: नमी को कम करने के लिए खेत की जल निकासी में सुधार करें। अच्छा वायु प्रवाह सुनिश्चित करने के लिए घनी खेती से बचें।\nसफाई: संक्रमित बॉल्स को खेत से हटा दें।\nरासायनिक नियंत्रण: गीली परिस्थितियों में बॉल सड़न के जोखिम को कम करने के लिए फफूंदनाशकों का उपयोग करें।",
    "Leaf_Curl_Disease": "प्रतिरोधी किस्में: पत्तियों के मुड़ने वाले वायरस प्रतिरोधी कपास किस्मों का उपयोग करें।\nवाहक नियंत्रण: सफेद मक्खियों की आबादी को नियंत्रित करें, जो इस वायरस को फैलाती हैं, इसके लिए कीटनाशकों या जैविक विधियों का उपयोग करें।\nसांस्कृतिक प्रथाएं: संक्रमित पौधों को हटा कर खेत की स्वच्छता बनाए रखें।",
    "Powdery_Mildew": "फफूंदनाशकों का प्रयोग: सल्फर-आधारित या अन्य फफूंदनाशकों का उपयोग करें।\nप्रतिरोधी किस्में: पाउडरी मिल्ड्यू के प्रति प्रतिरोधी किस्में लगाएं।\nसांस्कृतिक नियंत्रण: पौधों को उचित दूरी पर लगाकर हवा के संचार को बढ़ाएं।",
    "Target_spot": "फफूंदनाशकों का प्रयोग: शुरुआती चरण में फफूंदनाशक का प्रयोग करें।\nफसल चक्र: गैर-होस्ट फसलों के साथ फसल चक्र अपनाएं।\nसंक्रमित अवशेष हटाएं: कटाई के बाद खेत के अवशेष हटा दें।",
    "wilt": "प्रतिरोधी किस्में: उकठा रोग प्रतिरोधी कपास किस्में लगाएं।\nफसल चक्र: रोग के चक्र को तोड़ने के लिए फसल चक्र अपनाएं।\nमृदा उपचार: रोगजनकों को कम करने के लिए मृदा का उपचार करें।",
}

def load_and_preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image
    return img_array

def show_popup(disease_name, measures, bg_color, language):
    popup = tk.Toplevel()
    popup.title("Prediction Result")
    popup.configure(bg=bg_color)

    # Determine label color based on background
    label_color = "black" if bg_color in ["orange", "green"] else "white"
    message = (f"{disease_name} detected!\n\n{measures}" if language == "English"
               else f"{disease_name} आढळले!\n\n{measures}" if language == "Marathi"
               else f"{disease_name} पाया गया!\n\n{measures}")
    label = tk.Label(popup, text=message, fg=label_color, bg=bg_color, font=("Arial", 12), wraplength=350)
    label.pack(padx=20, pady=20)

    # Center the popup window
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (popup.winfo_width() // 2)
    y = (popup.winfo_screenheight() // 2) - (popup.winfo_height() // 2)
    popup.geometry(f"+{x}+{y}")

    # Add a close button
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=10)

def predict_disease(model, img_path, class_indices, language):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    disease_name = list(class_indices.keys())[predicted_class]

    confidence_threshold = 0.7
    if confidence < confidence_threshold:
        show_popup("Unable to predict", "", "red", language)
    elif confidence < 0.85:
        show_popup("Prediction is uncertain", "", "orange", language)
    elif disease_name == "healthy":
        show_popup("Healthy", "", "green", language)
    else:
        measures = (control_measures_en[disease_name]
                    if language == "English"
                    else control_measures_mr.get(disease_name, "No information available.")
                    if language == "Marathi"
                    else control_measures_hi.get(disease_name, "No information available."))
        show_popup(disease_name, measures, "orange", language)

def upload_image(language):
    img_path = filedialog.askopenfilename(parent=root, title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if img_path:
        predict_disease(model, img_path, class_indices, language)

def select_language():
    def proceed():
        selected_language = language_var.get()
        language_window.destroy()
        upload_image(selected_language)

    language_window = tk.Toplevel(root)
    language_window.title("Select Language")
    language_window.geometry("300x200")
    language_window.lift()
    language_window.grab_set()
    language_window.focus_force()

    tk.Label(language_window, text="Choose your language:", font=("Arial", 12)).pack(pady=10)
    language_var = tk.StringVar(value="English")

    ttk.Radiobutton(language_window, text="English", variable=language_var, value="English").pack(anchor="w", padx=20)
    ttk.Radiobutton(language_window, text="मराठी", variable=language_var, value="Marathi").pack(anchor="w", padx=20)
    ttk.Radiobutton(language_window, text="हिंदी", variable=language_var, value="Hindi").pack(anchor="w", padx=20)

    tk.Button(language_window, text="Proceed", command=proceed).pack(pady=10)

# Create main Tkinter window
root = tk.Tk()
root.title("Cotton Disease Prediction")
root.geometry("300x150")

# Add a button to open the language selection window
tk.Button(root, text="Start Prediction", command=select_language).pack(pady=20)

# Start the GUI event loop
root.mainloop()
