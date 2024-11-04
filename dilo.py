import torch
import flask
from flask import Flask, request, redirect, url_for, render_template_string, flash
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from pyngrok import ngrok

# Initialize Flask app
app = flask.Flask(__name__)
app.secret_key = 'fheuohsk'

# Define the upload folder
UPLOAD_FOLDER = 'flaskimgs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(r'C:\Users\dochu\OneDrive\Pictures\pythonstuff\custom-model_upd_latest.pt', map_location=device)
#model.load('custom-model-sd_upd.pt')
model.eval()  # Set model to evaluation mode
dog_classes_rev={'Blue_Heeler': 0,
 'n02085620-Chihuahua': 1,
 'n02085782-Japanese_spaniel': 2,
 'n02085936-Maltese_dog': 3,
 'n02086079-Pekinese': 4,
 'n02086240-Shih-Tzu': 5,
 'n02086646-Blenheim_spaniel': 6,
 'n02086910-papillon': 7,
 'n02087046-toy_terrier': 8,
 'n02087394-Rhodesian_ridgeback': 9,
 'n02088094-Afghan_hound': 10,
 'n02088238-basset': 11,
 'n02088364-beagle': 12,
 'n02088466-bloodhound': 13,
 'n02088632-bluetick': 14,
 'n02089078-black-and-tan_coonhound': 15,
 'n02089867-Walker_hound': 16,
 'n02089973-English_foxhound': 17,
 'n02090379-redbone': 18,
 'n02090622-borzoi': 19,
 'n02090721-Irish_wolfhound': 20,
 'n02091032-Italian_greyhound': 21,
 'n02091134-whippet': 22,
 'n02091244-Ibizan_hound': 23,
 'n02091467-Norwegian_elkhound': 24,
 'n02091635-otterhound': 25,
 'n02091831-Saluki': 26,
 'n02092002-Scottish_deerhound': 27,
 'n02092339-Weimaraner': 28,
 'n02093256-Staffordshire_bullterrier': 29,
 'n02093428-American_Staffordshire_terrier': 30,
 'n02093647-Bedlington_terrier': 31,
 'n02093754-Border_terrier': 32,
 'n02093859-Kerry_blue_terrier': 33,
 'n02093991-Irish_terrier': 34,
 'n02094114-Norfolk_terrier': 35,
 'n02094258-Norwich_terrier': 36,
 'n02094433-Yorkshire_terrier': 37,
 'n02095314-wire-haired_fox_terrier': 38,
 'n02095570-Lakeland_terrier': 39,
 'n02095889-Sealyham_terrier': 40,
 'n02096051-Airedale': 41,
 'n02096177-cairn': 42,
 'n02096294-Australian_terrier': 43,
 'n02096437-Dandie_Dinmont': 44,
 'n02096585-Boston_bull': 45,
 'n02097047-miniature_schnauzer': 46,
 'n02097130-giant_schnauzer': 47,
 'n02097209-standard_schnauzer': 48,
 'n02097298-Scotch_terrier': 49,
 'n02097474-Tibetan_terrier': 50,
 'n02097658-silky_terrier': 51,
 'n02098105-soft-coated_wheaten_terrier': 52,
 'n02098286-West_Highland_white_terrier': 53,
 'n02098413-Lhasa': 54,
 'n02099267-flat-coated_retriever': 55,
 'n02099429-curly-coated_retriever': 56,
 'n02099601-golden_retriever': 57,
 'n02099712-Labrador_retriever': 58,
 'n02099849-Chesapeake_Bay_retriever': 59,
 'n02100236-German_short-haired_pointer': 60,
 'n02100583-vizsla': 61,
 'n02100735-English_setter': 62,
 'n02100877-Irish_setter': 63,
 'n02101006-Gordon_setter': 64,
 'n02101388-Brittany_spaniel': 65,
 'n02101556-clumber': 66,
 'n02102040-English_springer': 67,
 'n02102177-Welsh_springer_spaniel': 68,
 'n02102318-cocker_spaniel': 69,
 'n02102480-Sussex_spaniel': 70,
 'n02102973-Irish_water_spaniel': 71,
 'n02104029-kuvasz': 72,
 'n02104365-schipperke': 73,
 'n02105056-groenendael': 74,
 'n02105162-malinois': 75,
 'n02105251-briard': 76,
 'n02105412-kelpie': 77,
 'n02105505-komondor': 78,
 'n02105641-Old_English_sheepdog': 79,
 'n02105855-Shetland_sheepdog': 80,
 'n02106030-collie': 81,
 'n02106166-Border_collie': 82,
 'n02106382-Bouvier_des_Flandres': 83,
 'n02106550-Rottweiler': 84,
 'n02106662-German_shepherd': 85,
 'n02107142-Doberman': 86,
 'n02107312-miniature_pinscher': 87,
 'n02107574-Greater_Swiss_Mountain_dog': 88,
 'n02107683-Bernese_mountain_dog': 89,
 'n02107908-Appenzeller': 90,
 'n02108000-EntleBucher': 91,
 'n02108089-boxer': 92,
 'n02108422-bull_mastiff': 93,
 'n02108551-Tibetan_mastiff': 94,
 'n02108915-French_bulldog': 95,
 'n02109047-Great_Dane': 96,
 'n02109525-Saint_Bernard': 97,
 'n02109961-Eskimo_dog': 98,
 'n02110063-malamute': 99,
 'n02110185-Siberian_husky': 100,
 'n02110627-affenpinscher': 101,
 'n02110806-basenji': 102,
 'n02110958-pug': 103,
 'n02111129-Leonberg': 104,
 'n02111277-Newfoundland': 105,
 'n02111500-Great_Pyrenees': 106,
 'n02111889-Samoyed': 107,
 'n02112018-Pomeranian': 108,
 'n02112137-chow': 109,
 'n02112350-keeshond': 110,
 'n02112706-Brabancon_griffon': 111,
 'n02113023-Pembroke': 112,
 'n02113186-Cardigan': 113,
 'n02113624-toy_poodle': 114,
 'n02113712-miniature_poodle': 115,
 'n02113799-standard_poodle': 116,
 'n02113978-Mexican_hairless': 117,
 'n02115641-dingo': 118,
 'n02115913-dhole': 119,
 'n02116738-African_hunting_dog': 120}

dog_classes=dict(zip(dog_classes_rev.values(), dog_classes_rev.keys()))


#website_url=ngrok.connect(5000)
#print(f'URL: {website_url}')

# Prediction function
def make_predictions(model: torch.nn.Module, data: torch.Tensor, device: torch.device = device):
    data = data.to(device)  # Add batch dimension and send to device
    with torch.inference_mode():
        logits = model(data)  # Forward pass
        pred_prob = torch.softmax(logits.squeeze(), dim=0)  # Apply softmax
        top_preds, top_probs= torch.topk(pred_prob, 5)
        top_preds = [dog_classes.get(int(pred.item())) for pred in top_preds]
    return pred_prob.cpu()
#top_preds, top_probs= torch.topk(pred_prob, 5)
# Route to render upload form
@app.route('/')
def upload_form():
    upload_html = """
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Dog Breed Prediction</title>
      </head>
      <body>
        <h1>Upload an Image for Dog Breed Prediction</h1>
        <form action="/upload" method="POST" enctype="multipart/form-data">
          <input type="file" name="file"><br><br>
          <button type="submit">Upload</button>
        </form>
        <ul>
          {% with messages = get_flashed_messages() %}
            {% if messages %}
              {% for message in messages %}
                <li>{{ message }}</li>
              {% endfor %}
            {% endif %}
          {% endwith %}
        </ul>
        {% if image_url %}
          <h2>Uploaded Image:</h2>
          <img src="{{ image_url }}" alt="Uploaded Image" width="300"><br>
        {% endif %}
        {% if prediction %}
          <h2>Prediction: {{ prediction }}</h2>
        {% endif %}
      </body>
    </html>
    """
    return render_template_string(upload_html, image_url=None)

# Route to handle file upload and predictions
@app.route('/upload', methods=['POST'])
def upload_the_file():
    file = request.files.get('file')  # Get the uploaded file
    if file and file.filename != '':  # Check if the file is valid
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)  # Save the file to the upload folder

        # Preprocess the uploaded image
        image = Image.open(file_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(image).unsqueeze(0)  # Convert to tensor and add batch dimension

        # Make prediction
        pred_prob = make_predictions(model, image_tensor)
        #pred_label_num = torch.argmax(pred_probs).item()  # Get predicted class
        #pred_label=dog_classes.get(pred_label_num)
        #for pred, prob in zip(top_preds, top_prob):
            #pred_label_num = torch.argmax(top_preds).item()
            #pred_label=dog_classes.get(top_preds)
        print(pred_prob)
        pred_text="Predicted Labels:\n"
        for i, prob in enumerate(pred_prob):
          label = dog_classes.get(i)
          if prob > float(0.01):
            breed_name=label.split('-')[-1]
            prob=prob*100
            pred_text += f"{breed_name}: {prob:.4f}%\n"
        
        
        # Flash success message with the prediction
        flash(f'Success! Predicted label: {pred_text}')

        # Display the uploaded image along with the prediction
        return render_template_string(
            upload_form(), image_url=f"/{UPLOAD_FOLDER}/{file.filename}"
        )
    else:
        flash('Error: No file selected or invalid file. Please try again.')
        return redirect(url_for('upload_form'))

# Serve uploaded files
@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    return flask.send_from_directory(UPLOAD_FOLDER, filename)

# Run the Flask app
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(debug=True)

