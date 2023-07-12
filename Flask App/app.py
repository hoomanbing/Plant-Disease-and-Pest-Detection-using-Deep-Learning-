import os
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
model = tf.keras.models.load_model('EfficientNetB4-INSECTS-0.00.h5')
disease_model = tf.keras.models.load_model('plant_disease_model.h5') 
image_size = (200, 200)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.get_json().get('image_data')
    if image_data:
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)
        img = image.load_img(image_path, target_size=image_size)
        img = image.img_to_array(img)
        #img = tf.expand_dims(img, axis=0) / 255.0
        img = tf.expand_dims(img, axis=0) 
        prediction = model.predict(img)
        class_index = prediction.argmax()
        pest_list = [
    "rice leaf roller",
    "rice leaf caterpillar",
    "Paddy Stem Maggot",
    "asiatic rice borer",
    "yellow rice borer",
    "rice gall midge",
    "Rice Stemfly",
    "brown plant hopper",
    "white backed plant hopper",
    "small brown plant hopper",
    "rice water weevil",
    "rice leafhopper",
    "grain spreader thrips",
    "rice shell pest",
    "grub",
    "mole cricket",
    "wireworm",
    "white margined moth",
    "black cutworm",
    "large cutworm",
    "yellow cutworm",
    "red spider",
    "corn borer",
    "army worm",
    "aphids",
    "Potosiabre vitarsis",
    "peach borer",
    "english grain aphid",
    "green bug",
    "bird cherry-oataphid",
    "wheat blossom midge",
    "penthaleus major",
    "longlegged spider mite",
    "wheat phloeothrips",
    "wheat sawfly",
    "cerodonta denticornis",
    "beet fly",
    "flea beetle",
    "cabbage army worm",
    "beet army worm",
    "Beet spot flies",
    "meadow moth",
    "beet weevil",
    "sericaorient alismots chulsky",
    "alfalfa weevil",
    "flax budworm",
    "alfalfa plant bug",
    "tarnished plant bug",
    "Locustoidea",
    "lytta polita",
    "legume blister beetle",
    "blister beetle",
    "therioaphis maculata Buckton",
    "odontothrips loti",
    "Thrips",
    "alfalfa seed chalcid",
    "Pieris canidia",
    "Apolygus lucorum",
    "Limacodidae",
    "Viteus vitifoliae",
    "Colomerus vitis",
    "Brevipoalpus lewisi McGregor",
    "oides decempunctata",
    "Polyphagotars onemus latus",
    "Pseudococcus comstocki Kuwana",
    "parathrene regalis",
    "Ampelophaga",
    "Lycorma delicatula",
    "Xylotrechus",
    "Cicadella viridis",
    "Miridae",
    "Trialeurodes vaporariorum",
    "Erythroneura apicalis",
    "Papilio xuthus",
    "Panonchus citri McGregor",
    "Phyllocoptes oleiverus ashmead",
    "Icerya purchasi Maskell",
    "Unaspis yanonensis",
    "Ceroplastes rubens",
    "Chrysomphalus aonidum",
    "Parlatoria zizyphus Lucus",
    "Nipaecoccus vastalor",
    "Aleurocanthus spiniferus",
    "Tetradacus c Bactrocera minax",
    "Dacus dorsalis(Hendel)",
    "Bactrocera tsuneonis",
    "Prodenia litura",
    "Adristyrannus",
    "Phyllocnistis citrella Stainton",
    "Toxoptera citricidus",
    "Toxoptera aurantii",
    "Aphis citricola Vander Goot",
    "Scirtothrips dorsalis Hood",
    "Dasineura sp",
    "Lawana imitata Melichar",
    "Salurnis marginella Guerr",
    "Deporaus marginatus Pascoe",
    "Chlumetia transversa",
    "Mango flat beak leafhopper",
    "Rhytidodera bowrinii white",
    "Sternochetus frigidus",
    "Cicadellidae"
]

        result = pest_list[class_index]
        return jsonify({'prediction': result})
    return jsonify({'prediction': 'Error: No image data'})

@app.route('/disease-predict', methods=['POST'])
def disease_predict():
    image_size= (224,224)
    image_data = request.get_json().get('image_data')
    if image_data:
        image_data = image_data.replace('data:image/png;base64,', '')
        image_data = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.png')
        with open(image_path, 'wb') as f:
            f.write(image_data)
        img = image.load_img(image_path, target_size=image_size)
        img = image.img_to_array(img)
        #img = tf.expand_dims(img, axis=0) / 255.0
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.expand_dims(img, axis=0) 
        prediction = model.predict(img)
        class_index = prediction.argmax()
        class_labels = {
            0: 'Rice Scab',
            1: 'Rice Leaf Rot',
            2: 'Rice  Leaf Scab',
            3: 'Rice Healthy',
            4: 'Blueberry Healthy',
            5: 'Cherry Powdery Mildew',
            6: 'Cherry Healthy',
            7: 'Cercospora_leaf_spot in Corn',
            8: 'Common rust in Corn',
            9: 'Northern_Leaf_Blight',
            10: 'Corn_(maize)___healthy',
            11: 'Grape___Black_rot',
            12: 'Grape___Esca_(Black_Measles)',
            13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            14: 'Grape___healthy',
            15: 'Orange___Haunglongbing_(Citrus_greening)',
            16: 'Peach___Bacterial_spot',
            17: 'Peach___healthy',
            18: 'Pepper,_bell___Bacterial_spot',
            19: 'Pepper,_bell___healthy',
            20: 'Potato___Early_blight',
            21: 'Potato___Late_blight',
            22: 'Potato___healthy',
            23: 'Raspberry___healthy',
            24: 'Soybean___healthy',
            25: 'Squash___Powdery_mildew',
            26: 'Strawberry___Leaf_scorch',
            27: 'Strawberry___healthy',
            28: 'Tomato___Bacterial_spot',
            29: 'Tomato___Early_blight',
            30: 'Tomato___Late_blight',
            31: 'Tomato___Leaf_Mold',
            32: 'Tomato___Septoria_leaf_spot',
            33: 'Tomato___Spider_mites Two-spotted_spider_mite',
            34: 'Tomato___Target_Spot',
            35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            36: 'Tomato___Tomato_mosaic_virus',
            37: 'Tomato___healthy'
        }
        result = class_labels[class_index]
        return jsonify({'prediction': result})
    return jsonify({'prediction': 'Error: No image data'})
if __name__ == '__main__':
    app.run(debug=True)
