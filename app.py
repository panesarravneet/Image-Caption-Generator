from flask import Flask, render_template, request, url_for
import os
from keras.models import load_model, Model  # Import the Model class
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pickle import load



base_model = InceptionV3(weights='model/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)

model = load_model('model/model-25-10-2023.h5')

with open("model/wordtoix.pkl", "rb") as pickle_in:
    wordtoix = load(pickle_in)
    
with open("model/ixtoword.pkl", "rb") as pickle_in:
    ixtoword = load(pickle_in)


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}


# # Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
# word_to_index = {}
# with open ("textData/word_to_idx.pkl", 'rb') as file:
#     word_to_index = pd.read_pickle(file, compression=None)

# index_to_word = {}
# with open ("textData/idx_to_word.pkl", 'rb') as file:
#     index_to_word = pd.read_pickle(file, compression=None)



print("Loading the model...")
# model_res = load_model('model/model_1.h5')

# resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
# resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)



# Generate Captions for a random image
# Using Greedy Search Algorithm

def beam_search(image, beam_index=3):
    start = [wordtoix["startseq"]]
    
    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]
    max_length = 74
    
    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
            e = image
            preds = model.predict([e, np.array(par_caps)])
            
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def predict_caption(photo):
    # Using beam_search with k=3
    caption = beam_search(photo, 3)

    return caption

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image(img_path):
    image = preprocess_img(img_path)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

def runModel(img_name):
    print("Encoding the image ...")
    photo = encode_image(img_name).reshape((1, 2048))

    print("Running model to generate the caption...")
    caption = predict_caption(photo)
    print(caption)

    return caption

def allowed_file(filename):
    """
    Check if the filename is allowed based on its extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_uploads_directory():
    upload_folder = "uploads"
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



app = Flask(__name__, static_folder='uploads')

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ""
    img_url = None  # Initialize img_url to None

    if request.method == 'POST':
        if 'image_file' in request.files:
            img_file = request.files['image_file']

            if img_file.filename == '':
                caption = "No selected file."

            elif not allowed_file(img_file.filename):
                # Handle the invalid file format here immediately upon upload
                caption = "Invalid file format. Please upload a .jpg, .jpeg, or .png file."

            else:
                # Clear the uploads directory before saving new image
                clear_uploads_directory()
                
                img_path = os.path.join("uploads", img_file.filename)
                img_file.save(img_path)

                # Get the URL for the uploaded image
                img_url = url_for('static', filename=img_file.filename)

                caption = runModel(img_path)

    return render_template('index.html', caption=caption, img_url=img_url)  # Pass img_url to the template

if __name__ == '__main__':
    app.run(debug=True)