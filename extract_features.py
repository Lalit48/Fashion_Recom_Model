import os
import numpy as np
import pickle as pkl
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.models import Sequential
from numpy.linalg import norm

# Define the model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([model, GlobalMaxPool2D()])

# Function to extract features from an image
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_expand)
    features = model.predict(img_preprocessed).flatten()
    norm_features = features / norm(features)
    return norm_features

# Directory containing your images
img_dir = r'C:\Users\lalit\Downloads\Fashion_Recom_Model\images'
filenames = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

# Extract features and store them
features = []
for file in filenames:
    features.append(extract_features(file))

# Convert the list to a numpy array
features = np.array(features)

# Save the features and filenames using pickle
with open('Images_features.pkl', 'wb') as f:
    pkl.dump(features, f)

with open('filenames.pkl', 'wb') as f:
    pkl.dump(filenames, f)

print("Features and filenames saved successfully.")
