import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("FRP.csv")

# Rename columns for consistency
df.rename(columns={
    'Organic Fertilizer': 'Organic_Fertilizer', 
    'Soil Type': 'Soil_Type', 
    'Crop Type': 'Crop_Type', 
    'Temparature': 'Temperature', 
    'Humidity ': 'Humidity'
}, inplace=True)

# Check and clean unnecessary columns if any
if 'Unnamed: 9' in df.columns:
    df.drop('Unnamed: 9', axis=1, inplace=True)

# Label encoding for Organic_Fertilizer, video, Soil_Type, Crop_Type, and Image target variable
label_encoder_fertilizer = LabelEncoder()
df['Encoded_Organic_Fertilizer'] = label_encoder_fertilizer.fit_transform(df['Organic_Fertilizer'])

label_encoder_video = LabelEncoder()
df['Encoded_Video'] = label_encoder_video.fit_transform(df['video'])

label_encoder_soil = LabelEncoder()
df['Encoded_Soil_Type'] = label_encoder_soil.fit_transform(df['Soil_Type'])

label_encoder_crop = LabelEncoder()
df['Encoded_Crop_Type'] = label_encoder_crop.fit_transform(df['Crop_Type'])

label_encoder_image = LabelEncoder()
df['Encoded_Image'] = label_encoder_image.fit_transform(df['image'])  # New image target column

# Define input features and target variables
X = df[['Temperature', 'Humidity', 'Moisture', 'Encoded_Soil_Type', 'Encoded_Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
Y_fertilizer = df['Encoded_Organic_Fertilizer']
Y_video = df['Encoded_Video']
Y_image = df['Encoded_Image']  # New target for image

# Split the data into training and testing sets
X_train, X_test, Y_train_fertilizer, Y_test_fertilizer = train_test_split(X, Y_fertilizer, test_size=0.2, random_state=42, shuffle=True)
_, _, Y_train_video, Y_test_video = train_test_split(X, Y_video, test_size=0.2, random_state=42, shuffle=True)
_, _, Y_train_image, Y_test_image = train_test_split(X, Y_image, test_size=0.2, random_state=42, shuffle=True)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train models
model_fertilizer = RandomForestClassifier(random_state=42)
model_fertilizer.fit(X_train, Y_train_fertilizer)

model_video = RandomForestClassifier(random_state=42)
model_video.fit(X_train, Y_train_video)

model_image = RandomForestClassifier(random_state=42)  # Image model
model_image.fit(X_train, Y_train_image)

# Function for recommendation, updated to include image prediction
def recommend(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    try:
        encoded_soil_type = label_encoder_soil.transform([str(Soil_Type)])[0]  
        encoded_crop_type = label_encoder_crop.transform([str(Crop_Type)])[0]
    except ValueError:
        print(f"Error: Soil_Type '{Soil_Type}' or Crop_Type '{Crop_Type}' not found in training data.")
        return None, None, None
    
    input_data = np.array([[Temperature, Humidity, Moisture, encoded_soil_type, encoded_crop_type, Nitrogen, Potassium, Phosphorous]])
    
    # Scale input data using the fitted scaler
    scaled_data = scaler.transform(input_data)

    # Predict using the models
    encoded_fertilizer = model_fertilizer.predict(scaled_data)
    encoded_video = model_video.predict(scaled_data)
    encoded_image = model_image.predict(scaled_data)

    # Decode predictions
    fertilizer_name = label_encoder_fertilizer.inverse_transform(encoded_fertilizer)[0]
    video_link = label_encoder_video.inverse_transform(encoded_video)[0]
    image_label = label_encoder_image.inverse_transform(encoded_image)[0]

    return fertilizer_name, video_link, image_label

# Example usage
Temperature = 29
Humidity = 54
Moisture = 45
Soil_Type = 'Loamy'
Crop_Type = 'Sugarcane'
Nitrogen = 12
Potassium = 0
Phosphorous = 36

fertilizer, video, image = recommend(Temperature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous)
print("Recommended Fertilizer:", fertilizer)
print("Recommended Video:", video)
print("Recommended Image Label:", image)

# Save models and encoders
joblib.dump(model_fertilizer, 'fertilizer_model.pkl')
joblib.dump(model_video, 'video_model.pkl')
joblib.dump(model_image, 'image_model.pkl')  # Save the new image model
joblib.dump(label_encoder_fertilizer, 'label_encoder_fertilizer.pkl')
joblib.dump(label_encoder_video, 'label_encoder_video.pkl')
joblib.dump(label_encoder_image, 'label_encoder_image.pkl')  # Save the image encoder
joblib.dump(label_encoder_soil, 'label_encoder_soil.pkl')
joblib.dump(label_encoder_crop, 'label_encoder_crop.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')

print("Models and Encoders saved successfully!")