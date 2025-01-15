from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Define symptoms and labels
symptoms_columns = [f'Symptom_{i}' for i in range(1, 18)]
X = dataset[symptoms_columns].fillna('None')  # Handle missing values

# Convert symptom strings to integers using LabelEncoder
for col in symptoms_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Disease'])

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the model, scaler, and label encoder
model.save('disease_prediction_model.keras')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
