import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your dataset
df = pd.read_csv('data2.csv')

# Encode the 'Substance' column to numerical values
le = LabelEncoder()
df['Substance'] = le.fit_transform(df['Substance'])

# Split the dataset into features (X) and labels (y)
X = df['Substance']
y = df['Impact']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Convert 'Substance' to sequences and pad them
max_words = len(le.classes_)
X_train = pad_sequences(X_train.to_numpy().reshape(-1, 1), maxlen=1)
X_test = pad_sequences(X_test.to_numpy().reshape(-1, 1), maxlen=1)

# Create a simple feedforward neural network
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=10, input_length=1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >  0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Test Accuracy: {accuracy}')

# Save the model
model.save("model.h5")
