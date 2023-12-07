#%%
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

#%%
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

#%%
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, SpatialDropout1D, Input
from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Attention, Concatenate, GlobalMaxPooling1D
from keras.models import Model

#%%
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertModel, RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup, RobertaModel, RobertaConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import ModuleList

#%%
from scipy.stats import f_oneway

#%%
# Download NLTK resources (if not already)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#%%
# Loading the airline tweets dataset
tweets_data = pd.read_csv("Tweets.csv")
tweets_data.head()

#%%
# Identifying Missing Values
missing_vals = tweets_data.isnull().sum()
print("Columns with Missing Values:")
print(missing_vals[missing_vals > 0])

#%%
#Now after seeing the columns with missing values, we should drop the columns that have missing values which are airline_sentiment_gold, neagtivereason_gold and tweet_coord as these have more than 30% data missing and these columns are not that relevant in predicting sentiment of the airline tweet. So it's better to drop these columns.
tweets_data = tweets_data.dropna(axis=1)

#%%
# Identifying Missing Values
missing_vals = tweets_data.isnull().sum()
print("Columns with Missing Values:")
print(missing_vals[missing_vals > 0])
# Now we don't have any missing values columns

#%%
# Now we will drop all the columns except for the airline_sentiment which is the target variable in this case and the text column which actually contains the text of the tweet and is responsible for telling the sentiment of the tweet and also the airline column as it tells us which airline are the tweets talking about.

# Selecting only the required columns
selected_columns = ['airline_sentiment', 'text', 'airline']
tweets_data = tweets_data[selected_columns]

#%%
tweets_data.head()

#%%
# Tokenizing and Cleaning the actual text of Tweet
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Performing the process of Tokenization
    tweets_tokens = word_tokenize(text)
    
    # Cleaning the text of the tweet
    tweets_tokens = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in tweets_tokens]
    
    # Eliminating special characters, URLs, and mentions (or handles)
    tweets_tokens = [word for word in tweets_tokens if not re.match(r'^https?:\/\/.*[\r\n]*', word)]
    tweets_tokens = [word for word in tweets_tokens if word.isalnum()]
    
    # Converting text to lowercase
    tweets_tokens = [word.lower() for word in tweets_tokens]
    
    # Removing stopwords
    tweets_tokens = [word for word in tweets_tokens if word not in stop_words]
    
    # Performing the process of Lemmatization
    tweets_tokens = [lemmatizer.lemmatize(word) for word in tweets_tokens]
    
    return ' '.join(tweets_tokens) # Now after preprocessing the tweets_tokens, we join the tweets back

#%%
tweets_data['text'] = tweets_data['text'].apply(preprocess_text)

#%%
tweets_data.head()

#%%
# Encoding Categorical Variables
# Encoding the Airline column in place
airline_encoder = LabelEncoder()
tweets_data['airline'] = airline_encoder.fit_transform(tweets_data['airline'])

# Encoding the Sentiment column in place
sentiment_encoder = LabelEncoder()
tweets_data['airline_sentiment'] = sentiment_encoder.fit_transform(tweets_data['airline_sentiment'])

#%%
tweets_data.head()

#%%
# For vectorization we are gonna use Glove Vectorization(Global Vectors)
glove_path = 'glove.twitter.27B.200d.txt'
embeddings_index = {}
with open(glove_path, 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

#%%
embeddings_index

#%%
# Calculating the average length of tweets in your dataset
avg_length = tweets_data['text'].apply(lambda x: len(x.split())).mean()

print(avg_length) # for our dataset it comes out to be 10

#%%
# Setting max_sequence_length based on the average length 
max_sequence_length = 20

#%%
embedding_dim = 200

#%%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets_data['text'])
sequences = tokenizer.texts_to_sequences(tweets_data['text'])
X = pad_sequences(sequences, maxlen=max_sequence_length)

#%%
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#%%
embedding_matrix.shape
        
#%%
# Handling Imbalanced Classes (if any)
# Performing the Class Distribution Analysis
class_distribution = tweets_data['airline_sentiment'].value_counts()

# Handling Imbalanced Classes -> (Using class weights during model training)
class_weights = dict(1 / class_distribution)

#%%
# Visualizing Class Distribution of the dataset
plt.figure(figsize=(8, 5))
class_distribution.plot(kind='bar', color=['red', 'grey', 'green'])
plt.title('Sentiment Class Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
# We can see that negative sentiment tweets  are really higher here in the dataset

#%%
# Handling Duplicate Tweets
tweets_data.drop_duplicates(subset='text', inplace=True)
# Now we have dropped the duplicate values
#%%
# Finally after all the pre-processing, saving the preprocessed dataset
tweets_data.to_csv("preprocessed_airline_tweets.csv", index=False)

#%%
# Adding N-grams (bi(2)-grams and tri(3)-grams) for better context
ngram_vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)

# Fitting and transforming the text data
ngrams = ngram_vectorizer.fit_transform(tweets_data['text'])

# Convert the sparse matrix to a DataFrame
ngram_tweets_data = pd.DataFrame.sparse.from_spmatrix(ngrams, columns=ngram_vectorizer.get_feature_names_out())

ngram_tweets_data.head()

# %%
# Loading the preprocessed dataset
preprocessed_tweets_data = pd.read_csv("preprocessed_airline_tweets.csv")
preprocessed_tweets_data.head()

#%%
# Setting a random seed for reproducibility
random_seed = 42

# Randomly shuffling the dataset
preprocessed_tweets_data = preprocessed_tweets_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

#%%
# Commonly used split ratio: 80% for training, 10% for validation, and 10% for testing
train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1

# Splitting the dataset into training, validation, and testing sets
train_data, test_data = train_test_split(preprocessed_tweets_data, test_size=(val_ratio + test_ratio), random_state=random_seed)
val_data, test_data = train_test_split(test_data, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

#%%
# Validating Split Ratios by printing Subset Sizes
print(f"Training Set Size: {len(train_data)} samples ({train_ratio * 100}%)")
print(f"Validation Set Size: {len(val_data)} samples ({val_ratio * 100}%)")
print(f"Testing Set Size: {len(test_data)} samples ({test_ratio * 100}%)")

#%%
# Visualizing Class Distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='airline_sentiment', data=train_data, palette='viridis')
plt.title('Sentiment Class Distribution in Training Set')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#%%
# Storing the Split Datasets and saving the split datasets for reproducibility
train_data.to_csv("train_data.csv", index=False)
val_data.to_csv("val_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# %%
# defining a function to calculate and print evaluation metrics
def calculate_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

# Calculate confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

#%%
# Loading the split datasets
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# Extracting features (X) and labels (y)
X_train = train_data['text']
y_train = train_data['airline_sentiment']

X_val = val_data['text']
y_val = val_data['airline_sentiment']

X_test = test_data['text']
y_test = test_data['airline_sentiment']

#%%
# calling th etokenizer on training, validation and testing datasets to generate the respecitve sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_val = tokenizer.texts_to_sequences(X_val)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(sequences_train, maxlen=max_sequence_length)
X_val_pad = pad_sequences(sequences_val, maxlen=max_sequence_length)
X_test_pad = pad_sequences(sequences_test, maxlen=max_sequence_length)

#%%
def create_embedding_matrix(word_index, embedding_path, embedding_dim):
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
#%%
embedding_matrix = create_embedding_matrix(tokenizer.word_index, glove_path, embedding_dim)

# %%
# Defining various LSTM architectures

# For Single-layer Bidirectional LSTM architecture
model_bidirectional = Sequential()
model_bidirectional.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
                                  weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model_bidirectional.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model_bidirectional.add(Dense(3, activation='softmax'))
model_bidirectional.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#%%
# For Single-layer Unidirectional LSTM architecture
model_unidirectional = Sequential()
model_unidirectional.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
                                  weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model_unidirectional.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_unidirectional.add(Dense(3, activation='softmax'))
model_unidirectional.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#%%
# For Stacked Bidirectional LSTM architecture
model_stacked_bidirectional = Sequential()
model_stacked_bidirectional.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
                                          weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model_stacked_bidirectional.add(Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model_stacked_bidirectional.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
model_stacked_bidirectional.add(Dense(3, activation='softmax'))
model_stacked_bidirectional.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#%%
# For Stacked Unidirectional LSTM architecture
model_stacked_unidirectional = Sequential()
model_stacked_unidirectional.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
                                            weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model_stacked_unidirectional.add(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model_stacked_unidirectional.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model_stacked_unidirectional.add(Dense(3, activation='softmax'))
model_stacked_unidirectional.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#%%
# Now we move on to training the models
#%%
# Training the models
batch_size = 32
epochs = 10

#%%
# Single-layer Bidirectional LSTM
model_bidirectional.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

#%%
# Single-layer Unidirectional LSTM
model_unidirectional.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

#%%
# Stacked Bidirectional LSTM
model_stacked_bidirectional.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

#%%
# Stacked Unidirectional LSTM
model_stacked_unidirectional.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

#%%
# Evaluating the models on the test set
y_pred_bidirectional = model_bidirectional.predict(X_test_pad)
y_pred_unidirectional = model_unidirectional.predict(X_test_pad)
y_pred_stacked_bidirectional = model_stacked_bidirectional.predict(X_test_pad)
y_pred_stacked_unidirectional = model_stacked_unidirectional.predict(X_test_pad)

#%%
y_pred_bidirectional = np.argmax(y_pred_bidirectional, axis=1)

y_pred_unidirectional = np.argmax(y_pred_unidirectional, axis=1)

y_pred_stacked_bidirectional = np.argmax(y_pred_stacked_bidirectional, axis=1)

y_pred_stacked_unidirectional = np.argmax(y_pred_stacked_unidirectional, axis=1)

#%%
# Calculating the Evaluation metrics for the LSTM models
print("Evaluation Metrics for Single-layer Bidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_bidirectional)
plot_confusion_matrix(y_test, y_pred_bidirectional)
acc_lstm_single_bi = accuracy_score(y_test, y_pred_bidirectional)

#%%
print("\nEvaluation Metrics for Single-layer Unidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_unidirectional)
plot_confusion_matrix(y_test, y_pred_unidirectional)
acc_lstm_single_uni = accuracy_score(y_test, y_pred_unidirectional)

#%%
print("\nEvaluation Metrics for Stacked Bidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_stacked_bidirectional)
plot_confusion_matrix(y_test, y_pred_stacked_bidirectional)
acc_lstm_stacked_bi = accuracy_score(y_test, y_pred_stacked_bidirectional)

#%%
print("\nEvaluation Metrics for Stacked Unidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_stacked_unidirectional)
plot_confusion_matrix(y_test, y_pred_stacked_unidirectional)
acc_lstm_stacked_uni = accuracy_score(y_test, y_pred_stacked_unidirectional)

# %%
# Adding Attention Layer to the LSTM model
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim,
                            weights=[embedding_matrix], input_length=max_sequence_length, trainable=False)(input_layer)

#%%
# Stacked Bidirectional LSTM layers with attention mechanism
lstm_layer1 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding_layer)
lstm_layer2 = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(lstm_layer1)

# Attention mechanism
attention = Attention()([lstm_layer2, lstm_layer1])

# Concatenate attention output with the second LSTM layer output
attended_lstm = Concatenate(axis=-1)([lstm_layer2, attention])

# Global Max Pooling layer
pooled_layer = GlobalMaxPooling1D()(attended_lstm)

# Dense layer
output_layer = Dense(3, activation='softmax')(pooled_layer)

#%%
# Model
model_with_attention = Model(inputs=input_layer, outputs=output_layer)

#%%
# Compile the model
model_with_attention.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

#%%
# Print the model summary
model_with_attention.summary()

#%%
# Train the model
model_with_attention.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_pad, y_val))

#%%
# Evaluate the model
y_pred_with_attention = model_with_attention.predict(X_test_pad)
y_pred_with_attention = np.argmax(y_pred_with_attention, axis=1)

#%%
# Calculate evaluation metrics
calculate_evaluation_metrics(y_test, y_pred_with_attention)
plot_confusion_matrix(y_test, y_pred_with_attention)

#%%
# Load the preprocessed dataset
preprocessed_tweets_data = pd.read_csv("preprocessed_airline_tweets.csv")

# Extract features (X) and labels (y)
X = preprocessed_tweets_data['text']
y = preprocessed_tweets_data['airline_sentiment']

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# %%
# Tokenize and convert to input IDs
X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
X_val_tokens = tokenizer(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')
X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values)
y_val_tensor = torch.tensor(y_val.values)
y_test_tensor = torch.tensor(y_test.values)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], y_train_tensor)
val_dataset = TensorDataset(X_val_tokens['input_ids'], X_val_tokens['attention_mask'], y_val_tensor)
test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], y_test_tensor)

# %%
# Define batch size and create DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=5e-5)

# %%
# Training loop
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f'Training Loss: {avg_train_loss}')

# %%
# Validation loop
model.eval()
val_predictions, val_labels = [], []

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc='Validation', unit='batch'):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        outputs = model(**inputs)
        logits = outputs.logits
        val_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_labels.extend(inputs['labels'].cpu().numpy())

# Evaluate on validation set
print("\nValidation Set Metrics:")
calculate_evaluation_metrics(val_labels, val_predictions)
plot_confusion_matrix(val_labels, val_predictions)

# %%
# Test loop
model.eval()
test_predictions, test_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc='Testing', unit='batch'):
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}

        outputs = model(**inputs)
        logits = outputs.logits
        test_predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        test_labels.extend(inputs['labels'].cpu().numpy())

# Evaluate on test set
print("\nTest Set Metrics:")
calculate_evaluation_metrics(test_labels, test_predictions)
plot_confusion_matrix(test_labels, test_predictions)
    
# %%
# Load BERT tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# %%
# Tokenize and convert to input IDs for BERT
X_train_tokens_bert = tokenizer_bert(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
X_val_tokens_bert = tokenizer_bert(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')
X_test_tokens_bert = tokenizer_bert(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensors
y_train_tensor_bert = torch.tensor(y_train.values)
y_val_tensor_bert = torch.tensor(y_val.values)
y_test_tensor_bert = torch.tensor(y_test.values)

# Create TensorDatasets for BERT
train_dataset_bert = TensorDataset(X_train_tokens_bert['input_ids'], X_train_tokens_bert['attention_mask'], y_train_tensor_bert)
val_dataset_bert = TensorDataset(X_val_tokens_bert['input_ids'], X_val_tokens_bert['attention_mask'], y_val_tensor_bert)
test_dataset_bert = TensorDataset(X_test_tokens_bert['input_ids'], X_test_tokens_bert['attention_mask'], y_test_tensor_bert)

# Define batch size for BERT
batch_size_bert = 32
train_dataloader_bert = DataLoader(train_dataset_bert, sampler=RandomSampler(train_dataset_bert), batch_size=batch_size_bert)
val_dataloader_bert = DataLoader(val_dataset_bert, sampler=SequentialSampler(val_dataset_bert), batch_size=batch_size_bert)
test_dataloader_bert = DataLoader(test_dataset_bert, sampler=SequentialSampler(test_dataset_bert), batch_size=batch_size_bert)

#%%
# Load BERT model
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer_bert = AdamW(model_bert.parameters(), lr=5e-5)

#%%
# Defining hyperparameters for tuning
bert_hyperparameters = {'learning_rate': [5e-5, 3e-5, 1e-5, 1e-4],
                        'epochs': [3, 4],
                        'batch_size': [16, 32]}
#%%
# Performing hyperparameter tuning using grid search
best_bert_accuracy = 0.0
best_bert_params = None

#%%

for lr in bert_hyperparameters['learning_rate']:
    for epochs in bert_hyperparameters['epochs']:
        for batch_size in bert_hyperparameters['batch_size']:
            optimizer_bert = AdamW(model_bert.parameters(), lr=lr)
            
            # Training loop for BERT
            for epoch in range(epochs):
                model_bert.train()
                total_loss_bert = 0
                for batch in tqdm(train_dataloader_bert, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch'):
                    inputs_bert = {'input_ids': batch[0],
                                   'attention_mask': batch[1],
                                   'labels': batch[2]}

                    optimizer_bert.zero_grad()
                    outputs_bert = model_bert(**inputs_bert)
                    loss_bert = outputs_bert.loss
                    total_loss_bert += loss_bert.item()
                    loss_bert.backward()
                    optimizer_bert.step()

                avg_train_loss_bert = total_loss_bert / len(train_dataloader_bert)
                print(f'Training Loss (BERT): {avg_train_loss_bert}')

            # Validation loop for BERT
            model_bert.eval()
            val_predictions_bert, val_labels_bert = [], []

            with torch.no_grad():
                for batch in tqdm(val_dataloader_bert, desc='Validation (BERT)', unit='batch'):
                    inputs_bert = {'input_ids': batch[0],
                                   'attention_mask': batch[1],
                                   'labels': batch[2]}

                    outputs_bert = model_bert(**inputs_bert)
                    logits_bert = outputs_bert.logits
                    val_predictions_bert.extend(torch.argmax(logits_bert, dim=1).cpu().numpy())
                    val_labels_bert.extend(inputs_bert['labels'].cpu().numpy())

            # Evaluate on validation set for BERT
            val_accuracy_bert = accuracy_score(val_labels_bert, val_predictions_bert)
            print(f"\nValidation Accuracy (BERT): {val_accuracy_bert}")

            # Update best parameters if current configuration achieves higher accuracy
            if val_accuracy_bert > best_bert_accuracy:
                best_bert_accuracy = val_accuracy_bert
                best_bert_params = {'learning_rate': lr, 'epochs': epochs, 'batch_size': batch_size}

#%%
# Displaying the best hyperparameters for BERT
print("\nBest BERT Hyperparameters:", best_bert_params)

#%%
# Finally Now we will be training BERT model with best hyperparameters
best_optimizer_bert = AdamW(model_bert.parameters(), lr=best_bert_params['learning_rate'])
for epoch in range(best_bert_params["epochs"]):
    model_bert.train()
    total_loss_bert = 0
    for batch in tqdm(train_dataloader_bert, desc=f'Epoch {epoch + 1}/{best_bert_params["epochs"]}', unit='batch'):
        inputs_bert = {'input_ids': batch[0],
                       'attention_mask': batch[1],
                       'labels': batch[2]}

        best_optimizer_bert.zero_grad()
        outputs_bert = model_bert(**inputs_bert)
        loss_bert = outputs_bert.loss
        total_loss_bert += loss_bert.item()
        loss_bert.backward()
        best_optimizer_bert.step()

    avg_train_loss_bert = total_loss_bert / len(train_dataloader_bert)
    print(f'Training Loss (Best BERT): {avg_train_loss_bert}')

#%%
# Evaluate the best BERT model on the test set
model_bert.eval()
test_predictions_bert, test_labels_bert = [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader_bert, desc='Testing (Best BERT)', unit='batch'):
        inputs_bert = {'input_ids': batch[0],
                       'attention_mask': batch[1],
                       'labels': batch[2]}

        outputs_bert = model_bert(**inputs_bert)
        logits_bert = outputs_bert.logits
        test_predictions_bert.extend(torch.argmax(logits_bert, dim=1).cpu().numpy())
        test_labels_bert.extend(inputs_bert['labels'].cpu().numpy())

# Evaluate on test set for the best BERT model
print("\nTest Set Metrics (Best BERT):")
calculate_evaluation_metrics(test_labels_bert, test_predictions_bert)
plot_confusion_matrix(test_labels_bert, test_predictions_bert)

# %%
# Load RoBERTa tokenizer
tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)

#%%
# Tokenize and convert to input IDs for RoBERTa
X_train_tokens_roberta = tokenizer_roberta(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
X_val_tokens_roberta = tokenizer_roberta(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')
X_test_tokens_roberta = tokenizer_roberta(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')

# Convert labels to PyTorch tensors
y_train_tensor_roberta = torch.tensor(y_train.values)
y_val_tensor_roberta = torch.tensor(y_val.values)
y_test_tensor_roberta = torch.tensor(y_test.values)

# Create TensorDatasets for RoBERTa
train_dataset_roberta = TensorDataset(X_train_tokens_roberta['input_ids'], X_train_tokens_roberta['attention_mask'], y_train_tensor_roberta)
val_dataset_roberta = TensorDataset(X_val_tokens_roberta['input_ids'], X_val_tokens_roberta['attention_mask'], y_val_tensor_roberta)
test_dataset_roberta = TensorDataset(X_test_tokens_roberta['input_ids'], X_test_tokens_roberta['attention_mask'], y_test_tensor_roberta)

# Define batch size for RoBERTa
batch_size_roberta = 32
train_dataloader_roberta = DataLoader(train_dataset_roberta, sampler=RandomSampler(train_dataset_roberta), batch_size=batch_size_roberta)
val_dataloader_roberta = DataLoader(val_dataset_roberta, sampler=SequentialSampler(val_dataset_roberta), batch_size=batch_size_roberta)
test_dataloader_roberta = DataLoader(test_dataset_roberta, sampler=SequentialSampler(test_dataset_roberta), batch_size=batch_size_roberta)

#%%
# Load RoBERTa model
model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
optimizer_roberta = AdamW(model_roberta.parameters(), lr=5e-5)

# Training loop for RoBERTa
epochs_roberta = 3
device_roberta = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_roberta.to(device_roberta)

for epoch in range(epochs_roberta):
    model_roberta.train()
    total_loss_roberta = 0
    for batch in tqdm(train_dataloader_roberta, desc=f'Epoch {epoch + 1}/{epochs_roberta}', unit='batch'):
        inputs_roberta = {'input_ids': batch[0].to(device_roberta),
                          'attention_mask': batch[1].to(device_roberta),
                          'labels': batch[2].to(device_roberta)}

        optimizer_roberta.zero_grad()
        outputs_roberta = model_roberta(**inputs_roberta)
        loss_roberta = outputs_roberta.loss
        total_loss_roberta += loss_roberta.item()
        loss_roberta.backward()
        optimizer_roberta.step()

    avg_train_loss_roberta = total_loss_roberta / len(train_dataloader_roberta)
    print(f'Training Loss (RoBERTa): {avg_train_loss_roberta}')
    
#%%
# Validation loop for RoBERTa
model_roberta.eval()
val_predictions_roberta, val_labels_roberta = [], []

with torch.no_grad():
    for batch in tqdm(val_dataloader_roberta, desc='Validation (RoBERTa)', unit='batch'):
        inputs_roberta = {'input_ids': batch[0].to(device_roberta),
                          'attention_mask': batch[1].to(device_roberta),
                          'labels': batch[2].to(device_roberta)}

        outputs_roberta = model_roberta(**inputs_roberta)
        logits_roberta = outputs_roberta.logits
        val_predictions_roberta.extend(torch.argmax(logits_roberta, dim=1).cpu().numpy())
        val_labels_roberta.extend(inputs_roberta['labels'].cpu().numpy())

# Evaluate on validation set for RoBERTa
print("\nValidation Set Metrics (RoBERTa):")
calculate_evaluation_metrics(val_labels_roberta, val_predictions_roberta)
plot_confusion_matrix(val_labels_roberta, val_predictions_roberta)

#%%
# Test loop for RoBERTa
model_roberta.eval()
test_predictions_roberta, test_labels_roberta = [], []

with torch.no_grad():
    for batch in tqdm(test_dataloader_roberta, desc='Testing (RoBERTa)', unit='batch'):
        inputs_roberta = {'input_ids': batch[0].to(device_roberta),
                          'attention_mask': batch[1].to(device_roberta),
                          'labels': batch[2].to(device_roberta)}

        outputs_roberta = model_roberta(**inputs_roberta)
        logits_roberta = outputs_roberta.logits
        test_predictions_roberta.extend(torch.argmax(logits_roberta, dim=1).cpu().numpy())
        test_labels_roberta.extend(inputs_roberta['labels'].cpu().numpy())

# Evaluate on test set for RoBERTa
print("\nTest Set Metrics (RoBERTa):")
calculate_evaluation_metrics(test_labels_roberta, test_predictions_roberta)
plot_confusion_matrix(test_labels_roberta, test_predictions_roberta)
# %%
# Evaluating and comparing the models that we used
print("\nEvaluation Metrics for Single-layer Bidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_bidirectional)

print("\nEvaluation Metrics for Single-layer Unidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_unidirectional)

print("\nEvaluation Metrics for Stacked Bidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_stacked_bidirectional)

print("\nEvaluation Metrics for Stacked Unidirectional LSTM:")
calculate_evaluation_metrics(y_test, y_pred_stacked_unidirectional)

# Evaluate and compare BERT models
print("\nEvaluation Metrics for BERT:")
calculate_evaluation_metrics(test_labels_bert, test_predictions_bert)
plot_confusion_matrix(test_labels_bert, test_predictions_bert)

# Evaluate and compare RoBERTa models
print("\nEvaluation Metrics for RoBERTa:")
calculate_evaluation_metrics(test_labels_roberta, test_predictions_roberta)
plot_confusion_matrix(test_labels_roberta, test_predictions_roberta)
# %%
# Performing Statistical Analysis ANOVA test on the LSTM models
data = {
    'Model_LSTM_single_bi': [acc_lstm_single_bi],
    'Model_LSTM_single_uni': [acc_lstm_single_uni],
    'Model_LSTM_stacked_bi': [acc_lstm_stacked_bi],
    'Model_LSTM_stacked_uni': [acc_lstm_stacked_uni]
}

df = pd.DataFrame(data)

# Performinng ANOVA
statistic, p_value = f_oneway(
    df['Model_LSTM_single_bi'],
    df['Model_LSTM_single_uni'],
    df['Model_LSTM_stacked_bi'],
    df['Model_LSTM_stacked_uni']
)

# Check the p-value
if p_value < 0.05:
    print(f"ANOVA: There are significant differences between at least two groups (p-value: {p_value:.4f})")
else:
    print(f"ANOVA: No significant differences between groups (p-value: {p_value:.4f})")