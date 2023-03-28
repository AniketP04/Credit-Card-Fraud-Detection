#importing lib
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout
from sklearn.preprocessing import MinMaxScaler                                    
from tensorflow.keras import activations
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from keras import regularizers
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split



df=pd.read_csv('/content/creditcard.csv')

df = df.drop(['Time'], axis=1)
df.dropna(inplace=True)
y=df['Class']

#splitting data into train and test 
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.33, random_state = 1)

#scaling
t = MinMaxScaler()
t.fit(X_train)
X_train = t.transform(X_train)
X_test = t.transform(X_test)

#Autoencoder

#Encoder
input_layer = Input(shape=(X_train.shape[1] ))
encoder = Dense(20, activation="relu")(input_layer)
encoder = Dense(15, activation="relu")(encoder)
#encoder = Dense(10, activation="relu")(encoder)


#Decoder
#decoder = Dense(15, activation='relu')(encoder)
decoder = Dense(20, activation="relu")(encoder)

decoder = Dense(X_train.shape[1], activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

#Encoder model
Encoder = Model(inputs=input_layer, outputs=encoder)

optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
autoencoder.compile(metrics=['accuracy'],
                    loss='mean_squared_error',
                    optimizer=optimizer)
history = autoencoder.fit(X_train, X_train,
                    epochs=25,
                    batch_size=32,
                     validation_data = (X_test,X_test)).history

#predicting train and test data on encoder
X_train_encode = Encoder.predict(X_train)
X_test_encode = Encoder.predict(X_test)

#KMean for classification
kmeans = KMeans(n_clusters=2)
clustered_training_set = kmeans.fit(X_train_encode, y_train)
yhat = kmeans.predict(X_test_encode)
acc = accuracy_score(y_test, yhat)
print(acc)