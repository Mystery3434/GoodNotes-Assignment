import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout


def remove_unimportant_features(data):
  if 'delay_time' in data.columns:
    data = data.drop(['Departure', 'flight_id', 'flight_date', 'delay_time', 'flight_no'], axis=1)
  else:
    data = data.drop(['Departure', 'flight_id', 'flight_date', 'flight_no'], axis=1)
  return data


def standardize_data(data):
  data['std_hour'] = data['std_hour'] / 23
  data['Week'] = (data['Week'] - 1)/(52-1)
  return data


def split_input_output(data):
  output = data['is_claim']
  input = data.drop('is_claim', axis = 1)
  return input, output


def add_one_hots(X):
  X = pd.get_dummies(X, columns=["Arrival", "Airline"])
  return X


def build_neural_network_regressor(train_X, summary = True):
  model = Sequential()
  model.add(Dense(128, input_dim = train_X.shape[1], activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam', metrics=['mse'])
  if summary:
    print(model.summary())
  return model

def train_model(model, train_X, train_Y, epochs = 50, batch_size = 512):
  history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epochs, batch_size=batch_size)
  model.save('./regression_model/')
  return model, history


def evaluate_model(model, test_X, test_Y, display = True, classification = False, exact = False):

    # classification: Boolean to indicate whether the model is a classifier or a regressor
    # exact: If true, the classifier outputs exact values 0 or 800, otherwise, it multiplies the sigmoid output by 800.


  if not classification:
    predictions = model.predict(test_X)
  else:
    if not exact:
      predictions = 800 * model.predict(test_X)
    else:
      predictions = model.predict(test_X)
      predictions = np.round(predictions)
      predictions *= 800

  predictions = np.clip(predictions, 0, 800)
  mae = mean_absolute_error(test_Y, predictions)
  mse = mean_squared_error(test_Y, predictions)


  if display:
    print("MAE: {}\nMSE:{}".format(mae, mse))

  return mae, mse


def plot_losses(history, epochs=50):
  plt.plot([i for i in range(1, epochs+1)], history.history['loss'], label="Training Loss")
  plt.plot([i for i in range(1, epochs+1)], history.history['val_loss'], label="Validation Loss")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend(loc="upper right")
  plt.show()


def build_neural_network_classifier(train_X, summary=True):
    model = Sequential()
    model.add(Dense(128, input_dim=train_X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

    if summary:
        print(model.summary())
    return model


def build_complex_neural_network_classifier(train_X, summary=True):
    model = Sequential()
    model.add(Dense(128, input_dim=train_X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, input_dim=train_X.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse', 'accuracy'])

    if summary:
        print(model.summary())
    return model


def train_model_classifier(model, train_X, train_Y, epochs = 50, batch_size = 512):
  history = model.fit(train_X, train_Y, validation_split=0.1, epochs=epochs, batch_size=batch_size)
  model.save('/content/drive/My Drive/GoodNotes Test/model_classifier_2/')
  return model, history


if __name__== "__main__":
    data = pd.read_csv('./flight_delays_data.csv')
    data = remove_unimportant_features(data)
    data = standardize_data(data)
    X, Y = split_input_output(data)
    X = add_one_hots(X)

    model = build_complex_neural_network_classifier(X)
    # When training the model, the last 10% of the data is not used for training and is only used for validation.
    model, history = train_model_classifier(model, X, Y / 800, epochs=100,
                                                                  batch_size=64)
    validation_X, validation_Y =  X[-89000:], Y[-89000:] # Roughly the last 10% of the data, which has not been used for training.
    evaluate_model(model, validation_X, validation_Y, display=True, classification=True)
    plot_losses(history, 500)