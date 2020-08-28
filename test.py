from train import *
from keras.models import load_model
import sys

def naive_individual_prediction(x):
  if x=="Cancelled" or float(x) <= 3:
    return 0
  else:
    return 800


def naive_prediction(data):
  output = data['delay_time'].apply(naive_individual_prediction)
  return output


def process_test_set(train, test):
  missing_cols = set( train.columns ) - set( test.columns )
  for c in missing_cols:
      test[c] = 0
  test = test[train.columns]
  return test

if __name__ == "__main__":
    classifier_model = load_model('classifier_model.h5')
    data = pd.read_csv('./flight_delays_data.csv')
    testing_data_location = sys.argv[1]
    testing_data = pd.read_csv(testing_data_location)

    if 'delay_time' in testing_data.columns:
        y = naive_prediction(testing_data)
        y.to_csv("output.csv", index=False, header=False)
        print("Predictions saved in output.csv.")

    else:
        data = remove_unimportant_features(data)
        data = standardize_data(data)
        X, Y = split_input_output(data)
        X = add_one_hots(X)

        testing_data = remove_unimportant_features(testing_data)
        testing_data = add_one_hots(testing_data)
        testing_data = process_test_set(X, testing_data)
        testing_data = standardize_data(testing_data)

        predictions = classifier_model.predict(testing_data)
        predictions *= 800
        predictions = np.clip(predictions, 0, 800)
        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv("output.csv", index=False, header=False)
        print("Predictions saved in output.csv.")
