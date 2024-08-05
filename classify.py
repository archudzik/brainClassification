import os
import pickle
import argparse
import pandas as pd
import numpy as np


def load_model(filename):
    path = os.path.join('model', filename)
    with open(path, 'rb') as file:
        return pickle.load(file)


def evaluate_model(model_data, row_df, use_larger):
    if use_larger:
        model = model_data['large_model']
    else:
        model = model_data['small_model']

    preprocessor = model_data['preprocessor']
    label_encoder = model_data['label_encoder']
    features_all = model_data['features_all']
    features_selected = model_data['features_selected']

    # Prepare the input for the model
    X_preprocessed = preprocessor.transform(row_df)

    # Select only feature_names from the preprocessed row
    X_preprocessed = pd.DataFrame(X_preprocessed, columns=features_all)
    X_preprocessed_selected_features = X_preprocessed[features_selected]

    # Predict probabilities
    if use_larger:
        probs = model.predict_proba(X_preprocessed_selected_features.values)[0]
        prediction = model.predict(X_preprocessed_selected_features.values)[0]
    else:
        probs = model.predict_proba(X_preprocessed_selected_features)[0]
        prediction = model.predict(X_preprocessed_selected_features)[0]

    predicted_label = label_encoder.inverse_transform([prediction])[0]

    return {
        'model': model_data['name'],
        'predicted_label': predicted_label,
        'probabilities': probs,
        'confidence': np.max(probs)
    }


def main():
    # Parse the CSV file path argument
    parser = argparse.ArgumentParser(
        description='Evaluate models with input features from CSV file.')
    parser.add_argument('--csv_file', type=str,
                        help='Path to the CSV file containing input features')
    parser.add_argument('--use_larger', type=bool, default=True, required=False,
                        help='Use larger model (trained including validation set)')
    args = parser.parse_args()
    use_larger = args.use_larger

    # Define confidence thresholds based on screening standards
    HIGH_CONFIDENCE_THRESHOLD = 0.95
    MODERATE_CONFIDENCE_THRESHOLD = 0.80
    LOW_CONFIDENCE_THRESHOLD = 0.60

    # Load models
    model_filenames = {
        'CONTROL_PARKINSON': 'model_CONTROL_PARKINSON-LogisticRegression.pkl',
        'CONTROL_PRODROMAL': 'model_CONTROL_PRODROMAL-LogisticRegression.pkl',
        'PARKINSON_PRODROMAL': 'model_PARKINSON_PRODROMAL-LogisticRegression.pkl'
    }

    models = {name: load_model(filename)
              for name, filename in model_filenames.items()}
    for name, model in models.items():
        model['name'] = name
    print("[ok] Models loaded successfully.")

    # Load the CSV file
    csv_file = args.csv_file
    data = pd.read_csv(csv_file, sep=';')
    print(f"[ok] CSV file {csv_file} loaded successfully.")

    # Evaluate models for each row in the CSV file
    for index, row in data.iterrows():
        row_df = pd.DataFrame([row])
        row_df = row_df.drop(columns=['Group'])

        try:
            print(f"[..] Running scan #{index}")
            control_parkinson_result = evaluate_model(
                models['CONTROL_PARKINSON'], row_df, use_larger)
            control_prodromal_result = evaluate_model(
                models['CONTROL_PRODROMAL'], row_df, use_larger)
            parkinson_prodromal_result = evaluate_model(
                models['PARKINSON_PRODROMAL'], row_df, use_larger)

            aggregated_probabilities = {
                "CONTROL": control_parkinson_result['probabilities'][0] + control_prodromal_result['probabilities'][0],
                "PRODROMAL": control_prodromal_result['probabilities'][1] + parkinson_prodromal_result['probabilities'][1],
                "PARKINSON": control_parkinson_result['probabilities'][1] + parkinson_prodromal_result['probabilities'][0],
            }

            # Determine the class with the highest summed probability
            final_prediction = max(aggregated_probabilities,
                                   key=aggregated_probabilities.get)
            final_confidence = aggregated_probabilities[final_prediction] / 2

            # Determine the confidence marker and level
            if final_confidence > HIGH_CONFIDENCE_THRESHOLD:
                marker = ''
                confidence_level = 'high'
            elif final_confidence > MODERATE_CONFIDENCE_THRESHOLD:
                marker = '?'
                confidence_level = 'moderate'
            elif final_confidence > LOW_CONFIDENCE_THRESHOLD:
                marker = '??'
                confidence_level = 'low'
            else:
                marker = '???'
                confidence_level = 'very low'

            # Print the final prediction and confidence level
            print(
                f"[>>] Prediction for scan #{index} is {final_prediction}{marker} with {confidence_level} confidence ({final_confidence:.2f})")

        except ValueError as e:
            print(f"[!!] Error evaluating models for row {index}: {e}")

    print("[ok] Processing is finished.")


if __name__ == '__main__':
    main()
