import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO  # For handling FASTA files
import sys
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Add the path to the external directory
external_libs_path = os.path.abspath("/data/leuven/359/vsc35907/ibp-optimal-temperature-dev/scripts/generate_features")  # Adjust the path as needed
sys.path.append(external_libs_path)

from BERT_extraction import ProteinEmbeddingGenerator  # Replace with the actual file name (without .py)


# *********** Methods ***********


# Function to run the model on input
def predict(input_df):
    """
    Predicts the output based on the input DataFrame.
    Uses a classifier to determine the class and then predicts with the respective model.
    
    Parameters:
        input_df (pd.DataFrame): Input features for prediction.
        
    Returns:
        np.ndarray: Predicted values as a vector.
    """

    # Load the trained model
    model_classifier = joblib.load('./models/classifier.pkl')
    model_cold = joblib.load('./models/cold_regressor.pkl')
    model_middle = joblib.load('./models/mid_regressor.pkl')
    model_hot = joblib.load('./models/hot_regressor.pkl')
    
    # Convert input to numpy array (if not already) and reshape for prediction
    y_class = model_classifier.predict(input_df)  # Class predictions (vector)

    # Initialize y_pred as an empty list to store predictions
    y_pred = []

    # Iterate over each prediction and input row
    for idx, class_label in enumerate(y_class):
        # Select the corresponding row of input_df
        input_row = input_df.iloc[idx:idx+1]

        # Predict using the appropriate model based on the class
        if class_label == 0: # "cold"
            y_pred.append(model_cold.predict(input_row)[0])
        elif class_label == 1: # "hot" 
            y_pred.append(model_hot.predict(input_row)[0])
        else:  # middle
            y_pred.append(model_middle.predict(input_row)[0])

    # Convert y_pred to a numpy array for consistency
    y_pred = np.array(y_pred)

    return y_pred

def fasta_to_csv(fasta_file, csv_file):
    """
    Converts a FASTA file to a CSV file with columns for sequence ID and sequence.
    
    Parameters:
        fasta_file (str): Path to the input FASTA file.
        csv_file (str): Path to the output CSV file.
    """
    try:
        # Open the CSV file for writing
        with open(csv_file, 'w', newline='') as csv_out:
            writer = csv.writer(csv_out)
            writer.writerow(['ID', 'Sequence'])  # Write header
            
            # Parse the FASTA file using SeqIO
            for record in SeqIO.parse(fasta_file, "fasta"):
                writer.writerow([record.id, str(record.seq)])  # Write ID and sequence

        print(f"FASTA file converted successfully to {csv_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

def features_from_sequence(fasta_csv, output_file):

    # Generating features
    generator = ProteinEmbeddingGenerator()
    
    generator.generate_embeddings(
        input_file=fasta_csv,
        output_file=output_file,
        sequence_col='Sequence',
        id_col='ID'
    )

    # PCA 
    # Load the protlearn_features.csv file (the main feature file)
    df = pd.read_csv(output_file)

    # Separate features (X) and response variable (y)
    id = df['ID']
    X = df.drop(columns=['ID'])

    # Load Scaler and PCA
    scaler_loaded = joblib.load('./models/BERT_scaler.pkl')
    pca_loaded = joblib.load('./models/BERT_pca_model.pkl')

    # Apply Scaler transformation
    X = scaler_loaded.transform(X)

    # Apply the PCA transformation
    X_pca = pca_loaded.transform(X)
 
    # Convert the transformed data to a pandas DataFrame
    X_selected = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    # join response and features
    new_df = pd.concat([id, X_selected], axis=1)

    return new_df
