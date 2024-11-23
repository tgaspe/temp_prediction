import pandas as pd
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from backend import fasta_to_csv, features_from_sequence, predict
import os
import matplotlib.colors as mcolors

# *********** Streamlit App ***********

st.title("Protein Temperature Prediction")

# Step 1: Upload FASTA file or paste a protein sequence
st.header("Step 1: Provide Protein Sequence(s)")
uploaded_file = st.file_uploader("Upload your FASTA file", type=["fasta", "fa", "faa"])
pasted_sequence = st.text_area(
    "Or paste your protein sequence (use FASTA format with '>ID' followed by the sequence on a new line):"
)

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_fasta:
        temp_fasta.write(uploaded_file.read())
        temp_fasta_path = temp_fasta.name

    # Convert FASTA to CSV
    fasta_csv_path = temp_fasta_path.replace(".fasta", ".csv")
    fasta_to_csv(temp_fasta_path, fasta_csv_path)
    st.success("FASTA file successfully converted to CSV.")
elif pasted_sequence:
    # Save the pasted sequence as a temporary FASTA file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_fasta:
        temp_fasta.write(pasted_sequence.encode('utf-8'))
        temp_fasta_path = temp_fasta.name

    # Convert the temporary FASTA to CSV
    fasta_csv_path = temp_fasta_path.replace(".fasta", ".csv")
    fasta_to_csv(temp_fasta_path, fasta_csv_path)
    st.success("Pasted sequence successfully converted to CSV.")
else:
    st.warning("Please upload a FASTA file or paste a protein sequence.")
    fasta_csv_path = None

# Process the CSV file if available
if fasta_csv_path:
    # Display CSV preview
    st.subheader("Converted CSV Preview")
    csv_preview = pd.read_csv(fasta_csv_path)
    st.dataframe(csv_preview)

    # Step 2: Generate Features
    st.header("Step 2: Generate Features")
    output_features_path = fasta_csv_path.replace(".csv", "_features.csv")
    features_df = features_from_sequence(fasta_csv_path, output_features_path)
    st.success("Features successfully generated.")
    st.subheader("Generated Features Preview")
    st.dataframe(features_df)

    # Step 3: Predict Temperature
    st.header("Step 3: Predict Temperature")
    if st.button("Run Prediction"):
        predictions = predict(features_df.drop(columns=["ID"]))
        features_df["Predicted Temperature"] = predictions
        features_df = features_df[['ID', 'Predicted Temperature']]
        st.success("Predictions generated successfully.")
        st.subheader("Predictions")
        st.dataframe(features_df)

        # Allow users to download the predictions
        st.download_button(
            label="Download Predictions",
            data=features_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Step 4: Plot a bar chart of predicted temperatures with color based on value
        st.header("Step 4: Bar Plot of Predicted Temperatures")
        plot_data = features_df[['ID', 'Predicted Temperature']]

        # Normalize the values for coloring
        norm = mcolors.Normalize(vmin=plot_data['Predicted Temperature'].min(), vmax=plot_data['Predicted Temperature'].max())
        cmap = plt.cm.get_cmap('coolwarm')

        # Create a color list based on the normalized temperatures
        colors = [cmap(norm(value)) for value in plot_data['Predicted Temperature']]

        plt.rcParams['font.family'] = 'monospace'
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(plot_data['ID'], plot_data['Predicted Temperature'], color=colors)

        # Add annotations (values) to each bar
        for bar in bars:
            height = bar.get_height()  # Get the height of the bar (which is the value)
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

        # Labels and title
        plt.xlabel('Protein ID', fontweight='bold', fontsize=14, labelpad=15)
        plt.ylabel('Predicted Temperature', fontweight='bold', fontsize=14, labelpad=15)
        plt.title('Predicted Temperature for Each Protein', fontweight='bold', fontsize=24, pad=20)

        # Rotate x-axis labels for better visibility
        plt.xticks(rotation=90)

        # Display the plot in Streamlit
        st.pyplot(plt)  # Display the plot in Streamlit


