# plagiarism_detection
📝 Plagiarism Detection System (NLP Mini Project)
Project Overview

This is a NLP-based Plagiarism Detection System that identifies whether a given text is original or plagiarized.

The system uses:

TF-IDF for text feature extraction

Logistic Regression as a machine learning classifier

Evaluation metrics including accuracy, confusion matrix, and classification report

It’s designed to be run in Jupyter Notebook or VS Code.

Features

Detects plagiarized content from a text dataset

Preprocessing: lowercasing, stopword removal, tokenization

Visualizes results with confusion matrix heatmaps

Easy to extend with larger datasets or advanced NLP models (e.g., BERT)

Dataset

The dataset plagiarism_dataset.csv contains 50 text samples:

label = 0 → Original text

label = 1 → Plagiarized text

Example:

text,label
"Machine learning allows computers to learn patterns from data.",0
"This document is copied word for word from an existing article.",1

Installation & Setup

Clone the repository

git clone https://github.com/<YOUR-USERNAME>/plagiarism-detection.git
cd plagiarism-detection


Create a virtual environment

python -m venv env


Activate the environment

# Windows
env\Scripts\activate
# Mac/Linux
source env/bin/activate


Install dependencies

pip install -r requirements.txt


Run Jupyter Notebook

jupyter notebook


Open notebook.ipynb and run all cells.

Project Structure
plagiarism-detection/
│── plagiarism_dataset.csv   # Dataset (50 rows)
│── notebook.ipynb           # Jupyter notebook with code
│── requirements.txt         # Python dependencies
│── README.md                # GitHub documentation
│── .gitignore               # Ignore venv and cache files

Evaluation

The system outputs:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

Example:

Accuracy: 0.92
Confusion Matrix:
[[5 0]
 [1 4]]

Future Improvements

Use a larger real-world dataset

Detect paraphrased content with embeddings (Word2Vec, BERT)

Build a GUI or Web App (Tkinter / Streamlit)

Deploy as an API (Flask / FastAPI)

License

This project is licensed under the MIT License – free to use, modify, and distribute with attribution.
