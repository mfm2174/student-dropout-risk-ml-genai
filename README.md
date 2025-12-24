# student-dropout-risk-ml-genai
Student Dropout Risk – ML + GenAI Pipeline

This project implements an end-to-end machine learning pipeline to predict and explain student dropout risk using Classic ML, Deep Learning, and Generative AI with RAG.

1. Project Overview

The pipeline combines:

-Classic Machine Learning (Logistic Regression with hyperparameter optimization)

-Deep Learning (Neural Network with Keras/TensorFlow)

-RAG (Retrieval-Augmented Generation) using sentence embeddings

-LLMs (OpenAI GPT models) to generate human-readable explanations of predictions

* The goal is not only to predict dropout risk, but also to explain the reasons and suggest institutional actions in accessible language *

2. Pipeline Architecture

-Data Loading

-Reads student data from a CSV file

-Numeric features + binary target (0 = no dropout, 1 = dropout)

-Classic ML

-Logistic Regression

-Feature scaling with StandardScaler

-Hyperparameter tuning using GridSearchCV

- Evaluation with F1-score

- Deep Learning

- Dense neural network (Keras/TensorFlow)

- Binary classification with sigmoid output

- RAG (Context Retrieval)

- Sentence embeddings using sentence-transformers

- Semantic search over institutional knowledge snippets

- LLM Explanation

- Combines model predictions + retrieved context

- Generates natural language explanations via OpenAI GPT models

3. Technologies Used

Python, Pandas, NumPy, Scikit-learn, Keras / TensorFlow, Sentence-Transformers, OpenAI API, Google Colab + Google Drive

4. Expected Data Format

The input CSV file must contain:

- Multiple numeric feature columns

- A binary target column named target

Example:

age,absences,grade_average,engagement_score,target
20,12,6.5,0.3,1

▶️ How to Run

- Upload the dataset to your Google Drive

- Update the CSV path in the script:

- PATH = "/content/drive/MyDrive/Colab Notebooks/GenAI/students.csv"

- Set your OpenAI API key as an environment variable

- Run the script

📌 Output

- Model performance metrics (Classic ML and Deep Learning)

 - Generated RAG-based prompt

- LLM explanation describing:

- Why the student is at risk

Possible institutional interventions

⚠️ Important Notes

- Do not hardcode API keys in production environments

- This project is educational and demonstrative, not intended for automated decision-making

- Predictions should always be reviewed by education professionals

📜 License

This project is for educational and research purposes.
