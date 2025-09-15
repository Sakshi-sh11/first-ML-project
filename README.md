# Email Spam Detection System (Python)

## Project Overview
This project implements an **Email Spam Detection system** using machine learning in Python.  
The system classifies SMS messages as **spam** or **ham (not spam)** using natural language processing (NLP) techniques and supervised learning algorithms.

---

## Dataset
- Dataset used: **SMS Spam Collection Dataset** from Kaggle  
- Number of rows: 5572  
- Columns: `Category` (spam/ham) and `Message` (text message)  
- Public dataset link: [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code)

---

## Data Preprocessing
1. **Label Encoding:**  
   - `ham` → 1  
   - `spam` → 0  
2. **Text Vectorization and TF-IDF:**  
   - Converts text into numerical features, reweighting words based on importance.  
3. **Data Balancing:**  
   - Techniques like **undersampling, oversampling, or SMOTE** are applied to balance the classes and prevent bias toward the majority class.  
4. **Train-Test Split:**  
   - 80% of data used for training, 20% for testing.

---

## Feature Extraction
- Extracts numerical features from raw text data to improve model learning.  
- Helps the model understand patterns in messages for better prediction accuracy.

---

## Model Training and Evaluation
- **Training:** Machine learning models are trained on the processed training dataset.  
- **Testing:** Models are evaluated on the test set to measure predictive performance.  
- **Performance Metrics:**  
  - **Accuracy:** Overall percentage of correctly classified messages.  
  - **F1 Score:** Harmonic mean of precision and recall.  
  - **Recall:** Proportion of actual spam messages correctly identified.  

---

## Skills and Techniques Used
- Python Programming (pandas, scikit-learn)  
- Natural Language Processing (NLP)  
- TF-IDF Vectorization  
- Data Preprocessing and Label Encoding  
- Data Balancing Techniques (SMOTE, oversampling)  
- Machine Learning Model Training and Evaluation  
- Performance Metrics: Accuracy, F1 Score, Recall  

---

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Email-Spam-Detection.git
2.Install required packages (if not installed):
pip install pandas scikit-learn imbalanced-learn

3. Run the Python file containing the model training and prediction code:
python spam_detection.py

4. Follow the instructions or check the results printed/output by the program.
