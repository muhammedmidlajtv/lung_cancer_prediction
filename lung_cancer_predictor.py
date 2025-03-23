# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# from sklearn.pipeline import Pipeline
# import joblib
# import warnings
# import streamlit as st

# st.markdown(
#     """
#     <style>
#         /* Set background color for the whole page */
#         .stApp {
#             background-color: white !important;
#             color: black !important;
#         }

#         /* Ensure all text appears black */
#         h1, h2, h3, h4, h5, h6, p, div, span {
#             color: black !important;
#         }

#         /* Customize the button */
#         .stButton>button {
#             background-color: green !important;
#             color: white !important;
#             border-radius: 10px;
#             border: 2px solid white;
#             padding: 10px;
#             font-size: 16px;
#         }

#         /* Button hover effect */
#         .stButton>button:hover {
#             background-color: darkgreen !important;
#             color: white !important;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # st.title("Styled Streamlit App")
# # st.write("This is a Streamlit app with a white background and black text.")

# # Streamlit button
# # if st.button("Click Me"):
# #     st.write("Button clicked!")



# # st.markdown('<p class="big-font">Customized Title</p>', unsafe_allow_html=True)



# # The code is structured as follows:
# # 1. Data loading and exploration
# # 2. Data preprocessing
# # 3. Model training and evaluation
# # 4. User interface for predictions
# # 5. Model saving for deployment

# # --------------------------------
# # 1. DATA LOADING AND EXPLORATION
# # --------------------------------

# # For this example, I'll use a Kaggle lung cancer dataset
# # You can replace this with your preferred dataset
# def load_data():
#     """
#     Load the lung cancer dataset
#     You would typically download this from Kaggle or another source
#     """
#     # Sample URL (you should replace with actual dataset)
#     # url = "https://raw.githubusercontent.com/datasets/lung-cancer/main/data.csv"
    
#     # For demonstration purposes, I'll create a sample dataset
#     # In your actual project, you'd load real data
#     np.random.seed(42)
#     n_samples = 300
    
#     # Creating synthetic data based on common lung cancer risk factors
#     data = {
#         'Age': np.random.randint(20, 90, n_samples),
#         'Smoking': np.random.randint(0, 60, n_samples),  # Pack-years
#         'YellowFingers': np.random.randint(0, 2, n_samples),
#         'Anxiety': np.random.randint(0, 2, n_samples),
#         'PeerPressure': np.random.randint(0, 2, n_samples),
#         'ChronicDisease': np.random.randint(0, 2, n_samples),
#         'Fatigue': np.random.randint(0, 2, n_samples),
#         'Allergy': np.random.randint(0, 2, n_samples),
#         'Wheezing': np.random.randint(0, 2, n_samples),
#         'AlcoholConsuming': np.random.randint(0, 2, n_samples),
#         'Coughing': np.random.randint(0, 2, n_samples),
#         'ShortnessOfBreath': np.random.randint(0, 2, n_samples),
#         'SwallowingDifficulty': np.random.randint(0, 2, n_samples),
#         'ChestPain': np.random.randint(0, 2, n_samples),
#     }
    
#     # Create a target variable with some correlation to features
#     df = pd.DataFrame(data)
    
#     # Create a target variable (simplified for demonstration)
#     # In real life, the relationship would be much more complex
#     target = (
#         0.3 * df['Age'] / 90 + 
#         0.4 * df['Smoking'] / 60 + 
#         0.3 * df['ChestPain'] +
#         0.2 * df['Coughing'] +
#         0.2 * df['ShortnessOfBreath'] +
#         0.1 * df['Wheezing'] +
#         0.1 * df['ChronicDisease'] +
#         np.random.normal(0, 0.2, n_samples)
#     )
    
#     df['LungCancer'] = (target > 0.5).astype(int)
    
#     return df

# def explore_data(df):
#     """Perform exploratory data analysis on the dataset"""
#     print(f"Dataset shape: {df.shape}")
#     print("\nData overview:")
#     print(df.head())
    
#     print("\nData types:")
#     print(df.dtypes)
    
#     print("\nMissing values:")
#     print(df.isnull().sum())
    
#     print("\nStatistical summary:")
#     print(df.describe())
    
#     print("\nTarget distribution:")
#     print(df['LungCancer'].value_counts(normalize=True))
    
#     # Create correlation matrix
#     plt.figure(figsize=(12, 10))
#     corr = df.corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('Feature Correlation Matrix')
#     plt.savefig('correlation_matrix.png')
    
#     # Plot feature importance based on correlation with target
#     corr_with_target = corr['LungCancer'].sort_values(ascending=False)
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=corr_with_target.index, y=corr_with_target.values)
#     plt.xticks(rotation=90)
#     plt.title('Feature Correlation with Lung Cancer')
#     plt.tight_layout()
#     plt.savefig('feature_importance.png')

# # --------------------------------
# # 2. DATA PREPROCESSING
# # --------------------------------

# def preprocess_data(df):
#     """Preprocess the data for model training"""
#     # Separate features and target
#     X = df.drop('LungCancer', axis=1)
#     y = df['LungCancer']
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     return X_train, X_test, y_train, y_test

# # --------------------------------
# # 3. MODEL TRAINING AND EVALUATION
# # --------------------------------

# def train_model(X_train, y_train):
#     """Train the lung cancer prediction model"""
#     # Create a pipeline with preprocessing and model
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('model', RandomForestClassifier(n_estimators=100, random_state=42))
#     ])
    
#     # Train the model
#     pipeline.fit(X_train, y_train)
    
#     return pipeline

# def evaluate_model(model, X_test, y_test):
#     """Evaluate the model and print performance metrics"""
#     # Make predictions
#     y_pred = model.predict(X_test)
#     y_pred_proba = model.predict_proba(X_test)[:, 1]
    
#     # Calculate metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
    
#     # Calculate ROC curve and AUC
#     fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#     roc_auc = auc(fpr, tpr)
    
#     # Cross-validation score
#     cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
    
#     # Print evaluation metrics
#     print(f"Accuracy: {accuracy:.4f}")
#     print(f"Cross-validation scores: {cv_scores}")
#     print(f"Average CV score: {np.mean(cv_scores):.4f}")
#     print(f"ROC AUC: {roc_auc:.4f}")
    
#     print("\nConfusion Matrix:")
#     print(conf_matrix)
    
#     print("\nClassification Report:")
#     print(report)
    
#     # Plot ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.savefig('roc_curve.png')
    
#     # Plot confusion matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No Cancer', 'Cancer'],
#                 yticklabels=['No Cancer', 'Cancer'])
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.title('Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
    
#     return {
#         'accuracy': accuracy,
#         'cv_scores': cv_scores,
#         'avg_cv_score': np.mean(cv_scores),
#         'roc_auc': roc_auc,
#         'confusion_matrix': conf_matrix,
#         'classification_report': report
#     }

# def get_feature_importance(model, X):
#     """Extract and visualize feature importance"""
#     # Extract feature importance from the Random Forest model
#     feature_importance = model.named_steps['model'].feature_importances_
#     feature_names = X.columns
    
#     # Create a DataFrame for better visualization
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': feature_importance
#     }).sort_values(by='Importance', ascending=False)
    
#     # Plot feature importance
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x='Importance', y='Feature', data=importance_df)
#     plt.title('Feature Importance for Lung Cancer Prediction')
#     plt.tight_layout()
#     plt.savefig('model_feature_importance.png')
    
#     return importance_df

# # --------------------------------
# # 4. USER INTERFACE FOR PREDICTIONS
# # --------------------------------

# def create_prediction_interface():
#     """Create a Streamlit interface for users to input data and get predictions"""
#     st.title('Lung Cancer Prediction Tool')
#     st.write('Enter patient information to predict lung cancer risk')
    
#     # Load the trained model
#     model = joblib.load('lung_cancer_model.pkl')
    
#     # Create input fields for each feature
#     col1, col2 = st.columns(2)
    
#     with col1:
#         age = st.slider('Age', 20, 90, 50)
#         smoking = st.slider('Smoking (pack-years)', 0, 60, 0)
#         yellow_fingers = st.checkbox('Yellow Fingers')
#         anxiety = st.checkbox('Anxiety')
#         peer_pressure = st.checkbox('Peer Pressure')
#         chronic_disease = st.checkbox('Chronic Disease')
#         fatigue = st.checkbox('Fatigue')
    
#     with col2:
#         allergy = st.checkbox('Allergy')
#         wheezing = st.checkbox('Wheezing')
#         alcohol = st.checkbox('Alcohol Consumption')
#         coughing = st.checkbox('Coughing')
#         shortness_of_breath = st.checkbox('Shortness of Breath')
#         swallowing_difficulty = st.checkbox('Swallowing Difficulty')
#         chest_pain = st.checkbox('Chest Pain')
    
#     # Create a DataFrame from user input
#     input_data = pd.DataFrame({
#         'Age': [age],
#         'Smoking': [smoking],
#         'YellowFingers': [int(yellow_fingers)],
#         'Anxiety': [int(anxiety)],
#         'PeerPressure': [int(peer_pressure)],
#         'ChronicDisease': [int(chronic_disease)],
#         'Fatigue': [int(fatigue)],
#         'Allergy': [int(allergy)],
#         'Wheezing': [int(wheezing)],
#         'AlcoholConsuming': [int(alcohol)],
#         'Coughing': [int(coughing)],
#         'ShortnessOfBreath': [int(shortness_of_breath)],
#         'SwallowingDifficulty': [int(swallowing_difficulty)],
#         'ChestPain': [int(chest_pain)]
#     })
    
#     # Make prediction when user clicks the button
#     if st.button('Predict'):
#         # Get prediction and probability
#         prediction = model.predict(input_data)[0]
#         probability = model.predict_proba(input_data)[0][1]
        
#         # Display result
#         st.subheader('Prediction Result')
#         if prediction == 1:
#             st.error(f'High risk of lung cancer detected (Probability: {probability:.2%})')
#             st.write('Please consult with a healthcare professional for further evaluation.')
#         else:
#             st.success(f'Low risk of lung cancer (Probability: {probability:.2%})')
#             st.write('Regular health check-ups are still recommended.')
        
#         # Display risk factors
#         st.subheader('Key Risk Factors:')
#         importance_df = get_feature_importance(model, input_data)
#         top_factors = importance_df.head(5)['Feature'].tolist()
        
#         active_risk_factors = []
#         for factor in top_factors:
#             if factor == 'Age' and age > 50:
#                 active_risk_factors.append(f"Age ({age} years)")
#             elif factor == 'Smoking' and smoking > 0:
#                 active_risk_factors.append(f"Smoking ({smoking} pack-years)")
#             elif factor in input_data.columns and input_data[factor].values[0] == 1:
#                 active_risk_factors.append(factor)
        
#         if active_risk_factors:
#             st.write("Your highest risk factors:")
#             for factor in active_risk_factors:
#                 st.write(f"- {factor}")
#         else:
#             st.write("No significant risk factors identified in your inputs.")

# # --------------------------------
# # 5. MODEL SAVING FOR DEPLOYMENT
# # --------------------------------

# def save_model(model):
#     """Save the trained model for future use"""
#     joblib.dump(model, 'lung_cancer_model.pkl')
#     print("Model saved as 'lung_cancer_model.pkl'")

# # --------------------------------
# # MAIN EXECUTION
# # --------------------------------

# def main():
#     """Main function to execute the entire pipeline"""
#     # Step 1: Load data
#     print("Loading dataset...")
#     df = pd.read_csv("survey_lung_cancer.csv")
    
#     # Step 2: Explore data
#     print("\nExploring dataset...")
#     explore_data(df)
    
#     # Step 3: Preprocess data
#     print("\nPreprocessing data...")
#     X_train, X_test, y_train, y_test = preprocess_data(df)
    
#     # Step 4: Train model
#     print("\nTraining model...")
#     model = train_model(X_train, y_train)
    
#     # Step 5: Evaluate model
#     print("\nEvaluating model...")
#     metrics = evaluate_model(model, X_test, y_test)
    
#     # Step 6: Get feature importance
#     print("\nCalculating feature importance...")
#     importance_df = get_feature_importance(model, X_train)
#     print(importance_df)
    
#     # Step 7: Save model
#     print("\nSaving model...")
#     save_model(model)
    
#     print("\nModel development complete! Run the Streamlit app to use the prediction interface.")
#     print("Command: streamlit run this_script.py")

# # Uncomment to run the model training pipeline
# # if __name__ == "__main__":
# #     main()

# if __name__ == "__main__":
#     # df = load_data()
#     # explore_data(df)  # Optional, only for EDA

#     # X_train, X_test, y_train, y_test = preprocess_data(df)

#     # # Train and evaluate the model
#     # model = train_model(X_train, y_train)
#     # evaluation_results = evaluate_model(model, X_test, y_test)

#     # # Save the trained model
#     # joblib.dump(model, 'lung_cancer_model.pkl')
#     # print("Model training complete. Model saved as 'lung_cancer_model.pkl'.")
    
#     create_prediction_interface()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

# Custom styling for the Streamlit app
st.markdown(
    """
    <style>
        /* Set background color for the whole page */
        .stApp {
            background-color: white !important;
            color: black !important;
        }

        /* Ensure all text appears black */
        h1, h2, h3, h4, h5, h6, p, div, span {
            color: black !important;
        }

        /* Customize the button */
        .stButton>button {
            background-color: green !important;
            color: white !important;
            border-radius: 10px;
            border: 2px solid white;
            padding: 10px;
            font-size: 16px;
        }

        /* Button hover effect */
        .stButton>button:hover {
            background-color: darkgreen !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def get_feature_importance(model, X):
    """Extract feature importance from the model"""
    feature_importance = model.named_steps['model'].feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    return importance_df

def create_prediction_interface():
    """Create a Streamlit interface for users to input data and get predictions"""
    st.title('Lung Cancer Prediction Tool')
    st.write('Enter patient information to predict lung cancer risk')
    
    # Load the trained model
    model = joblib.load('lung_cancer_model.pkl')
    
    # Create input fields for each feature
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider('Age', 20, 90, 50)
        smoking = st.slider('Smoking (pack-years)', 0, 60, 0)
        yellow_fingers = st.checkbox('Yellow Fingers')
        anxiety = st.checkbox('Anxiety')
        peer_pressure = st.checkbox('Peer Pressure')
        chronic_disease = st.checkbox('Chronic Disease')
        fatigue = st.checkbox('Fatigue')
    
    with col2:
        allergy = st.checkbox('Allergy')
        wheezing = st.checkbox('Wheezing')
        alcohol = st.checkbox('Alcohol Consumption')
        coughing = st.checkbox('Coughing')
        shortness_of_breath = st.checkbox('Shortness of Breath')
        swallowing_difficulty = st.checkbox('Swallowing Difficulty')
        chest_pain = st.checkbox('Chest Pain')
    
    # Create a DataFrame from user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Smoking': [smoking],
        'YellowFingers': [int(yellow_fingers)],
        'Anxiety': [int(anxiety)],
        'PeerPressure': [int(peer_pressure)],
        'ChronicDisease': [int(chronic_disease)],
        'Fatigue': [int(fatigue)],
        'Allergy': [int(allergy)],
        'Wheezing': [int(wheezing)],
        'AlcoholConsuming': [int(alcohol)],
        'Coughing': [int(coughing)],
        'ShortnessOfBreath': [int(shortness_of_breath)],
        'SwallowingDifficulty': [int(swallowing_difficulty)],
        'ChestPain': [int(chest_pain)]
    })
    
    # Make prediction when user clicks the button
    if st.button('Predict'):
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.subheader('Prediction Result')
        if prediction == 1:
            st.error(f'High risk of lung cancer detected (Probability: {probability:.2%})')
            st.write('Please consult with a healthcare professional for further evaluation.')
        else:
            st.success(f'Low risk of lung cancer (Probability: {probability:.2%})')
            st.write('Regular health check-ups are still recommended.')
        
        # Display risk factors
        st.subheader('Key Risk Factors:')
        importance_df = get_feature_importance(model, input_data)
        top_factors = importance_df.head(5)['Feature'].tolist()
        
        active_risk_factors = []
        for factor in top_factors:
            if factor == 'Age' and age > 50:
                active_risk_factors.append(f"Age ({age} years)")
            elif factor == 'Smoking' and smoking > 0:
                active_risk_factors.append(f"Smoking ({smoking} pack-years)")
            elif factor in input_data.columns and input_data[factor].values[0] == 1:
                active_risk_factors.append(factor)
        
        if active_risk_factors:
            st.write("Your highest risk factors:")
            for factor in active_risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No significant risk factors identified in your inputs.")

if __name__ == "__main__":
    create_prediction_interface()
