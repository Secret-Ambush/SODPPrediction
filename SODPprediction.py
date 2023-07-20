import streamlit as st
import pandas as pd
from PIL import Image  
import pandas as pd
import os
import sqlalchemy


# execute to ensure local proxy variables not used
for k in ['HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)
    os.environ.pop(k.lower(), None)

engine= sqlalchemy.create_engine(
    os.environ['snowflake_conn'],
    execution_options=dict(autocommit=True)
)

print("success")
img = Image.open("emirates_logo.png")
img2 = Image.open("graphics.png")

st.sidebar.title('Menu')
st.sidebar.image(img2,width = 100)
side_bar = st.sidebar.radio('What would you like to view?', ['Classification and Regression Model', 'About the Chosen Classifier Model', 'Classifier Model Evaluation', 'About the Regression Model'])

if side_bar == 'Classification and Regression Model':
    header = st.container()
    data = st.container()
    features = st.container()

    with header:
        
        text_col,image_col = st.columns((5.5,1))
        
        with text_col:
            st.title("SODP Classification and Prediction Model")
            st.markdown("This webapp classifies whether the flight will get full 30 days prior to departure")
            st.markdown("Additionally, it predicts when exactly the flight will be full")
        
        with image_col:
            st.write("##")
            st.image(img,width = 100)

    with data:
        st.header("DATASET")
        st.markdown("I was given a comprehensive **dataset** exhibiting flights from **2017 to 2019**")

        dtypes = {'flight_number': str}
        query = "Select * from trainingdata limit 100"
        training_data = pd.read_sql(query, engine)

        st.dataframe(training_data.tail(50),height=510)

        st.write("##")

    with features:
        st.header("Classifier Model / Analytics")
        st.markdown("SODP (30 days prior):")
        
        # chart = Chart()
        # chart.animate(
        # Config.pie(
        #         {
        #             "angle": "Quantity",
        #             "by": "Product",
        #             "title": "Pie Chart",
        #         }
        #     )
        # )

        df1 = pd.read_sql("SELECT DISTINCT lego FROM trainingdata ORDER BY lego", engine)
        options_1 = df1['lego'].tolist()
        selected_option_1 = st.selectbox('Select Leg Origin', options_1)

        df2 = pd.read_sql(f"SELECT DISTINCT legd FROM trainingdata WHERE lego = '{selected_option_1}' ORDER BY legd", engine)
        options_2 = df2['legd'].tolist()
        selected_option_2 = st.selectbox('Select Leg Destination', options_2)

        df3 = pd.read_sql(f"SELECT DISTINCT fltdep FROM trainingdata WHERE lego = '{selected_option_1}' AND legd = '{selected_option_2}' AND fltdep > '2018-12-31' ORDER BY fltdep", engine)
        options_3 = df3['fltdep'].tolist()
        selected_option_3 = st.selectbox('Select Departure Date', options_3)

        df4 = pd.read_sql(f"SELECT DISTINCT fltnum FROM trainingdata WHERE lego = '{selected_option_1}' AND legd = '{selected_option_2}' AND fltdep = '{selected_option_3}' ORDER BY fltnum", engine)
        options_4 = df4['fltnum'].tolist()
        selected_option_4 = st.selectbox('Select Flight Number', options_4)

        st.write("You have selected:")
        st.write('Origin:', selected_option_1, ", Destination: ", selected_option_2, ", Flt dep: ", selected_option_3, ", Flt num: ", selected_option_4)
        
        st.button("PREDICT")
        
        
        
elif side_bar == 'About the Chosen Classifier Model':
    print("blah")
        
    st.markdown(""" ## SODP Classification Documentation

### Overview

This code performs a flight capacity prediction task using historical data for different flights. The goal is to predict whether a flight will be full 30 days prior to its departure date. The code uses the `pandas`, `sqlalchemy`, `tqdm`, and `scikit-learn` libraries to read data from a SQL database, process it, build a predictive model using XGBoost, and evaluate the model's performance.

### Prerequisites

Before running the code, ensure that you have the following libraries installed:

- `pandas`: For data manipulation and analysis.
- `sqlalchemy`: For creating a connection to the SQL database.
- `tqdm`: For creating a progress bar to monitor the processing of chunks.
- `scikit-learn`: For building the predictive model and evaluating its performance.
- `xgboost`: For the XGBoost classifier.

### Steps

1. Import the necessary libraries:

```python
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
import gc
from sklearn.metrics import roc_auc_score, accuracy_score
```

2. Connect to the SQL database using the `create_engine` function from `sqlalchemy`.

3. Define the SQL query to retrieve distinct flight numbers (`fltnum`) from the database, sorted in descending order.

4. Set the `chunksize` to determine the number of rows to be processed at a time. This is useful for handling large datasets efficiently.

5. Calculate the total number of rows and chunks to be processed.

6. Initialize a progress bar using `tqdm` to track the processing of chunks.

7. Loop through the chunks obtained from the SQL query and perform the following steps for each chunk:

    a. Process each flight number (`fltnum`) in the chunk:
    
        - Retrieve data for the specific flight from the database.
        - Preprocess the data, filling NaN values, and calculating a new feature `sodp`.
        - Check if the flight is always full, never full, or if it requires balancing.
        - If the flight requires balancing due to an imbalanced class distribution, perform oversampling using the `resample` function.
        - Encode categorical features using frequency encoding.
        - Split the data into training and testing sets based on the snapshot date.
        - Build an XGBoost classifier model and fit it to the training data.
        - Predict the labels for the test data and evaluate the model's performance using ROC AUC score and accuracy.

    b. Update the progress bar after processing each chunk.

8. Close the progress bar after processing all chunks.


### Note

Please ensure that you have configured the `engine` object to correctly connect to your SQL database before running the code. Additionally, ensure that all the required libraries are installed and accessible in your Python environment. The code makes use of `XGBoost`, so it's essential to have it installed as well.

This documentation serves as a high-level overview of the code and its functionality. If you have any questions or need further details, feel free to ask.""")
    
elif side_bar == "Classifier Model Evaluation":
   st.markdown("""# Classifier Model Evaluation and Comparision

## Overview

I developed a flight capacity prediction model to forecast whether a flight will be fully booked 30 days prior to its departure date. Throughout the project, I experimented with several machine learning algorithms, including Random Forest Classifier, XGBoost Classifier, Gaussian Naive Bayes, Decision Tree Classifier, and Support Vector Machine (SVM). Each model was evaluated using relevant metrics, such as accuracy, ROC AUC score, and the confusion matrix.

## Data Processing

I started by preprocessing the data, including feature selection and target variable preparation. I extracted the features from the dataset and encoded the target variable to represent discrete classes.

## Model Evaluation

### Random Forest Classifier

I trained the Random Forest Classifier on the training data and predicted the labels for the testing data. The model achieved an accuracy of X% and an ROC AUC score of Y%. The confusion matrix visualized the model's performance, showcasing the number of true positives, true negatives, false positives, and false negatives.

### Hyperparameter Tuning with GridSearchCV

To optimize the Random Forest Classifier, I performed hyperparameter tuning using GridSearchCV. By trying various combinations of hyperparameters (e.g., n_estimators, max_features, max_depth, and max_leaf_nodes), I found the best model configuration. After fitting the model with the best hyperparameters, I re-evaluated its performance, yielding an accuracy of X% and an ROC AUC score of Y%.

### Hyperparameter Tuning with RandomizedSearchCV

For a different approach to hyperparameter tuning, I used RandomizedSearchCV. This technique randomly selects hyperparameter values from a parameter grid, and I evaluated the model with these random hyperparameters. The best combination of hyperparameters was determined, and the model was refitted with these values. The final evaluation showed an accuracy of X% and an ROC AUC score of Y%.

### XGBoost Classifier

Moving on to the XGBoost Classifier, I trained the model on the training data and predicted the labels for the testing data. The model achieved an accuracy of X%.

### Gaussian Naive Bayes

For the Gaussian Naive Bayes model, I trained the classifier on the training data and predicted the labels for the testing data. The model achieved an accuracy of X%.

### Decision Tree Classifier

Next, I evaluated the Decision Tree Classifier. After training the model and predicting the labels for the testing data, the model achieved an accuracy of X%.

### Support Vector Machine (SVM)

Finally, I evaluated the Support Vector Machine model with a linear kernel. The model was trained on the training data and predicted the labels for the testing data. The SVM achieved an accuracy of X%.

## Conclusion

In conclusion, I tested various machine learning models for flight capacity prediction and evaluated their performances. The Random Forest Classifier with hyperparameter tuning using GridSearchCV demonstrated the highest accuracy and ROC AUC score, making it the most suitable model for this task. However, the final model selection should be based on specific project requirements and consideration of other factors, such as interpretability and computational resources. This evaluation process provided valuable insights into the strengths and weaknesses of each model, enabling me to make informed decisions for flight capacity prediction.""")
    
elif side_bar == 'About the Regression model':
    print("blah")
    
    
    
    
    

    
    
    
    

    




