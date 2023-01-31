#!/usr/bin/env python
# coding: utf-8

#Importing the dependencies
import pandas as pd
import numpy as np 
import streamlit as st
from IPython import get_ipython

# Loading the Dataset
RtData = pd.read_csv('Restaurant_Data.csv', encoding='latin')
# Selecting the restaurants located in India
RtData = RtData[(RtData.Currency == "Indian Rupees(Rs.)")]
# Removing the data where Average cost is 0
RtData = RtData.loc[(RtData['Average Cost for two'] > 0)]


# Deleting those columns which are not useful in predictive analysis because these variables are qualitative
UselessColumns = ['Restaurant ID', 'Restaurant Name','City','Address',
                  'Locality', 'Locality Verbose','Cuisines']
RtData = RtData.drop(UselessColumns,axis=1)
RtData.head(5)

RtData.rename(columns={'Has Table booking': 'Has_Table_booking', 'Has Online delivery' : 'Has_Online_delivery', 'Average Cost for two':'Average_Cost_for_two', 'Price range':'Price_range'}, inplace=True)

# Finding nearest values to 4000 mark 
RtData['Votes'][RtData['Votes']<4000].sort_values(ascending=False)

# Above result shows the nearest logical value is 3986, hence, replacing any value above 4000 with it.
# Replacing outliers with nearest possibe value
RtData['Votes'][RtData['Votes']>4000] =3986

# Above result shows the nearest logical value is 8000, hence, replacing any value above 50000 with it.
## Replacing outliers with nearest possibe value
RtData['Average_Cost_for_two'][RtData['Average_Cost_for_two']>50000] = 8000

#Final Selected Predictors
SelectedColumns=['Votes','Average_Cost_for_two','Has_Table_booking',
                 'Has_Online_delivery','Price_range']

# Selecting final columns
DataForML=RtData[SelectedColumns]

# Converting the binary nominal variable sex to numeric
DataForML['Has_Table_booking'].replace({'Yes':1, 'No':0}, inplace=True)
DataForML['Has_Online_delivery'].replace({'Yes':1, 'No':0}, inplace=True)

# Treating all the nominal variables at once using dummy variables
DataForML_Numeric=pd.get_dummies(DataForML)

# Adding Target Variable to the data
DataForML_Numeric['Rating']=RtData['Rating']

# Printing sample rows
DataForML_Numeric.head()

# Separate Target Variable and Predictor Variables
TargetVariable='Rating'
Predictors=['Votes', 'Average_Cost_for_two', 'Has_Table_booking',
           'Has_Online_delivery', 'Price_range']

X=DataForML_Numeric[Predictors].values
y=DataForML_Numeric[TargetVariable].values

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)

# XGBOOST Model
# Xtreme Gradient Boosting (XGBoost)
from xgboost import XGBRegressor
RegModel=XGBRegressor(max_depth=2, learning_rate=0.1, verbosity = 0, silent=True, n_estimators=1000, objective='reg:linear', booster='gbtree')

# Printing all the parameters of XGBoost
print(RegModel)

# Creating the model on Training Data
XGB=RegModel.fit(X_train,y_train)
prediction=XGB.predict(X_test)

@st.cache()

# Defining the function which will make the prediction using the data which the user inputs 
 

def prediction(Votes, Average_Cost_for_two, Has_Table_booking, Has_Online_delivery, Price_range):   
    pred = None
   
    if Has_Table_booking  == "No":
        Has_Table_booking = 0
    else:
        Has_Table_booking = 1
 
    if Has_Online_delivery == "No":
        Has_Online_delivery = 0
    else:
        Has_Online_delivery = 1
 
      
     # Making predictions 
    pred_inputs = XGB.predict(pd.DataFrame([[Votes, Average_Cost_for_two, Has_Table_booking, Has_Online_delivery, Price_range]]))

    
    if pred_inputs[0] <= 2:
        pred = 'It is a Low Rated Restaurant.'
    elif ((pred_inputs[0] >= 3) and (pred_inputs[0] <= 4)):
        pred = 'It is a Decent Rated Restaurant'
    elif pred_inputs[0] >= 4:
        pred = 'It is a High Rated Restaurant'

    return pred

        
    

def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:orange;padding:13px"> 
    <h1 style ="color:black;text-align:center;"> Restaurant Rating
     Prediction App</h1>
     <h8 style ="color:black;text-align:center;"> The data from an online food app, 
     which needs assistance in predicting the future success or failure of a business (restaurant),
      has been used in this case study. Such that they can choose whether to delete the restaurant 
      from their app or keep it. They have provided information from of 8643 eateries from different 
      states of India that are currently accessible on their app. It contains details about the 
      restaurants, including the overall rating. Below I have developed a machine learning model 
      that can predict a restaurant's rating based on its attributes.</h8> 
    </div> 
    """
    
     # Display dataset when check box is ON
    if st.checkbox('View dataset in table data format'):
       st.dataframe(RtData)

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Votes = st.number_input("No. of Votes (Range between 0 to 2500)")
    Average_Cost_for_two= st.number_input("Cost of 2 person between 50 to 8000 (Indian Rupees(Rs.))")
    Price_range = st.number_input("Price Range between 1(Inexpensive) to 4(Most Expensive)")
    Has_Table_booking= st.selectbox(' Has Table Booking',("Yes","No"))
    Has_Online_delivery= st.selectbox(' Has Online Delivery',("Yes","No"))
    result =""


    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Votes, Average_Cost_for_two, Has_Table_booking, Has_Online_delivery, Price_range) 
        st.success('Final Decision: {}'.format(result))
        
     
if __name__=='__main__': 
    main()

