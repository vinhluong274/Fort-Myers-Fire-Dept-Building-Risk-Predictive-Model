# Fort-Myers-Fire-Dept-Building-RIsk-Predictive-Model
This repository contains the necessary files to train, test, and deploy a decision tree classifier that can assign buildings within Fort Myers, FL a risk score between 1-9. 
 
To run this code, make sure all requirements are installed on your virtual machine. These are listed in the requirements.txt file. 

To update the data within this model, we have included UMInformationScienceProject.sql file which contains the exact query used to obtain the data we used. Please contact  Richard Caulkins from City of Fort Myers GIS department to request updated data. 

In order to edit/update this the interactive map, the following need to be obtained: <br>
Create a Plotly account and generate an API Key: https://plot.ly/python/getting-started/ <br>
Create a Mapbox account and get an access token: https://docs.mapbox.com/help/how-mapbox-works/access-tokens/



The following provides documents a high-level overview of our methods and implementation: 
## Loading & Preparing Datasets
We used four datasets in this analysis: violations, fire incidents, addresses, and parcel details. These files are included along with an sql file containing the queries used to obtain them. 

## Defining Risk within the Data
In our model, we define building risk as the combination of how frequent a building has fire calls as well as the duration those fire calls last. Buildings with more frequent calls are likely to be of higher risk as more incidents are occurring at the address. While shorter fire call durations indicate false alarms and small incidents, longer fire call durations indicate a higher risk incident and situation. Combining these metrics we can define a general measure of risk. 

### Calculating Risk
Now that we've created the necessary features within the dataset for risk calculation, we can assign risk scores to each building.

## Assigning Overall Risk Scores
Now that we given a frequency score and duration score to each address in our fire incidents dataset, we can average them to obtain an overall score, which will serve as the risk score we train our model on.

## Preparing the Model
Now that we have the necessary features and scores to train the model, we can begin training and testing it. 

## Training & Tuning the Model
In order to find the best parameters to run our model at, we wrote the depthTuning function below that finds runs the model and checks accuracies to compare. 

## Running the Model on the Idenified Parameters
Now that we identified the best depths to run the model at, we train and test the model at those set parameters. 

## Feature Importance
Seeing that the model was successful and accurate at predicting risk, we can check to see which features were most inclfuenced risk.

## Using our Model on Commercial Buildings
Now that we've trained the model on the features above, we can use it to predict commercial and multi-fam buildings' risk scores.

## Exporting to Excel
We can export these results to an excel spreadsheet that teh FMFD inspectors and firefighters can reference when doing their inspections.

## Plotting Prediction on an Interactive Map
To allow our users to explore our results we've plotted our predictions on a map of Fort Myers. Green points indicate buildings with risk scores from 0-3, orange are scores 3-6, and red are scores 6-10. The full interactive map can be found here: https://plot.ly/~fortmyersfiredept/2/

In order to edit/update this code, the following need to be obtained: 
Create a Plotly account and generate an API Key: https://plot.ly/python/getting-started/
Create a Mapbox account and get an access token: https://docs.mapbox.com/help/how-mapbox-works/access-tokens/

#### View Full Interactive Map here: https://plot.ly/~fortmyersfiredept/2
