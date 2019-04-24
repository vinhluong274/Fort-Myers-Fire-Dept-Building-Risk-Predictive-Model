
# coding: utf-8

# # Fort Myers Fire Department: Risk Identification Model

# In[23]:


#importing all the libraries and functions we'll need.
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import os
import requests
from tqdm import tnrange, tqdm
from time import sleep
tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')
import gmaps
import datetime

import scipy as sp
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import sklearn.ensemble as skens
import sklearn.metrics as skmetric
import sklearn.naive_bayes as sknb
import sklearn.tree as sktree
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.externals.six as sksix
import IPython.display as ipd

from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import accuracy_score
import pickle

import seaborn as sns
# sns.set(style='white', color_codes=True, font_scale=2)
# sns.set(rc={'figure.figsize':(20,14)})
sns.set(rc={'figure.figsize':(12,8)})
rc={'font.size': 20, 'axes.labelsize': 20, 'legend.fontsize': 18, 
    'axes.titlesize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
sns.set(rc=rc)


# ## Loading & Preparing Datasets

# In[2]:


#violations data prep
#all violations datasets
v = pd.read_csv('codecases_violationtype_minimumhousing.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)
v1 = pd.read_csv('codecases_violationtype_multifamily.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)
v2 = pd.read_csv('codecases_violationtype_permitrequired.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)
v3 = pd.read_csv('codecases_violationtype_propertymaintenance.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)
v4 = pd.read_csv('codecases_violationtype_unsafestructure.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)
v5 = pd.read_csv('codecases_violationtype_vacantbuilding.csv', names=['Code', 'TimeSpan', 'CaseNum', 'Address', 'Owner', 'STRAP','DateCreated'], header=1)

#making a master violations dataframe
violations = pd.concat([v, v1, v2, v3, v4, v5], ignore_index=True)


#formats the violation description to exclude unnessary info
violations.Code = violations.Code.apply(lambda x : x.split(':', 1)[1])

#dropping an irrelevant columns and reformatting the Date
violations.drop(['TimeSpan'], axis=1, inplace=True) #drops irrelevant data column
violations.DateCreated = violations.DateCreated.apply(lambda x: x.strip("Rec'd:   "))

#Renaming columns
violations = violations[['DateCreated', 'Code', 'Address', 'STRAP',]]

#Counting the total violations at each address so we can show the model which buildings have had more violations
violations['total_violations'] = violations.groupby('Address')['Address'].transform('count')

# one-hot encode the types of violations
violations = violations.join(pd.get_dummies(violations['Code'], prefix='Type'))

#giving a violation score to each building out of 10, buildings with more violations have a higher score. 
#this will help later as a feature the model can use to make its predictions. 
violationBuckets = [0,1,2,3,4,5,6,8,12,18,max(violations.total_violations)]
scores = [1,2,3,4,5,6,7,8,9,10]

violations['violationScore'] = pd.cut(violations['total_violations'], violationBuckets, right=True, include_lowest=True, labels=scores)

violations.head()


# In[3]:


#Cleaning and prepping the 'CFM' dataset which contains parcel details
cfm = pd.read_csv('CFM_Parcels.csv')

#Excluding columns that are irrelevant
cfm = cfm[['STRAP', 'FolioID','GISAcres', 'FireDistDesc',
       'LandUseCode', 'LandUseDesc','SiteAddress','Just', 'Assessed', 'Taxable', 'NewBuilt', 'BldgCount',
       'MinBuiltYear', 'MaxBuiltYear', 'TotalArea', 'HeatedArea', 'MaxStories',
       'Bedrooms', 'Bathrooms']]

#Renaming the columns so that the Address columns are consistent across datasets
cfm.columns = ['STRAP', 'FolioID','GISAcres', 'FireDistDesc','LandUseCode', 'LandUseDesc',
               'Address','Just', 'Assessed', 'Taxable', 'NewBuilt', 'BldgCount',
               'MinBuiltYear', 'MaxBuiltYear', 'TotalArea', 'HeatedArea', 'MaxStories',
               'Bedrooms', 'Bathrooms']

def oneHot(num): 
    if num > 0: 
        return 1
    else: 
        return 0

cfm.Taxable = cfm.Taxable.apply(lambda x: oneHot(x))
cfm.NewBuilt = cfm.NewBuilt.apply(lambda x: oneHot(x))

cfm = cfm.fillna(0)
cfm['Age'] = int(datetime.datetime.now().year) - ((cfm.MaxBuiltYear + cfm.MinBuiltYear)/2)

cfm.head()


# In[4]:


#Doing the same with the 'Addresses' dataset
adds = pd.read_csv('Address.csv')

#Looking at this dataset, the longitude and latitude coordinates are swapped, so we renamed it accordingly
adds = adds[['Updated', 'Add_Number', 'StreetName', 'StN_PosTyp', 'Post_Code', 'Unit', 'Long', 'Lat', 'Elev', 'SITEADDR']]
adds.columns = ['Updated', 'Add_Number', 'StreetName', 'StN_PosTyp', 'Post_Code','Unit', 'Lat', 'Long', 'Elev', 'Address']
adds.head()


# In[5]:


#Cleaning and prepping the 'Historical Fire Incidents' dataset which contains records of past fires
#Some lines in this dataset are improperly formatted we are skipping those
fires = pd.read_csv('FireIncidents.csv', error_bad_lines=False) 

#Excluding irrelevant columns
fires = fires[['FireAgency1','IncidentAddress','Longitude','Latitude','ApartmentNumber','TimeIncidentCreated','TimeIncidentClosed',
               'City','FireAgency','IncidentTypeCode','IncidentTypeDescription','DispositionCode', 'DispositionDescription']]

#Renaming columns
fires.columns = ['FireAgency1','Address','Longitude','Latitude','ApartmentNumber','TimeIncidentCreated','TimeIncidentClosed',
               'City','FireAgency','IncidentTypeCode','IncidentTypeDescription','DispositionCode', 'DispositionDescription']

fires.head()


# ## Defining Risk within the Data
# In our model, we define building risk as the combination of how frequent a building has fire calls as well as the duration those fire calls last. Buildings with more frequent calls are likely to be of higher risk as more incidents are occurring at the address. While shorter fire call durations indicate false alarms and small incidents, longer fire call durations indicate a higher risk incident and situation. Combining these metrics we can define a general measure of risk. 

# In[6]:


#Deriving fire call duration and frequency from the fire history dataset.

#Calculating time spent on each incident ~ longer durations indicate higher risk issues. 
fires.TimeIncidentCreated = pd.to_datetime(fires.TimeIncidentCreated)
fires.TimeIncidentClosed = pd.to_datetime(fires.TimeIncidentClosed)
fires['IncidentDuration'] = fires.TimeIncidentClosed - fires.TimeIncidentCreated
fires['IncidentDurationMinutes'] = fires.IncidentDuration.apply(lambda x : round(x.total_seconds()/60))

#filter out calls greater than 6 hours (likely to be reporting errors)
fires = fires[~(fires.IncidentDurationMinutes > 360)]

#Count the frequency of fires at each address 
fires.sort_values(by='IncidentDurationMinutes', ascending=False)
fires.groupby('Address').count()
fires['fireFrequency'] = fires.groupby('Address')['Address'].transform('count')

#Creating a master dataframe of all buildings with a fire call history, we will use this to calculate risk and train the model. 
merged = pd.merge(fires, violations, on='Address', how='left')
merged.violationScore = merged.violationScore.cat.add_categories([0])
merged = merged.fillna(0)
merged = pd.merge(merged, cfm, on='Address')

merged.head()


# ### Calculating Risk
# Now that we've created the necessary features within the dataset for risk calculation, we can assign risk scores to each building.

# In[7]:


#Calculating Risk For the Model

#Dividing the range of fire frequecies and fire call durations into 11 quantiles for scoring out of 10. 
frequencyQuantiles = list(fires.fireFrequency.quantile(np.linspace(.1,1,11,0)))
durationQuantiles = list(fires.IncidentDurationMinutes.quantile(np.linspace(.1,1,11,0)))

#first and last quantiles must include min and maxes within each range
frequencyQuantiles[0] = fires.fireFrequency.min()
frequencyQuantiles[10] = fires.fireFrequency.max()
durationQuantiles[0] = fires.IncidentDurationMinutes.min()
durationQuantiles[10] = fires.IncidentDurationMinutes.max()

#Assigning scores
#From the 11 quantiles above, we assign a respective score 1 for the lowest frequency/duration quantile and so on to 10 for the highest.
scores = [1,2,3,4,5,6,7,8,9,10]
merged['frequencyRisk'] = pd.cut(merged.fireFrequency, frequencyQuantiles, right=True, include_lowest=True, labels=scores)
merged['durationRisk'] = pd.cut(merged.IncidentDurationMinutes, durationQuantiles, right=True, include_lowest=True, labels=scores)

merged.head()


# ## Assigning Overall Risk Scores
# Now that we given a frequency score and duration score to each address in our fire incidents dataset, we can average them to obtain an overall score, which will serve as the risk score we train our model on.

# In[8]:


#Making sure all columns are of the same datatype.
merged.frequencyRisk = merged.frequencyRisk.astype(float)
merged.durationRisk = merged.durationRisk.astype(float)
merged.violationScore = merged.violationScore.astype(float)
riskScores = merged[['Address', 'frequencyRisk', 'durationRisk']]

#averaging both frequency and risk scores.
riskScores = riskScores.groupby('Address').mean()
riskScores['overallRisk'] = ((riskScores.frequencyRisk + riskScores.durationRisk)/2)

#Adding the scores back onto the master dataframe
riskScores = riskScores.reset_index()
riskScores = riskScores[['Address', 'overallRisk']]
merged = pd.merge(merged, riskScores, on='Address')

merged.head()


# In[9]:


#Bucketing risk scores into a classification between 1 and 9, 1 being low risk, 9 being highest risk.
merged['riskClassification'] = pd.cut(merged.overallRisk, [0,1,2,3,4,5,6,7,8,9] , right=True, include_lowest=True, labels=[1,2,3,4,5,6,7,8,9])


# In[10]:


merged.drop_duplicates('Address').riskClassification.value_counts(sort=False)


# ## Preparing the Model
# Now that we have the necessary features and scores to train the model, we can begin training and testing it. 

# In[11]:


#Only including the data fields that will help us identify the building or be useful for the model as a feature
modelData = merged[['Address', 'Longitude', 'Latitude', 'STRAP_y','FolioID','LandUseCode', 'LandUseDesc',
       'FireDistDesc','frequencyRisk', 'durationRisk', 'overallRisk', 'riskClassification',
        'total_violations','Type_ Minimum Housing', 'Type_ Multi-Family', 'Type_ Permit Required',
       'Type_ Property Maintenance', 'Type_ Site Work/Utilities',
       'Type_ Unsafe', 'Type_ Vacant Building','GISAcres', 'Just',
       'Assessed', 'Taxable', 'NewBuilt', 'BldgCount',
       'TotalArea', 'HeatedArea', 'MaxStories', 'Bedrooms',
       'Bathrooms', 'Age', 'violationScore']]
modelData.head()


# ## Training & Tuning the Model
# In order to find the best parameters to run our model at, we wrote the depthTuning function below that finds runs the model and checks accuracies to compare. 

# In[48]:


#Create a list to store accuracies of each depth value tested.
accuracies = []

#The following function tests each max_tree depth value from 1-30 five times and averages the accuracy.
def depthTuning(depth):
    acc = []
    for i in range(5):
        train, test = train_test_split(modelData, test_size=.25)

        features = ['total_violations','Type_ Minimum Housing', 'Type_ Multi-Family', 'Type_ Permit Required',
       'Type_ Property Maintenance', 'Type_ Site Work/Utilities',
       'Type_ Unsafe', 'Type_ Vacant Building','GISAcres', 'Just',
       'Assessed', 'Taxable', 'NewBuilt', 'BldgCount',
       'TotalArea', 'HeatedArea', 'MaxStories', 'Bedrooms',
       'Bathrooms', 'Age', 'violationScore']

        model = sktree.DecisionTreeClassifier(max_depth=depth, splitter='best', criterion='entropy')

        model.fit(train[features], train.riskClassification)

        dot_data = StringIO()
        export_graphviz(model, out_file=dot_data,  
                        filled=True, rounded=True,
                        special_characters=True)


        predictedRisk = model.predict(test[features])
        test['predictedRisk'] = predictedRisk
        test.head(10)

        accuracy = accuracy_score(test.riskClassification, predictedRisk)
        acc.append(accuracy)
    return sum(acc)/len(acc)

for d in range(1,31): 
    accuracies.append(depthTuning(d))

#Plot the average accuracy for each max_tree depth value to identify the best one. 
plt = pd.DataFrame({'Accuracy':accuracies, 'Max Depth': range(1,31)},)
sns.lineplot(y=plt['Accuracy'], x=plt['Max Depth'], color='b')


# ## Running the Model on the Idenified Parameters
# Now that we identified the best depths to run the model at, we train and test the model at those set parameters. 

# In[51]:


train, test = train_test_split(modelData, test_size=.25)

# features = ['GISAcres', 'Just', 'Assessed', 'Taxable', 'Age', 'NewBuilt',
#        'BldgCount', 'TotalArea', 'HeatedArea', 'MaxStories', 'Bedrooms',
#        'Bathrooms','total_violations', 'violationScore']

features = ['total_violations','Type_ Minimum Housing', 'Type_ Multi-Family', 'Type_ Permit Required',
       'Type_ Property Maintenance', 'Type_ Site Work/Utilities',
       'Type_ Unsafe', 'Type_ Vacant Building','GISAcres', 'Just',
       'Assessed', 'Taxable', 'NewBuilt', 'BldgCount',
       'TotalArea', 'HeatedArea', 'MaxStories', 'Bedrooms',
       'Bathrooms', 'Age', 'violationScore']

model = sktree.DecisionTreeClassifier(max_depth=27, criterion='entropy')

model.fit(train[features], train.riskClassification)

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)


predictedRisk = model.predict(test[features])
test['predictedRisk'] = predictedRisk
test.head(10)

accuracy = accuracy_score(test.riskClassification, predictedRisk)
print("Accuracy: {0:.2%}".format(accuracy))

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())


# ## Feature Importance
# Seeing that the model was successful and accurate at predicting risk, we can check to see which features were most inclfuenced risk.

# In[52]:


#Graphing the most important features
feat_importance = model.feature_importances_
plt = pd.DataFrame({'Feature Importance':feat_importance},
            index=train.columns[12:35])
plt = plt.sort_values(by='Feature Importance', ascending=False)
sns.barplot(x=plt['Feature Importance'], y=plt.index, color='y')


# ## Using our Model on Commercial Buildings
# Now that we've trained the model on the features above, we can use it to predict commercial and multi-fam buildings' risk scores.

# In[53]:


#First we need to generate a dataset of all commercial and multi-fam buildings. 

#mergiing the parcel details with violations data will create a dataframe with the necessary features the model needs to make predictions. 
commercial = pd.merge(cfm, violations, on='Address', how='left')

#Creating a list of all commercial and multi-fam building use descriptions to use to filter our buildings that do not fall in these categories. 
commercialList = ['SHOPPING CENTER, NEIGHBORHOOD', 'OFFICE BUILDING, ONE STORY', 'OFFICE BUILDING, MULTI-STORY', 'AUTO SALES', 'CONVENIENCE STORE', 
              'SANITARIUMS', 'COUNTY OWNED, OFFICES, LIBRARY, GOVERNMENT BLDG', 'VACANT GOVERNMENTAL', 'STORE, ONE (1) FLOOR', 'RESTAURANT', 'MOTEL', 
              'WAREHOUSING', 'MEDICAL OFFICE BUILDING', 'GARAGE, REPAIR', 'FEDERALLY OWNED, OFFICES, LIBRARY, GOVERNMENT BLDG', 'OPEN STORAGE', 'COMMERCIAL, VACANT',
              'GOLF COURSE', 'CONDOMINIUM RESERVE PARCEL', 'SHOPPING CENTER, COMMUNITY', 'RESTAURANT, DRIVE-IN', 'FINANCIAL INSTITUTION', 'PROFESSIONAL BUILDING', 
              'MULTI-FAMILY, 10 OR MORE UNITS', 'MUNICIPALLY OWNED, OFFICES, LIBRARY, GOVERNMENT BLDG', 'DAY CARE CENTERS', 'SCHOOL, PRIVATE', 'LODGES, CLUBS, UNION HALLS', 
              'TOURIST ATTRACTION', 'RETIREMENT HOME', 'LAND CONDO', 'GOVERNMENT OWNED, PUBLIC SCHOOL', 'NONPROFIT SERVICES', 'MULTI-FAMILY, LESS THAN 10 UNITS, RIVER', 
              'VEHICLE LUBE/WASH', 'NIGHT CLUB, BAR, LOUNGE', 'DEPARTMENT STORE', 'AUDITORIUMS, FREESTANDING', 'GOVERNMENT OWNED, PARK', 'SHOPPING CENTER, REGIONAL',
              'STATE OWNED, OFFICES, LIBRARY, GOVERNMENT BLDG', 'UTILITIES', 'LIGHT MANUFACTURING', 'SERVICE SHOP', 'LAUNDROMAT', 'VACANT INSTITUTIONAL', 
              'COUNTRY CLUBS', 'MOBILE HOME SUBDIVISION', 'ACREAGE, BEACH FRONT', 'MINERAL RIGHTS', 'SUBMERGED RIVER', 'APARTMENTS','SERVICE STATION', 'HOTEL', 'CO-OPERATIVE', 
              'MANUFACTURING OFFICES', 'HOME FOR THE AGED, ALF', 'MORTUARY, FUNERAL HOME', 'FLORIST', 'THEATRE', 'SWAMP', 'HEAVY MANUFACTURING', 'MINERAL PROCESSING', 
              'HOSPITAL, PRIVATE', 'LAUNDRY', 'GOVERNMENT OWNED, HOSPITAL', 'DORMITORY', 'MULTI-FAMILY, LESS THAN 10 UNITS, GOLF COURSE', 'FOOD PROCESSING', 
              'CENTRALLY ASSESSED', 'MULTI-FAMILY, LESS THAN 10 UNITS, CANAL', 'BOWLING ALLEY']

#Filtering out all buildings that do not fall in the categories above. 
commercial = commercial[commercial.LandUseDesc.isin(commercialList)]
commercial.violationScore = commercial.violationScore.cat.add_categories([0])
commercial = commercial.fillna(0)
commercial = commercial.drop_duplicates('Address')
commercial.head()


# In[54]:


#Let the model predict on the commercial set of buildings
commercial['Predicted Risk Score'] = model.predict(commercial[features])
commercial[['Address', 'Predicted Risk Score']].head()


# In[55]:


#Graph and visualize the distribution of risk scores
plt = sns.countplot(x='Predicted Risk Score', data=commercial, order=[1,2,3,4,5,6,7,8,9], color='r')
ax = plt
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=18, color='black', ha='center', va='bottom')


# Although there are over 5000 commercial buildings in the city, we were only able to identify 3,049 buildings within the data that had sufficient data to predict on. From the figure above we see that there are roughly 950 buildings out of 3000, or close to one third, that are classified as high risk (risk score greater than or equal to 6). 

# ## Exporting to Excel
# We can export these results to an excel spreadsheet that teh FMFD inspectors and firefighters can reference when doing their inspections.

# In[30]:


#Including only features that may be useful to inspectors and firefighters
excelExport = commercial[['Address', 'Predicted Risk Score', 'LandUseDesc', 'total_violations', 'GISAcres', 'TotalArea', 'HeatedArea',
       'MaxStories', 'Bedrooms', 'Bathrooms', 'Age','violationScore','STRAP_x', 'STRAP_y', 'FolioID', ]]

#Sorting by highest risk score
excelExport = excelExport.sort_values('Predicted Risk Score', ascending=False)
excelExport = excelExport.set_index('Address')

#exporting to excel spreadsheet
excelExport.to_excel('Commercial and Multi-Fam Buildings and Risk Scores.xlsx')


# ## Plotting Prediction on an Interactive Map
# To allow our users to explore our results we've plotted our predictions on a map of Fort Myers. Green points indicate buildings with risk scores from 0-3, orange are scores 3-6, and red are scores 6-10. The full interactive map can be found here: https://plot.ly/~fortmyersfiredept/2/
# 
# In order to edit/update this code, the following need to be obtained: 
# Create a Plotly account and generate an API Key: https://plot.ly/python/getting-started/
# Create a Mapbox account and get an access token: https://docs.mapbox.com/help/how-mapbox-works/access-tokens/
# 
# Copy and paste the keys into the following lines capitalized portions of code below:

# In[35]:


#PLOTLY API KEY
PLOTLY_USERNAME = 'YOUR-USERNAME-HERE' 
API_KEY = 'YOUR-API-KEY-HERE'

#MAPBOX ACCESS TOKEN
MAPBOX_TOKEN = 'YOUR-ACCESS-TOKEN-HERE'


# In[36]:


#Merge commercial dataset with addresses data to obtain long, lat coordinates
mapping = pd.merge(commercial, adds, on='Address')

#Drop all duplicate addresses
mapData = mapping.drop_duplicates('Address')


# In[37]:


#Set plotly credentials to use their mapping library
import plotly 
plotly.tools.set_credentials_file(username=PLOTLY_USERNAME, api_key=API_KEY)
import plotly.plotly as py
import plotly.graph_objs as go
from pandas import Series

mapbox_access_token = MAPBOX_TOKEN

#Text that will display when hovering over each building
mapData['text']  = "RISK SCORE: "+ mapData["Predicted Risk Score"].astype(str) + "<br>ADDRESS: " + mapData.Address +  "<br>USE TYPE: " + mapData.LandUseDesc

mapData['riskMarkerSize'] = pd.cut(mapData['Predicted Risk Score'], [0,3,4,5,6,7,8,9,10] , right=True, include_lowest=True, labels=[8.5,9,9.5,10,10.5,11,12,13])

mapData['riskMarkerColor'] = pd.cut(mapData['Predicted Risk Score'], [0,3,6,10] , right=True, include_lowest=True, labels=['rgb(63,204,104)', 'rgb(255,164,94)', 'rgb(225,0,0)'])

data = [
    go.Scattermapbox(
        lat=Series(mapData.Lat),
        lon=Series(mapData.Long),
        mode='markers',
        
        marker=go.scattermapbox.Marker(
            size=mapData.riskMarkerSize,
            color=mapData.riskMarkerColor
        ),
        text=mapData.text
    )
]

layout = go.Layout(
    mapbox=go.layout.Mapbox(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=26.625,
            lon=-81.84
        ),
        pitch=0,
        zoom=11.5
    ),
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='FMFD Interactive Building Risk Map')


# #### View Full Interactive Map here: https://plot.ly/~fortmyersfiredept/2

# In[ ]:




