#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[2]:


lead_df = pd.read_csv('Downloads/Lead Scoring Assignment/Leads.csv')
pd.set_option('display.max_columns', 100)
lead_df.head(10)


# ## Data Understanding

# In[3]:


lead_df.shape


# In[4]:


lead_df.info()


# - As we can see that there are only 7 numerical columns and remaining 30 columns are categorical columns.

# In[5]:


lead_df.describe()


# - From above we can observe that the columns namely 'TotalVisits', 'Total Time Spent on Website' & 'Page Views Per Visit' are having outliers, so we need to treat them to make the data clean.

# ## Data Cleaning

# In[6]:


# Removing the unnecessary columns

redun_col = ['Prospect ID', 'Lead Number', 'Country', 'I agree to pay the amount through cheque',
            'A free copy of Mastering The Interview', 'City']

lead_cl_df = lead_df.drop(redun_col, axis=1)


# In[7]:


# Checking the dataset

lead_cl_df.head()


# In[8]:


lead_cl_df.shape


# - From above we observe that there are some columns with labels 'Select', which means that the customer didn't select any of the given options, hence it is better to treat them as null values.

# In[9]:


# Now replacing label 'Select' with nan values

df_1 = pd.DataFrame(lead_cl_df['Specialization'])
df_2 = pd.DataFrame(lead_cl_df['How did you hear about X Education'])
df_3 = pd.DataFrame(lead_cl_df['Lead Profile'])


# In[10]:


# For df_1
df_1.loc[df_1['Specialization'] == 'Select', 'Specialization'] = np.nan

# For df_2
df_2.loc[df_2['How did you hear about X Education'] == 'Select', 'How did you hear about X Education'] = np.nan

# For df_3
df_3.loc[df_3['Lead Profile'] == 'Select', 'Lead Profile'] = np.nan


# In[11]:


# After replacing in temporary dataframe, we update our original dataset with new set of data from dataset

lead_cl_df['Specialization'] = df_1['Specialization']
lead_cl_df['How did you hear about X Education'] = df_2['How did you hear about X Education']
lead_cl_df['Lead Profile'] = df_3['Lead Profile']


# In[12]:


# Checking the null values

round(100*lead_cl_df.isnull().sum()/len(lead_cl_df),2)


# ## Dropping the columns having more than 35% of the null values

# In[13]:


# Dropping the columns having more than 35% of the null values

lead_cl_df_drop = lead_cl_df.loc[:, lead_cl_df.isnull().mean() > 0.35]
lead_cl_df_drop_1 = list(lead_cl_df_drop.keys())
lead_cl_df = lead_cl_df.drop(lead_cl_df_drop_1, axis=1)


# In[14]:


# Checking the null values

round(100*lead_cl_df.isnull().sum()/len(lead_cl_df),2)


# In[15]:


# Checking the shape

lead_cl_df.shape


# ## Checking those columns having less than 35% of null values and imputing there respective values

# In[16]:


# Checking the Lead Source col.

lead_cl_df['Lead Source'].value_counts().head()


# - Since 'Google' is having a higher number of occurences so, we will impute the null values with 'Google'.

# In[17]:


# Checking the Total Visits col.

lead_cl_df['TotalVisits'].value_counts().head()


# - Since the most occuring value here is '0.0', therefore imputing the missing values with '0.0'.

# In[18]:


# Checking the Page Views Per Visit col.

lead_cl_df['Page Views Per Visit'].value_counts().head()


# - Since the most occuring value here is '0.0', therefore imputing the missing values with '0.0'.

# In[19]:


# Checking the Last Activity col.

lead_cl_df['Last Activity'].value_counts().head()


# - Since the most occuring value here is 'Email Opened', therefore imputing the missing values with 'Email Opened'.

# In[20]:


# Checking the What is your current occupation  col.

lead_cl_df['What is your current occupation'].value_counts()


# - Since the most occuring value here is 'Unemployed', therefore imputing the missing values with 'Unemployed'.

# In[21]:


# Checking the What matters most to you in choosing a course col.

lead_cl_df['What matters most to you in choosing a course'].value_counts().head()


# - Since the most occuring value here is 'Better Career Prospects', therefore imputing the missing values with 'Better Career Prospects'.

# In[22]:


# Now imputing the missing values as per their respective values.

missing_val = {'Lead Source':'Google', 'TotalVisits':'0.0', 'Page Views Per Visit':'0.0', 'Last Activity':'Email Opened',
              'What is your current occupation':'Unemployed', 
               'What matters most to you in choosing a course':'Better Career Prospects'}

lead_cl_df = lead_cl_df.fillna(value = missing_val)


# In[23]:


# Checking the null values once more

lead_cl_df.isnull().sum()


# ## Now there are no null values present in the dataset.

# In[24]:


# Checking the Lead Source column for any spelling mistake

lead_cl_df['Lead Source'].value_counts()


# - We found that the 'Google' is being misprinted as 'google' which is making a duplicate in our data and can harm our analysis.
# - We need to treat it to make it same as 'Google'.

# In[25]:


# Treating the misprinted word

lead_cl_df['Lead Source'] = lead_cl_df['Lead Source'].apply(lambda x:x.capitalize())
lead_cl_df['Lead Source'].value_counts()


# - Now our dataset is good for the further analysis as all the values and this is our final step for the data cleaning.

# ## Data Transformation

# - Now converting the columns having 'Yes / No' to '1 / 0'.
# - Changing the numerical columns to categorical columns with the help of above conversion

# In[26]:


# Yes : 1
# No : 0

category = {'Yes':1, 'No':0}

# Do not Email col.
lead_cl_df['Do Not Email'] = lead_cl_df['Do Not Email'].map(category)

# Do Not Call col.
lead_cl_df['Do Not Call'] = lead_cl_df['Do Not Call'].map(category)

# Magazine col.
lead_cl_df['Magazine'] = lead_cl_df['Magazine'].map(category)

# Search col.
lead_cl_df['Search'] = lead_cl_df['Search'].map(category)

# Newspaper Article col.
lead_cl_df['Newspaper Article'] = lead_cl_df['Newspaper Article'].map(category)

# X Education Forums col.
lead_cl_df['X Education Forums'] = lead_cl_df['X Education Forums'].map(category)

# Newspaper col.
lead_cl_df['Newspaper'] = lead_cl_df['Newspaper'].map(category)

# Digital Advertisement col.
lead_cl_df['Digital Advertisement'] = lead_cl_df['Digital Advertisement'].map(category)

# Through Recommendations col.
lead_cl_df['Through Recommendations'] = lead_cl_df['Through Recommendations'].map(category)

# Receive More Updates About Our Courses col.
lead_cl_df['Receive More Updates About Our Courses'] = lead_cl_df['Receive More Updates About Our Courses'].map(category)

# Update me on Supply Chain Content col.
lead_cl_df['Update me on Supply Chain Content'] = lead_cl_df['Update me on Supply Chain Content'].map(category)

# Get updates on DM Content col.
lead_cl_df['Get updates on DM Content'] = lead_cl_df['Get updates on DM Content'].map(category)


# In[27]:


lead_cl_df.info()


# - After converting the binary categories from 'Yes' to 1 & 'No' to 0, we will now create dummy variables.

# In[28]:


# Creating dummy variables for the 8 categories and dropping the first level

dummy = pd.get_dummies(lead_cl_df[['Lead Origin','Lead Source','Last Activity' 
                                   ,'What is your current occupation', 'What matters most to you in choosing a course',
                                   'Last Notable Activity']], drop_first=True)

# Adding the dummies to the original dataset

lead_cl_df = pd.concat([lead_cl_df,dummy], axis=1)


# In[29]:


lead_cl_df.shape


# - Now removing the Duplicate columns

# In[30]:


# We have created a dummies for the below categories hence removing the original column.

dup = ['Lead Origin','Lead Source','Last Activity' ,'What is your current occupation', 
       'What matters most to you in choosing a course','Last Notable Activity']

lead_cl_df = lead_cl_df.drop(dup, axis=1)
lead_cl_df.shape


# In[31]:


# Removing redundant columns from the dataset

redun = ['Receive More Updates About Our Courses','Update me on Supply Chain Content',
         'Get updates on DM Content','Magazine']

lead_cl_df = lead_cl_df.drop(redun, axis=1)


# In[32]:


# Converting some categorical variables to numerical variables

lead_cl_df['TotalVisits'] = lead_cl_df['TotalVisits'].astype('float64') 
lead_cl_df['Page Views Per Visit'] = lead_cl_df['Page Views Per Visit'].astype('float64')


# In[33]:


lead_cl_df.info()


# - Till here we have changed all the datatypes to numeric types

# In[34]:


for col in lead_cl_df.select_dtypes(include='bool').columns:
    lead_cl_df[col] = lead_cl_df[col].astype('uint8')


# In[35]:


lead_cl_df.info()


# ## Checking for Outliers

# In[36]:


round(lead_cl_df.describe(percentiles=[0.15,0.35,0.55,0.75,0.95,0.99]),2)


# - We can say that 'TotalVisits' & 'Page Views Per Visit' have outliers in them and we need to treat them to make our dataset fit for the analysis.

# In[37]:


# Let's visualize the outliers

plt.figure(figsize=[15,10])
plt.tight_layout()
sns.set_style('whitegrid')

plt.subplot(1,3,1)
sns.boxplot(data = lead_cl_df, x = 'TotalVisits', palette='gist_heat', orient='v')
plt.title('TotalVisits')

plt.subplot(1,3,2)
sns.boxplot(data = lead_cl_df, x = 'Total Time Spent on Website', palette='gist_heat', orient='v')
plt.title('Total Time Spent on Website')

plt.subplot(1,3,3)
sns.boxplot(data = lead_cl_df, x = 'Page Views Per Visit', palette='gist_heat', orient='v')
plt.title('Page Views Per Visit')

plt.show()


# - From the above boxplots we can observe two outlier variables in our dataset ('TotalVisits' and 'Page Views Per Visit').
# - We need to do a 0.99-0.1 analysis in order to correct the outliers.

# In[38]:


lead_cl_df['TotalVisits'].describe()


# In[39]:


#Outlier Treatment: Remove top & bottom 1% of the Column Outlier values

Q3 = lead_cl_df.TotalVisits.quantile(0.99)
lead_cl_df = lead_cl_df[(lead_cl_df.TotalVisits <= Q3)]

Q1 = lead_cl_df.TotalVisits.quantile(0.01)
lead_cl_df = lead_cl_df[(lead_cl_df.TotalVisits >= Q1)]

sns.boxplot(y=lead_cl_df['TotalVisits'])
plt.show()


# In[40]:


lead_cl_df['Page Views Per Visit'].describe()


# In[41]:


#Outlier Treatment: Remove top & bottom 1% 

Q3 = lead_cl_df['Page Views Per Visit'].quantile(0.99)
lead_cl_df = lead_cl_df[lead_cl_df['Page Views Per Visit'] <= Q3]

Q1 = lead_cl_df['Page Views Per Visit'].quantile(0.01)
lead_cl_df = lead_cl_df[lead_cl_df['Page Views Per Visit'] >= Q1]

sns.boxplot(y=lead_cl_df['Page Views Per Visit'])
plt.show()


# In[42]:


lead_cl_df['Page Views Per Visit'].describe()


# - The outliers have been removed from the dataset, now our data is clean and free from outliers.

# ## Data Preparation

# ### Train-Test Split

# In[43]:


# Separating the Target Variable

y = lead_cl_df['Converted']

y.head()


# In[44]:


# Remaining dataset

X = lead_cl_df.drop('Converted', axis=1)

X.head()


# In[45]:


# Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=100)


# ## Feature Standardization

# In[46]:


scaler = StandardScaler()


# In[47]:


# Scaling the Total Time Spent on Website for the easy analysis.

X_train[['Total Time Spent on Website']] = scaler.fit_transform(X_train[['Total Time Spent on Website']])


# In[48]:


X_train.head()


# In[49]:


# Checking the conversion rate from 'Converted' column

round((sum(y)/len(y.index)*100),2)


# - We have a conversion rate of ~38.45 %.

# ## Correlation of the dataset

# In[50]:


# With the help of Heatmap we can identify the high correlated data.

plt.figure(figsize=[20,18])

sns.heatmap(lead_cl_df.corr(method='spearman'))

plt.title('Correlations\n')

plt.show()


# In[51]:


# Here, X Education Forums has no data so, it is better to remove from the dataset.
# Also, we need to remove the highly correlated value.

high_corr = ['X Education Forums','Lead Source_Olark chat', 'What is your current occupation_Unemployed']

X_train = X_train.drop(columns=high_corr)
X_test = X_test.drop(columns=high_corr)


# In[52]:


# Checking for the correlation again

plt.figure(figsize=[20,18])

sns.heatmap(lead_cl_df[X_train.columns].corr(method='spearman'))

plt.title('Correlations\n')

plt.show()


# - We have removed some of the correlated and null columns but it is quiet difficult to spot the high correlation attributes.
# - We will start building the model and with the help of VIFs and p-Value we will find out the relations.

# ## Building the Model

# In[ ]:






# In[53]:


logis=sm.GLM(y_train,(sm.add_constant(X_train)),familt=sm.families.Binomial())
logis.fit().summary()


# - There are many attributes having an insignificant p-values and we will try out the RFE for the feature elimination.

# ## RFE

# In[54]:


# Instantiating

logreg = LogisticRegression()


# In[55]:


# Running rfe with different variable count

# Running with 19 variables

rfem = RFE(logreg, n_features_to_select=19)
rfem = rfem.fit(X_train, y_train)


# In[56]:


# Checking for the true and false for the varibales after rfe

rfem.support_


# In[57]:


# Selecting the 'True' columns in rfem.support_

col = X_train.columns[rfem.support_]

X_train_1 = sm.add_constant(X_train[col]) # Adding constant


# In[58]:


# Creating 1st model after RFE

logis1=sm.GLM(y_train,X_train_1,family=sm.families.Binomial())

reg1=logis1.fit()

reg1.summary()


# - Now, From the above summary presented there are some features having high p -values, we will drop features which is having insignificant values one by one and create new model again and again until all the features attain significant p- value.

# ## VIF

# In[59]:


# Creating VIF Dataframe
vif = pd.DataFrame()

# Adding features
vif['Features'] = X_train_1[col].columns

# Calculating VIF
vif['VIF'] = [variance_inflation_factor(X_train_1[col].values,i) for i in range(X_train_1[col].shape[1])]

# Rounding the VIF values
vif['VIF']=round(vif['VIF'],2)

# Sorting the VIF values
vif=vif.sort_values(by='VIF',ascending=False)
vif


# ### As we can see that all features are having vif values less than 5, hence there is no multicollinearity issue in the dataset.

# As expained before we will drop the highest in-significant features i.e 'What is your current occupation_Housewife' having 0.999 p - value.

# In[60]:


# Dropping the most insignificant values ('What is your current occupation_Housewife') and constant

X_train_2 = X_train_1.drop(columns=['const','What is your current occupation_Housewife'])


# In[61]:


# Creating a new model

X_train_2 = sm.add_constant(X_train_2)                          
logis2 = sm.GLM(y_train,X_train_2,families=sm.families.Binomial())  
reg2 = logis2.fit()                                                 
reg2.summary()       


# In[62]:


# Dropping 'Last Activity_Email Bounced' and recreating the result

X_train_3 = X_train_2.drop(columns=['const','Last Activity_Email Bounced'])


# In[63]:


# Re-Creating a new model

X_train_3 = sm.add_constant(X_train_3)                              
logis3 = sm.GLM(y_train,X_train_3,families=sm.families.Binomial())  
reg3 = logis3.fit()                                                 
reg3.summary()  


# In[64]:


# Dropping 'What is your current occupation_Other' and recreating the model

X_train_4 = X_train_3.drop(columns = ['const','What is your current occupation_Other'])


# In[65]:


# Re-Creating a new model

X_train_4 = sm.add_constant(X_train_4)                              
logis4 = sm.GLM(y_train,X_train_4,families=sm.families.Binomial())  
reg4 = logis4.fit()                                                 
reg4.summary() 


# - Now, from the above summary we can say that all the variables present in this model are significant as no variables is having p - value greater than 5% hence we can proceed with our next step

# In[66]:


# Re-Checking the VIFs for the confirmation

# Checking VIF again just to be sure

X_train_4_1 = X_train_4.drop(columns='const')
vif=pd.DataFrame()                        
vif['Features']=X_train_4_1.columns       

# Now calculating

vif['VIF']=[variance_inflation_factor(X_train_4_1.values,i) for i in range(X_train_4_1.shape[1])]

# Rounding the vif values

vif['VIF']=round(vif['VIF'],2)

# Sorting the vif dataset

vif=vif.sort_values(by='VIF',ascending=False)

vif   # viewing the dataset


# - As there are no multicollinearity issues as all the values are below 5.00 and hence we can proceed to Predict the model.
# - Our final model is X_train_4 & reg4 and we are predicting our dataset based on this.

# ## Predicting the train model

# In[67]:


# Predicting the train dataset

y_train_pred = reg4.predict(X_train_4)

y_train_pred_final = pd.DataFrame({'Converted': y_train.values, 'Converted_probability': y_train_pred, 'ID': y_train.index})

y_train_pred_final.head()


# ## ROC Curve Plotting
# 
# - ROC curve shows the trade off between sensitivity and specificity - means if sensitivity increases specificity will decrease.
# - The curve closer to the left side border then right side of the border is more accurate.
# - The curve closer to the 45-degree diagonal of the ROC space is less accurate.

# In[68]:


# Importing libraries for roc_curve

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[69]:


# Creating a function to plot roc curve

def lead_roc(real, probability):
    
    # Creating roc curve values like false positive rate , true positive rate and threshold
    fpr, tpr, thresholds = roc_curve(real, probability, drop_intermediate=True)
    
    # Calculating the auc score(area under the curve)
    auc_score = roc_auc_score(real, probability)
    
    # Setting the figure
    plt.figure(figsize=[8,4])
    
    # Plotting the roc_curve
    plt.plot(fpr,tpr,label='ROC Curve (area= %0.2f)' %auc_score)
    
    # Plotting the 45% dotted line
    plt.plot([0,1], [0,1], 'r--')
             
    # Setting the x-axis limit
    plt.xlim([0.0, 1.0])
             
    # Setting the y-axis limit
    plt.ylim([0.0, 1.05])
             
    # Setting the x-axis label
    plt.xlabel('False Positive Rate')
             
    # Setting the y-axis label
    plt.ylabel('True Positive Rate')
             
    # Setting the title
    plt.title('Receiver Operating Characteristic')
    
    # Setting the legend on the left below to show the value of auc    
    plt.legend(loc="lower right")
    
    # Showing the plot
    plt.show()

    # no return         
    return None             


# In[70]:


# Calling the roc curve for plotting

lead_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_probability)


# Points to be noted from the ROC Curve
# 
# - The curve is closer to the left border than to the right border hence our model is having great accuracy.
# - The curve area is 88% of the total area.

# ## Finding the Optimal Cutoff Point

# In[71]:


# 10 points are being created out of which we will only one point for cutoff point.

numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i] = y_train_pred_final.Converted_probability.map(lambda x:1 if x>i else 0)
y_train_pred_final.head()


# Now, after creating series of points let's check the possibilities of choosing any one points from 0 to 0.9. We will do this by finding 'Accuracy', 'Sensitivity' and 'Specificity' for each points. These three methods will tell us how our model is - whether it is having low accuray or high or number of relevance data points is high or low etc.

# In[72]:


# Importing necessary library

from sklearn.metrics import confusion_matrix


# In[73]:


# Creating a dataframe to store all the values to be created

df_cutoffs=pd.DataFrame(columns=['Probability','Accuracy','Sensitvity','Specificity'])

# from 0 to 0.9 with set size 0.1

var=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]   

for i in var:
    cm_matrix=confusion_matrix(y_train_pred_final['Converted'],y_train_pred_final[i])  # creating confusion matrix 
    total=sum(sum(cm_matrix))                                                          # Taking the sum of the matrix
    accuracy=(cm_matrix[0,0]+cm_matrix[1,1])/total                                     # Storing Accuracy Data 
    sensitivity=cm_matrix[1,1]/(cm_matrix[1,0]+cm_matrix[1,1])                         # Storing Sensitivity Data
    specificity=cm_matrix[0,0]/(cm_matrix[0,0]+cm_matrix[0,1])                         # Storing Specificity Data
    df_cutoffs.loc[i]=[i, accuracy, sensitivity, specificity]                          # Inserting all the data into the dataframe created earlier

print(df_cutoffs)   


# As we can see from the above data we have created points for accuracy , sensitivity and specificity for all probability points from 0 to 0.9. Out of this we have to choose one as a cutoff point and it is probability cutoff = 0.4 because all the accuracy , sensitivity and specificity are having nearly same value which is an ideal point to consider for as we can't ignore any one from three.
# 
# Lets plot this data and see the convergence point for the 'accuracy', 'sensitivity' and 'specificity'.

# In[74]:


# Plotting 'Accuracy', 'Sensitivity' and 'Specificity' for various possibilities from 0 to 0.9

df_cutoffs.plot.line(x='Probability', y=['Accuracy','Sensitvity','Specificity'])
plt.show()


# - From the above graph it is prominent that 0.4 is perfect for the probability cutoff.

# In[75]:


# Predicting the outcomes with probability cutoff as 0.4 by creating new columns in the final dataset

# Predicted value
y_train_pred_final['Predicted'] = y_train_pred_final['Converted_probability'].map(lambda x:1 if x >0.4 else 0)  
 
y_train_pred_final.head()


# ## Precision and Recall

# In[76]:


# Creating confusion matrix to find precision and recall score

confusion_pr=confusion_matrix(y_train_pred_final.Converted,y_train_pred_final.Predicted)
confusion_pr


# In[77]:


print('Precision',confusion_pr[1,1]/(confusion_pr[0,1]+confusion_pr[1,1]))    # Printing Pecision score
print('Recall',confusion_pr[1,1]/(confusion_pr[1,0]+confusion_pr[1,1]))       # Printing Recall score


# Important points to be noted from the outcomes for precision and recall score -
# 
# - Our precison percentage is ~73% approximately and recall percentage is 79%.
# - This means we have very good model which explains relevancy of ~73% and true relevant results about 79%.
# 

# ## Precision and Recall Trade-Off

# In[78]:


# Importing precision recall curve from sklearn library

from sklearn.metrics import precision_recall_curve


# In[79]:


# Creating precision recall curve by crreating three points and plotting

p ,r, thresholds = precision_recall_curve(y_train_pred_final.Converted,y_train_pred_final.Converted_probability)
plt.title('Precision vs Recall tradeoff')
plt.plot(thresholds, p[:-1], "g-")    # Plotting precision
plt.plot(thresholds, r[:-1], "r-")    # Plotting Recall
plt.show()


# ## Prediction on the test dataset

# ### Scaling the dataset

# In[80]:


# Scalling the variables 'Total Time Spent on Website' with standard scaler and tranforming the X - test dataset

X_test[['Total Time Spent on Website']] = scaler.transform(X_test[['Total Time Spent on Website']])


# ### Predicting

# In[81]:


# Predicting the test dataset with our final model

test_cols = X_train_4.columns[1:]              # Taking the same column train set has
X_test_final = X_test[test_cols]               # Updating it in the final test set
X_test_final = sm.add_constant(X_test_final)   # Adding constant to the final set set
y_pred_test = reg4.predict(X_test_final)       # Predicting the final test set


# In[82]:


# Creating a new dataset and saving the prediction values in it

y_test_pred_final = pd.DataFrame({'Converted':y_test.values,'Converted_Probability':y_pred_test,'ID':y_test.index})

y_test_pred_final.head()


# ## Model Evaluation

# In[83]:


# Predicting the outcomes with probability cutoff as 0.4 by creating new columns in the final test dataset

# Predicted value
y_test_pred_final['Predicted'] = y_test_pred_final['Converted_Probability'].map(lambda x:1 if x>0.4 else 0 ) 

y_test_pred_final.head()


# In[84]:


# Importing the metrics library

from sklearn import metrics


# In[85]:


# Checking the accuracy of the test dataset.

print('Accuracy score in predicting test dataset :', metrics.accuracy_score(y_test_pred_final.Converted, 
                                                                           y_test_pred_final.Predicted))


# In[86]:


# Importing the Precision and Recall metrics

from sklearn.metrics import precision_score, recall_score


# In[87]:


# Checking the Precision and Recall score

print('Precision score in predicting test dataset:',precision_score(y_test_pred_final.Converted, 
                                                                    y_test_pred_final.Predicted))

print('Recall score in predicting test dataset:',recall_score(y_test_pred_final.Converted, 
                                                              y_test_pred_final.Predicted))


# ## Lead Score Assigning

# In[88]:


# Creating new columns for lead number and lead score

y_test_pred_final['Lead Number'] = lead_df.iloc[y_test_pred_final['ID'],1]

y_test_pred_final['Lead Score'] = y_test_pred_final['Converted_Probability'].apply(lambda x:round(x*100))

y_test_pred_final.head()


# ## Conclusion
#  - The Accuracy, Precision and Recall score we got from the test data are in the acceptable region.
#  - In business terms, this model has an ability to adjust with the company’s requirements in coming future.
#  - Important features responsible for good conversion rate or the ones' which contributes more towards the probability of a lead getting converted are:
#     - Last Notable Activity_Had a Phone Conversation
#     - Lead Origin_Lead Add Form
#     - What is your current occupation_Working Professional.

# In[ ]:




