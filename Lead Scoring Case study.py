#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Leads.csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


# Describing data
df.describe()


# ## Data Cleaning

# In[8]:


col=[col for col in df.columns]


# In[9]:


for columns in col:
    print (df[columns].value_counts())
    print('-------------------------------------------------------------')


# In[10]:


df=df.replace('Select',np.NaN)


# In[11]:


for columns in col:
    print (df[columns].value_counts())
    print('-------------------------------------------------------------')


# In[12]:


df.isnull().sum()


# In[13]:


null=round((df.isnull().sum()/len(df.index)*100),2)


# In[14]:


null_40= list(null[null>40].index)


# In[15]:


null_40


# In[16]:


dfnew = df.drop(null_40, axis=1)


# In[17]:


dfnew.head()


# In[18]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[19]:


df['Country'].value_counts()


# In[20]:


plt.figure(figsize=(25,20))

sns.countplot(dfnew['Country'])


# In[21]:


dfnew.Country=dfnew.Country.replace(np.NaN,"India")


# In[22]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[23]:


plt.figure(figsize=(20,18))

sns.countplot(dfnew['Specialization'])
plt.xticks(rotation=90)
plt.show()


# In[24]:


dfnew.Specialization.value_counts()


# In[25]:


dfnew.Specialization=dfnew.Specialization.replace(np.NaN,"Others")


# In[26]:


plt.figure(figsize=(20,18))

sns.countplot(dfnew['What is your current occupation'])
plt.xticks(rotation=90)
plt.show()


# In[27]:


dfnew['What is your current occupation']=dfnew['What is your current occupation'].replace(np.nan,'Unemployed')


# In[28]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[29]:


plt.figure(figsize=(20,18))

sns.countplot(dfnew['What matters most to you in choosing a course'])
plt.xticks(rotation=90)
plt.show()


# In[30]:


dfnew=dfnew.drop(['What matters most to you in choosing a course'],axis=1)


# In[31]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[32]:


plt.figure(figsize=(20,18))

sns.countplot(dfnew['Tags'])
plt.xticks(rotation=90)
plt.show()


# In[33]:


# Imputing the missing data in the tags column with 'Will revert after reading the email'
dfnew['Tags']=dfnew['Tags'].replace(np.nan,'Will revert after reading the email')


# In[34]:


plt.figure(figsize=(20,18))

sns.countplot(dfnew['City'])
plt.xticks(rotation=90)
plt.show()


# In[35]:


dfnew.City=dfnew.City.replace(np.NaN,"Mumbai")


# In[36]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[37]:


dfnew=dfnew.dropna()


# In[38]:


round((dfnew.isnull().sum()/len(df.index)*100),2)


# In[39]:


dfnew.shape


# ### we had retained more than 9000 rows after cleaning the missing value

# In[40]:


# Checking for duplicates


# In[41]:


dfnew[dfnew.duplicated()]


# ### UNIVARIATE ANALYASIS

# In[42]:


Converted = (sum(dfnew['Converted'])/len(dfnew['Converted'].index))*100
Converted


# In[43]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Lead Origin'])


# In[44]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Lead Origin'],hue=dfnew['Converted'])


# ## Inference :
# API and Landing Page Submission have 30-35% conversion rate but count of lead originated from them are considerable.
# Lead Add Form has more than 90% conversion rate but count of lead are not very high.
# Lead Import are very less in count.
# To improve overall lead conversion rate, we need to focus more on improving lead converion of API and Landing Page Submission origin and generate more leads from Lead Add Form.
# 
# 

# In[45]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Lead Source'],hue=dfnew['Converted'])
plt.xticks(rotation=90)


# In[46]:


dfnew['Lead Source'] = dfnew['Lead Source'].replace(['google'], 'Google')


# In[47]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Lead Source'],hue=dfnew['Converted'])
plt.xticks(rotation=90)


# In[48]:


dfnew['Lead Source'] = dfnew['Lead Source'].replace(['Click2call', 'Live Chat', 'NC_EDM', 'Pay per Click Ads', 'Press_Release',
  'Social Media', 'WeLearn', 'bing', 'blog', 'testone', 'welearnblog_Home', 'youtubechannel'], 'Others')


# In[49]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Lead Source'],hue=dfnew['Converted'])
plt.xticks(rotation=90)


# ## Inference
# Google and Direct traffic generates maximum number of leads.
# Conversion Rate of reference leads and leads through welingak website is high.
# 
# To improve overall lead conversion rate, focus should be on improving lead converion of olark chat, organic search, direct traffic, and google leads and generate more leads from reference and welingak website.

# In[50]:


sns.countplot(x = "Do Not Email", hue = "Converted", data = dfnew)
plt.xticks(rotation = 90)


# In[51]:


sns.countplot(x = "Do Not Call", hue = "Converted", data = dfnew,palette='Set1')
plt.xticks(rotation = 90)


# In[52]:


dfnew['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[53]:


sns.boxplot(y=dfnew['TotalVisits'],orient='v',palette='magma')


# In[54]:


percentiles = dfnew['TotalVisits'].quantile([0.05,0.95]).values
dfnew['TotalVisits'][dfnew['TotalVisits'] <= percentiles[0]] = percentiles[0]
dfnew['TotalVisits'][dfnew['TotalVisits'] >= percentiles[1]] = percentiles[1]


# In[55]:


sns.boxplot(y=dfnew['TotalVisits'],orient='v',palette='magma')


# In[56]:


sns.boxplot(y=dfnew['TotalVisits'],x=dfnew['Converted'],orient='v',palette='magma')


# ## Inference
# Median for converted and not converted leads are the same.
# 
# Nothing can be concluded on the basis of Total Visits.

# In[57]:


dfnew['Total Time Spent on Website'].describe()


# In[58]:


sns.boxplot(y=dfnew['Total Time Spent on Website'],orient='v',palette='magma')


# In[59]:


sns.boxplot(y=dfnew['Total Time Spent on Website'],x=dfnew['Converted'],orient='v',palette='magma')


# ## Inference
# Leads spending more time on the weblise are more likely to be converted.
# Website should be made more engaging to make leads spend more time.

# In[60]:


dfnew['Page Views Per Visit'].describe()


# In[61]:


sns.boxplot(y=dfnew['Page Views Per Visit'],orient='v',palette='magma')


# In[62]:


percentiles = dfnew['Page Views Per Visit'].quantile([0.05,0.95]).values
dfnew['Page Views Per Visit'][dfnew['Page Views Per Visit'] <= percentiles[0]] = percentiles[0]
dfnew['Page Views Per Visit'][dfnew['Page Views Per Visit'] >= percentiles[1]] = percentiles[1]


# In[63]:


sns.boxplot(y=dfnew['Page Views Per Visit'],orient='v',palette='magma')


# In[64]:


sns.boxplot(y=dfnew['Page Views Per Visit'],x=dfnew['Converted'],orient='v',palette='magma')


# ### Inference
# Median for converted and unconverted leads is the same.
# Nothing can be said specifically for lead conversion from Page Views Per Visit

# In[65]:


dfnew['Last Activity'].describe()


# In[66]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Last Activity'],hue=dfnew['Converted'],palette='magma')
plt.xticks(rotation = 90)


# In[67]:


dfnew['Last Activity'] = dfnew['Last Activity'].replace(['Had a Phone Conversation', 'View in browser link Clicked', 
                                                       'Visited Booth in Tradeshow', 'Approached upfront',
                                                       'Resubscribed to emails','Email Received', 'Email Marked Spam'], 'Other_Activity')


# In[68]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Last Activity'],hue=dfnew['Converted'],palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most of the lead have their Email opened as their last activity.
# Conversion rate for leads with last activity as SMS Sent is almost 60%.

# In[69]:


plt.figure(figsize=(15,10))
sns.countplot(dfnew['Country'],hue=dfnew['Converted'],palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most values are 'India' no such inference can be drawn

# In[70]:


plt.figure(figsize=(15,6))
sns.countplot(x = "Specialization", hue = "Converted", data = dfnew,palette='Set1')
plt.xticks(rotation = 90)


# ## Inference
# Focus should be more on the Specialization with high conversion rate.

# In[71]:


sns.countplot(x = "Search", hue = "Converted", data = dfnew,palette='Set1')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.

# In[72]:


sns.countplot(x = "Newspaper Article", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter.

# In[73]:


sns.countplot(x = "X Education Forums", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[74]:


dfnew.info()


# In[75]:


sns.countplot(x = "Newspaper", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[76]:


sns.countplot(x = "Magazine", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[77]:


sns.countplot(x = "Digital Advertisement", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[78]:


sns.countplot(x = "Through Recommendations", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[79]:


sns.countplot(x = "Receive More Updates About Our Courses", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[80]:


plt.figure(figsize=(15,10))
sns.countplot(x = "Tags", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# In[81]:


sns.countplot(x = "Update me on Supply Chain Content", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[82]:


sns.countplot(x = "Get updates on DM Content", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[83]:


sns.countplot(x = "City", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most leads are from mumbai with around 50% conversion rate.

# In[84]:


dfnew.info()


# In[85]:


sns.countplot(x = "I agree to pay the amount through cheque", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[86]:


sns.countplot(x = "A free copy of Mastering The Interview", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# ## Inference
# Most entries are 'No'. No Inference can be drawn with this parameter

# In[87]:


sns.countplot(x = "Last Notable Activity", hue = "Converted", data = dfnew,palette='magma')
plt.xticks(rotation = 90)


# In[88]:


plt.figure(figsize=(25,20))
sns.heatmap(dfnew.corr(),annot=True,cmap='RdBu')


# ## Results
# Based on the univariate analysis we have seen that many columns are not adding any information to the model, hence we can drop them for further analysis

# In[89]:


dfnew = dfnew.drop(['Lead Number','Tags','Country','Search','Magazine','Newspaper Article','X Education Forums',
                            'Newspaper','Digital Advertisement','Through Recommendations','Receive More Updates About Our Courses',
                            'Update me on Supply Chain Content','Get updates on DM Content','I agree to pay the amount through cheque',
                            'A free copy of Mastering The Interview'],1)


# In[90]:


dfnew.head()


# In[91]:


dfnew.shape


# In[92]:


dfnew.info()


# In[93]:


vars =  ['Do Not Email', 'Do Not Call']

def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

dfnew[vars] = dfnew[vars].apply(binary_map)


# In[94]:


# Creating a dummy variable for the categorical variables and dropping the first one.
dummy_data = pd.get_dummies(dfnew[['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity']], drop_first=True)
dummy_data.head()


# In[95]:


# Concatenating the dummy_data to the lead_data dataframe
dfnew = pd.concat([dfnew, dummy_data], axis=1)
dfnew.head()


# In[96]:


dfnew = dfnew.drop(['Lead Origin', 'Lead Source', 'Last Activity', 'Specialization','What is your current occupation',
                             'City','Last Notable Activity'], axis = 1)


# In[97]:


dfnew.head()


# In[98]:


from sklearn.model_selection import train_test_split

dftrain,dftest=train_test_split(dfnew,train_size=0.7,random_state=341)


# In[99]:


dftrain.shape


# In[100]:


dftest.shape


# In[101]:


dftrain.info()


# In[102]:


xtrain=dftrain.drop(['Prospect ID','Converted'],axis=1)
xtest=dftest.drop(['Prospect ID','Converted'],axis=1)
ytrain=dftrain['Converted']
ytest=dftest['Converted']


# In[103]:


xtrain.info()


# In[104]:


xtest.info()


# In[105]:


xtrain.shape


# In[106]:


xtest.shape


# In[107]:


ytrain.shape


# In[108]:


ytest.shape


# In[109]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xtrain[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(xtrain[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

xtrain.head()


# In[110]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 20)             # running RFE with 20 variables as output
rfe = rfe.fit(xtrain, ytrain)


# In[111]:


rfe.support_


# In[112]:


list(zip(xtrain.columns, rfe.support_, rfe.ranking_))


# In[113]:


# Viewing columns selected by RFE
cols = xtrain.columns[rfe.support_]
cols


# In[114]:


import statsmodels.api as sm


# ### Model 1

# In[115]:


xtrain1=sm.add_constant(xtrain[cols])
lg1=sm.GLM(ytrain,xtrain1, family = sm.families.Binomial()).fit()


# In[116]:


lg1.summary()


# In[117]:


# Dropping the column 'What is your current occupation_Housewife' because of high p value
xtrain2 = xtrain1.drop(['What is your current occupation_Housewife'],axis=1)


# ## Model 2

# In[118]:


lg2=sm.GLM(ytrain,xtrain2,family = sm.families.Binomial()).fit()


# In[119]:


lg2.summary()


# ## Model 3

# In[120]:


# Dropping the column 'Last Notable Activity_Had a Phone Conversation' as it has high p value
xtrain3 = xtrain2.drop(['Last Notable Activity_Had a Phone Conversation'],axis=1)
lg3=sm.GLM(ytrain,xtrain3,family = sm.families.Binomial()).fit()


# In[121]:


lg3.summary()


# In[122]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = xtrain3.columns
vif['VIF'] = [variance_inflation_factor(xtrain3.values, i) for i in range(xtrain3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)


# In[123]:


vif


# ## Model-4

# In[124]:


# Dropping the column 'Lead Origin_Lead Add Form' as it has high vif value
xtrain4 = xtrain3.drop(['Lead Origin_Lead Add Form'],axis=1)
lg4=sm.GLM(ytrain,xtrain4,family = sm.families.Binomial()).fit()


# In[125]:


lg4.summary()


# In[126]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = xtrain4.columns
vif['VIF'] = [variance_inflation_factor(xtrain4.values, i) for i in range(xtrain4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)

vif


# ## Model 5

# In[127]:


# Dropping the column 'What is your current occupation_Unemployed' as it has high vif value
xtrain5 = xtrain4.drop(['What is your current occupation_Unemployed'],axis=1)
lg5=sm.GLM(ytrain,xtrain5,family = sm.families.Binomial()).fit()


# In[128]:


lg5.summary()


# In[129]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = xtrain5.columns
vif['VIF'] = [variance_inflation_factor(xtrain5.values, i) for i in range(xtrain5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)

vif


# ## model 6

# In[130]:


# Dropping the column 'What is your current occupation_Student' as it has high p value
xtrain6 = xtrain5.drop(['What is your current occupation_Student'],axis=1)
lg6=sm.GLM(ytrain,xtrain6,family = sm.families.Binomial()).fit()


# In[131]:


lg6.summary()


# In[132]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = xtrain6.columns
vif['VIF'] = [variance_inflation_factor(xtrain6.values, i) for i in range(xtrain6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)

vif


# #### Since the Pvalues of all variables is 0.05 and VIF values are low for all the variables, model-6 is our final model. We have 15 variables in our final model.

# In[133]:


# Getting the predicted values on the train set
ytrainpred = lg6.predict(xtrain6)
ytrainpred[:10]


# In[134]:


# Reshaping into an array
ytrainpred = ytrainpred.values.reshape(-1)
ytrainpred[:10]


# In[135]:


ytrainpredfinal=pd.DataFrame({'Converted':ytrain.values, 'Converted_pred':ytrainpred})
ytrainpredfinal['Prospect ID'] = ytrain.index
ytrainpredfinal.head()


# In[136]:


ytrainpredfinal['predicted']=ytrainpredfinal.Converted_pred.map(lambda x: 1 if x > 0.5 else 0)


# In[137]:


ytrainpredfinal.head(10)


# In[138]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(ytrainpredfinal.Converted, ytrainpredfinal.predicted )
print(confusion)


# In[139]:


# The confusion matrix indicates as below
# Predicted     not_converted    converted
# Actual
# not_converted        3460      472
# converted            733       1686  


# In[140]:


# Let's check the overall accuracy.
print('Accuracy :',metrics.accuracy_score(ytrainpredfinal.Converted, ytrainpredfinal.predicted))


# In[141]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[142]:


# Sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[143]:


# Let us calculate specificity
print("Specificity : ",TN / float(TN+FP))


# In[144]:


# Calculate false postive rate - predicting converted lead when the lead actually was not converted
print("False Positive Rate :",FP/ float(TN+FP))


# In[145]:


# positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[146]:


# Negative predictive value
print ("Negative predictive value :",TN / float(TN+ FN))


# We found out that our specificity was good (~88%) but our sensitivity was only 69%. Hence, this needed to be taken care of.
# We have got sensitivity of 69% and this was mainly because of the cut-off point of 0.5 that we had arbitrarily chosen.
# 
# Now, this cut-off point had to be optimised in order to get a decent value of sensitivity and for this we will use the ROC curve.

# In[147]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[148]:


fpr, tpr, thresholds = metrics.roc_curve( ytrainpredfinal.Converted, ytrainpredfinal.Converted_pred, drop_intermediate = False )


# In[149]:


draw_roc(ytrainpredfinal.Converted, ytrainpredfinal.Converted_pred)


# ### Since we have higher (0.89) area under the ROC curve , therefore our model is a good one.
# 
# ## Finding Optimal Cutoff Point
# Above we had chosen an arbitrary cut-off value of 0.5. We need to determine the best cut-off value and the below section deals with that. Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[150]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    ytrainpredfinal[i]= ytrainpredfinal.Converted_pred.map(lambda x: 1 if x > i else 0)
ytrainpredfinal.head()


# In[151]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(ytrainpredfinal.Converted, ytrainpredfinal[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[152]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# ### From the curve above, 0.35 is the optimum point to take it as a cutoff probability

# In[153]:


ytrainpredfinal['final_predicted'] = ytrainpredfinal.Converted_pred.map( lambda x: 1 if x > 0.35 else 0)

ytrainpredfinal.head(15)


# In[154]:


ytrainpredfinal['Lead Score']=round(ytrainpredfinal['Converted_pred']*100,2)


# In[155]:


ytrainpredfinal.head()


# In[156]:


# Confusion matrix
confusion2 = metrics.confusion_matrix(ytrainpredfinal.Converted, ytrainpredfinal.final_predicted )
confusion2


# In[157]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[158]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity : ",TP / float(TP+FN))


# In[159]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[160]:


# Calculate false postive rate - predicting converted lead when the lead was actually not have converted
print("False Positive rate : ",FP/ float(TN+FP))


# In[161]:


# Positive predictive value 
print("Positive Predictive Value :",TP / float(TP+FP))


# In[162]:


# Negative predictive value
print("Negative Predictive Value : ",TN / float(TN+ FN))


# In[163]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(ytrainpredfinal.Converted, ytrainpredfinal.final_predicted))


# ## Precision and Recall
# Precision = Also known as Positive Predictive Value, it refers to the percentage of the results which are relevant.
# 
# Recall = Also known as Sensitivity , it refers to the percentage of total relevant results correctly classified by the algorithm.

# In[164]:


# Precision
TP / TP + FP

print("Precision : ",confusion[1,1]/(confusion[0,1]+confusion[1,1]))


# In[165]:


# Recall
TP / TP + FN

print("Recall :",confusion[1,1]/(confusion[1,0]+confusion[1,1]))


# In[166]:


#Using sklearn utilities for the same
from sklearn.metrics import precision_score, recall_score


# In[167]:


print("Precision :",precision_score(ytrainpredfinal.Converted , ytrainpredfinal.predicted))


# In[168]:


print("Recall :",recall_score(ytrainpredfinal.Converted , ytrainpredfinal.predicted))


# In[169]:


from sklearn.metrics import precision_recall_curve

ytrainpredfinal.Converted, ytrainpredfinal.predicted


# In[170]:


p, r, thresholds = precision_recall_curve(ytrainpredfinal.Converted, ytrainpredfinal.Converted_pred)


# In[171]:


# plotting a trade-off curve between precision and recall
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# ### **The above graph shows the trade-off between the Precision and Recall .

# ### Making predictions on the test set

# In[172]:


#Scaling the test data
xtest[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(xtest[['TotalVisits',
                                                                                                        'Total Time Spent on Website',
                                                                                                        'Page Views Per Visit']])


# In[173]:


xtest.head()


# In[174]:


xtest = xtest.drop(['What is your current occupation_Housewife','Last Notable Activity_Had a Phone Conversation','Lead Origin_Lead Add Form','What is your current occupation_Unemployed','What is your current occupation_Student'],axis=1)


# In[175]:


col = list(xtrain.columns[~rfe.support_])


# In[176]:


col


# In[177]:


xtest=xtest.drop(col,axis=1)


# In[178]:


xtest.shape


# In[179]:


# Adding a const
xtestsm = sm.add_constant(xtest)

# Making predictions on the test set
ytestpred = lg6.predict(xtestsm)
ytestpred[:10]


# In[180]:


# Converting y_test_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(ytestpred)


# In[181]:


y_pred_1


# In[182]:


# Converting ytest to dataframe
ytestdf = pd.DataFrame(ytest)


# In[183]:


ytestdf.head()


# In[184]:


# Putting Prospect ID to index
ytestdf['Prospect ID'] = ytestdf.index


# In[185]:


ytestdf.head()


# In[186]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
ytestdf.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
ytestpredfinal = pd.concat([ytestdf, y_pred_1],axis=1)


# In[187]:


ytestpredfinal.head()


# In[188]:


# Renaming the column 
ytestpredfinal= ytestpredfinal.rename(columns={ 0 : 'Converted_prob'})


# In[189]:


ytestpredfinal['final_predicted'] = ytestpredfinal.Converted_prob.map(lambda x: 1 if x > 0.34 else 0)


# In[190]:


ytestpredfinal.head()


# In[191]:


# Making the confusion matrix
confusion2 = metrics.confusion_matrix(ytestpredfinal.Converted, ytestpredfinal.final_predicted )
confusion2


# In[192]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[193]:


# Let's see the sensitivity of our logistic regression model
print("Sensitivity :",TP / float(TP+FN))


# In[194]:


# Let us calculate specificity
print("Specificity :",TN / float(TN+FP))


# In[195]:


# Let's check the overall accuracy.
print("Accuracy :",metrics.accuracy_score(ytestpredfinal.Converted, ytestpredfinal.final_predicted))


# In[196]:


#Assigning Lead Score to the Testing data
ytestpredfinal['Lead Score']=round(ytestpredfinal['Converted_prob']*100,2)


# In[197]:


ytestpredfinal.head()


# ## Observations:
# After running the model on the Test Data , we obtain:
# 
# Accuracy : 79.4 %
# Sensitivity : 81.79 %
# Specificity : 77.97 %

# ## Results :
# 1) Comparing the values obtained for Train & Test:
# ### Train Data:
# Accuracy : 80.5 %
# Sensitivity : 82.3 %
# Specificity : 79.3 %
# 
# ### Test Data:
# Accuracy : 79.4 %
# Sensitivity : 81.79 %
# Specificity : 77.97 %
# 
# Thus we have achieved our goal of getting a ballpark of the target lead conversion rate to be around 80% . The Model seems to predict the Conversion Rate very well and we should be able to give the CEO confidence in making good calls based on this model to get a higher lead conversion rate of 80%

# ### 2) Finding out the leads which should be contacted:
# The customers which should be contacted are the customers whose "Lead Score" is equal to or greater than 85. They can be termed as 'Hot Leads'.

# In[198]:


hot_leads=ytestpredfinal[ytestpredfinal["Lead Score"]>=85]
hot_leads


# So there are 411 leads which can be contacted and have a high chance of getting converted. The Prospect ID of the customers to be contacted are :

# In[199]:


print("The Prospect ID of the customers which should be contacted are :")

hot_leads_ids = hot_leads["Prospect ID"].values.reshape(-1)
hot_leads_ids


# In[200]:


## 3) Finding out the Important Features from our final model:
lg6.params.sort_values(ascending=False)


# ## Recommendation
# The company should make calls to the leads coming from the lead sources "Welingak Websites" and "Reference" as these are more likely to get converted.
# 
# The company should make calls to the leads who are the "working professionals" as they are more likely to get converted.
# 
# The company should make calls to the leads who spent "more time on the websites" as these are more likely to get converted.
# 
# The company should make calls to the leads coming from the lead sources "Olark Chat" as these are more likely to get converted.
# 
# The company should make calls to the leads whose last activity was SMS Sent as they are more likely to get converted.
# 
# The company should not make calls to the leads whose last activity was "Olark Chat Conversation" as they are not likely to get converted.
# 
# The company should not make calls to the leads whose lead origin is "Landing Page Submission" as they are not likely to get converted.
# 
# The company should not make calls to the leads whose Specialization was "Others" as they are not likely to get converted.
# 
# The company should not make calls to the leads who chose the option of "Do not Email" as "yes" as they are not likely to get converted.

# In[ ]:




