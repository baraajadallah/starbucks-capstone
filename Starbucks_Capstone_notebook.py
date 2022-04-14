#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[2]:


import pandas as pd
import numpy as np
import math
import json
get_ipython().run_line_magic('matplotlib', 'inline')

# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import matplotlib.ticker as ticker


# In[5]:


import seaborn as sns


# In[6]:


from sklearn.preprocessing import MultiLabelBinarizer


# In[7]:


import re


# In[8]:


import progressbar


# In[ ]:





# In[9]:


portfolio.head()


# In[10]:


print(portfolio.shape)
portfolio.info()


# In[11]:


profile.head()


# In[12]:


print(profile.shape)
profile.info()


# In[13]:


transcript.head()


# In[14]:


print(transcript.shape)
transcript.info()


# In the below cells, I well start cleaning the data in which it serves my analysys in the nest way.

# ## Cleaning Data
# 
# ### portfolio dataframe
# 
# There are multiple point I wish to address on this datafrme:
# 
# 1- Change the name of the 'id' column to 'offer_id'
# 
# 2- One hot encode the 'offer_type' column
# 
# 3- One hot encode the 'channels' columns

# In[15]:


portfolio.rename(columns={'id': 'offer_id'}, inplace=True)


# In[16]:


offertype_df = pd.get_dummies(portfolio['offer_type'])


# In[17]:


offertype_df.head()


# In[18]:


portfolio = pd.concat([portfolio, offertype_df], axis=1)


# In[19]:


portfolio.head()


# In[20]:


mlb = MultiLabelBinarizer()

channels_df = pd.DataFrame(mlb.fit_transform(portfolio['channels']),columns=mlb.classes_, index=portfolio.index)


# In[21]:


channels_df.head()


# In[22]:


portfolio = pd.concat([portfolio, channels_df], axis=1)


# In[23]:


portfolio = portfolio.drop('channels', axis=1)


# In[24]:


portfolio.head()


# ## Cleaning Data
# ### profile dataframe
# 
# There are multiple point I wish to address on this datafrme:
# 
# 1- Rename id col name to customer_id.
# 
# 2- drop rows with no gender, income or age data
# 
# 3- Transform the 'became_member_on' column to a datetime object
# 
# 4- One hot encode a customer's age range
# 
# 5- Transform a customer's gender from a character to a number

# In[25]:


profile.duplicated().sum()


# In[26]:


profile.describe()


# In[27]:


profile.rename(columns={'id': 'customer_id'}, inplace=True)


# In[28]:


profile.became_member_on = profile['became_member_on'].astype(str).astype('datetime64[ns]', format = "%Y%m%d")


# In[29]:


profile.isna().sum()


# In[30]:


profile = profile.dropna(axis=0, how='any')


# In[31]:


profile.isna().sum()


# In[32]:


min_age_limit = np.int(np.floor(np.min(profile['age'])/10)*10)
max_age_limit = np.int(np.ceil(np.max(profile['age'])/10)*10)

profile['agerange'] = pd.cut(profile['age'], (range(min_age_limit,max_age_limit + 10, 10)), right=False)

profile['agerange'] = profile['agerange'].astype('str')

agerange_df = pd.get_dummies(profile['agerange'])


# In[33]:


agerange_df.head()


# In[34]:


# Re-arrangingin the dataframe
agerange_df = agerange_df[['[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '[60, 70)', '[70, 80)', '[80, 90)', '[90, 100)', '[100, 110)']]


# In[35]:


agerange_df.head()


# In[36]:


(profile.gender == 'O').mean()


# In[37]:


profile.gender.unique()


# In[38]:


# Will drop the rows in profile dataframe that are neither 'M' nor 'F'
a = ['M', 'F']
profile = profile[profile['gender'].isin(a)]


# In[39]:


profile.gender.unique()


# In[40]:


gender = {'M': 1,'F': 0}

profile.gender = [gender[item] for item in profile.gender]


# In[41]:


profile.head()


# In[42]:


# Dropping individuals in age range 100-110 because they represent a very small segemnt and won't affect the analysis
profile = profile[profile.agerange != '[100, 110)']


# In[43]:


profile.agerange.unique()


# In[44]:


# Checking for duplicates
profile.duplicated().any()


# In[45]:


transcript.head()


# ## Cleaning Data
# ### transcript dataframe
# 
# There are multiple point I wish to address on this datafrme:
# 
# 1- Change the name of the 'person' column to 'customerid'
# 
# 2- Remove customer id's that are not in the customer profile DataFrame
# 
# 3- Convert time variable units from hours to days
# 
# 4- Separate the dictionaries in 'value' column into two columns ('offerid'  and 'amount')
# 
# 5- Pull a transactions dataframe and offers_stats dataframe

# In[46]:


transcript.event.unique()


# In[47]:


transcript.rename(columns={'person': 'customer_id'}, inplace=True)


# In[48]:


transcript = transcript[transcript.customer_id.isin(profile.customer_id)]


# In[49]:


transcript['time'] /= 24.0


# In[50]:


# Here I will separate the dictionaries in 'value' column into two columns 
transcript = pd.concat([transcript, transcript['value'].apply(pd.Series)], axis = 1).drop('value', axis = 1)


# In[51]:


transcript['offerid'] = transcript[['offer_id', 'offer id']].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)


# In[52]:


transcript = transcript.drop(['offer_id', 'offer id'], axis=1)


# In[53]:


transcript.sample(20)


# In[54]:


transcript.duplicated().sum()


# In[55]:


transcript.drop_duplicates(keep='first', inplace=True)


# In[56]:


transaction_df = transcript[transcript.event=='transaction'].copy()
transaction_df.drop(columns=['offerid', 'event'], inplace=True)
transaction_df


# In[57]:


offers_df = transcript[transcript.event!='transaction'].copy()

# one-hot encode offer event
offers_df['received'] = offers_df.event.apply(lambda x: 1 if x == 'offer received' else 0)
offers_df['completed'] = offers_df.event.apply(lambda x: 1 if x == 'offer completed' else 0)
offers_df['viewed'] = offers_df.event.apply(lambda x: 1 if x == 'offer viewed' else 0)

offers_df.drop(columns=['event', 'amount'], inplace=True)


# In[58]:


offers_df


# ## Exploratory Data Analysis

# In[59]:


profile.head()


# In[60]:


transcript.head()


# In[61]:


transaction_df.head()


# In[62]:


offers_df.head()


# ### Compute what percent of customers are in each age range

# In[63]:


ages_count = profile['agerange'].value_counts()

ages_count *= 100 / ages_count.sum()
ages_count


# In[64]:


ages_array = profile.loc[:, 'agerange']


# In[65]:


labels, counts = np.unique(ages_array, return_counts=True)


# ### Plotting age range distribution and income destribution of all customers

# In[66]:


fig, ax = plt.subplots(figsize=(15, 4), nrows=1, ncols=2)

plt.sca(ax[0])
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution')
plt.xticks(rotation=90)

plt.sca(ax[1])
plt.hist(profile['income'] * 1E-3 )
plt.xlabel('Income [10K]')
plt.ylabel('Count')
plt.title('Income Distribution');


# Age distribution plot depicts that the median age range of a customer is 50-60 and most of our customers belong to age range between 40 to 70. Income distribution plot shows that the number of customers whose average salary is less than 70K is high than the other side considering 70K to be median of the income distribution. 

# ### Plotting income differences as a function of gender

# In[67]:


current_palette = sns.color_palette()
# extract male and female customers
male_customers = profile[profile['gender'] == 0]
female_customers = profile[profile['gender'] == 1]

# to be able to draw two subplots in a row
fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=2, sharex=True, sharey=True)

# plot a male customers income distribution
plt.sca(ax[0])
sns.distplot(male_customers['income'] * 1E-3)
plt.xlabel('Income [10K]')
plt.ylabel('P(Income)')
plt.title('Male Customer Income')

# plot a female customers income distribution
plt.sca(ax[1])
sns.distplot(female_customers['income'] * 1E-3, color=current_palette[1])
plt.xlabel('Income [10K]')
plt.ylabel('P(Income)')
plt.title('Female Customer Income');


# plots conclude that minimum and maximum income for both male and female are approximately same but the count of male customers in low-income level is slightly higher than that of female customers

# ### Plotting age ranges as a function of gender

# In[68]:


# groupby start_year and gender to plot a graph
age_range = profile.groupby(['agerange', 'gender']).size()
age_range = age_range.reset_index()
age_range.columns = ['agerange', 'gender', 'count']

# plot a bar graph for age distribution as a function of gender in membership program
plt.figure(figsize=(10, 5))
sns.barplot(x='agerange', y='count', hue='gender', data=age_range)
plt.xlabel('Age')
plt.ylabel('Count');


# The above bar plot shows that the males (0) have more presense than females (1) across most of the age range categories

# ### Combine offer, profile and transactions dataframes
# 
# Next, I will combine 3 dataframes applying the following steps:
# 
# 1-Select a customer's profile
# 
# 2-Select offer data for a specific customer
# 
# 3-Select transactions for a specific customer
# 
# 4-Initialize DataFrames that describe when a customer receives, views, and completes an offer
# 
# 5-Iterate over each offer a customer receives
# 
# For each offer id, our main goal is to find out if the offer is successful by deciding if a customer reacts to an offer within the offer validity time period

# In[69]:


def combine_data(profile, portfolio, offers_df, transaction_df):
    data = []
    customer_ids = offers_df['customer_id'].unique()
    
    widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    
    # loop through all customer ids in offers_df
    for ind in progressbar.progressbar(range(len(customer_ids)), widgets=widgets):
        
        # get customer id from the list
        cust_id = customer_ids[ind]
        
        # extract customer profile from profile data
        customer = profile[profile['customer_id']==cust_id]
        
        # extract offers associated with the customer from offers_df
        cust_offer_data = offers_df[offers_df['customer_id']==cust_id]
        
        # extract transactions associated with the customer from transactions_df
        cust_transaction_data = transaction_df[transaction_df['customer_id']==cust_id]
        
        # extract received, completed, viewed offer data from customer offers
        offer_received_data = cust_offer_data[cust_offer_data['received'] == 1]
        offer_completed_data = cust_offer_data[cust_offer_data['completed'] == 1]
        offer_viewed_data = cust_offer_data[cust_offer_data['viewed'] == 1]
        
        rows = []
        
        for i in range(offer_received_data.shape[0]):
            
            # fetch an offer id 
            offer_id = offer_received_data.iloc[i]['offerid']
            
            # extract offer row from portfolio
            offer_row = portfolio.loc[portfolio['offer_id'] == offer_id]
            
            # extract duration days of an offer from offer row
            duration_days = offer_row['duration'].values[0]
            
            # initialize start and end time of an offer
            start_time = offer_received_data.iloc[i]['time']
            end_time = start_time + duration_days
        
            # seggregate offers completed within end date
            off_completed_withintime = np.logical_and(
                offer_completed_data['time'] >= start_time, offer_completed_data['time'] <= end_time)
            
            # seggregate offers viewed within end date
            off_viewed_withintime = np.logical_and(
                offer_viewed_data['time'] >= start_time, offer_viewed_data['time'] <=end_time)

            # flag offer_successful to 1 if an offer is viewed and completed within end time else to 0
            offer_successful = off_completed_withintime.sum() > 0 and off_viewed_withintime.sum() > 0
            
            # extract transactions occured within time
            transaction_withintime = np.logical_and(
                cust_transaction_data['time'] >= start_time, cust_transaction_data['time'] <= end_time)
            
            transaction_data = cust_transaction_data[transaction_withintime]
            
            # total amount spent by a customer from given offers
            transaction_total_amount = transaction_data['amount'].sum()
            
            row = {
                'offer_id': offer_id,
                'customer_id': cust_id,
                'time': start_time,
                'total_amount': transaction_total_amount,
                'offer_successful': int(offer_successful),
            }
                
            row.update(offer_row.iloc[0,0:].to_dict())

            row.update(customer.iloc[0,:].to_dict())

            rows.append(row)
        
        data.extend(rows)
    
    data = pd.DataFrame(data)
    return data


# In[70]:


data = combine_data(profile, portfolio, offers_df, transaction_df)


# In[71]:


data.columns


# In[72]:


column_ordering = ['time', 'offer_id', 'customer_id', 'total_amount',
                       'offer_successful', 'difficulty', 'duration', 'offer_type',
                       'reward', 'bogo', 'discount', 'informational',
                       'email', 'mobile', 'social', 'web', 'gender', 'agerange',
                       'income', 'became_member_on', ]
data = data[column_ordering]


# In[73]:


data.head()


# In[74]:


data.shape


# In[75]:


data.to_csv('master_offer_analysis.csv', index=False)


# In[76]:


data = pd.read_csv('master_offer_analysis.csv')


# In[77]:


def calculate_percentage_success():
    successful_count = data[['offer_id', 'offer_successful']].groupby(
        'offer_id').sum().reset_index()

    offer_count = data['offer_id'].value_counts()

    offer_count = pd.DataFrame(list(zip(offer_count.index.values,
                                        offer_count.values)),
                               columns=['offer_id', 'count'])

    successful_count = successful_count.sort_values('offer_id')

    offer_count = offer_count.sort_values('offer_id')

    percent_success = pd.merge(offer_count, successful_count, on="offer_id")

    percent_success['percent_success'] = (
        100 * percent_success['offer_successful'] / percent_success['count'])

    percent_success = pd.merge(percent_success, portfolio, on="offer_id")

    percent_success = percent_success.drop(columns=['offer_successful'])

    percent_success = percent_success.sort_values('percent_success', ascending=False)

    return percent_success.reset_index(drop=True)


# In[78]:


percent_success = calculate_percentage_success()
percent_success


# In[79]:


percent_success.groupby('offer_type')['percent_success'].mean().plot(kind='bar')
plt.title('Success Percentage VS Offer Type')
plt.ylabel('Percentage')


# We notice from the above bar plot that the "discount" offer type gets the highest success percentage in terms of customers reactions to promotions. In the below bar plot, I illustrate how the duration of the promotion affects the it's success, it is only notable in the "buy one get one" offer type where shorter durations lead to more success.

# In[80]:


percent_success.groupby(['duration', 'offer_type'])['percent_success'].mean().plot(kind='bar')
plt.title('Success Percentage VS offer Type and Duration')
plt.ylabel('Percentage')


# In[81]:


gender_age = data.groupby(['agerange', 'gender'])['offer_successful'].mean()


# In[82]:


fig, ax = plt.subplots(figsize=(12,4))

plt.subplot(1,2,1)
data.groupby('agerange')['offer_successful'].mean().plot(kind='bar')
plt.title('Success Rate VS Age')
plt.ylabel('Success mean')

plt.subplot(1,2,2)
data.groupby('gender')['offer_successful'].mean().plot(kind='bar')
plt.title('Success Rate VS Gender')
plt.ylabel('Success mean')


# From the above bar plots, we notice that customers of ages between (80-90) and (50-60) are the most reacting to various types of offers, and offers are showing more success when they are sent to females rather than males

# In[83]:


data.columns


# In[84]:


col = ['offer_type', 'time', 'gender', 'income', 'agerange', 'offer_successful']
sns.set(style="ticks", color_codes=True)
sns.pairplot(data[col].dropna())
plt.show();


# ## Building models

# In[85]:


from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[86]:


from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[87]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[88]:


data.columns


# Having all the features related to offer id and customer id, we can drop these columns withput having as affect on our models that we are going to build

# In[89]:


data_model = data.drop(['offer_id', 'customer_id', 'offer_type'], axis=1)


# In[90]:


data.select_dtypes(include=['object']).copy().columns


# In[91]:


# Will produce dummy variables od the 'agerange' column
data_model = pd.concat([data_model.drop('agerange', axis=1), pd.get_dummies(data_model['agerange'], drop_first=True)], axis=1)


# In[92]:


# Will edit the 'became_member_on' column to contain only the year
data_model['became_member_on']=pd.to_datetime(data_model['became_member_on'], format='%Y-%m-%d')


# In[93]:


data_model['became_member_on'] = data_model['became_member_on'].dt.year


# In[94]:


data_model.head()


# In[95]:


features = data_model.drop(columns=['offer_successful'])

label = data_model.filter(['offer_successful'])

X_train, X_test, y_train, y_test = train_test_split(features.values, label.values, test_size=0.3, random_state=42)

# convert train and test labels to array
y_train = y_train.ravel()
y_test = y_test.ravel()


# In[96]:


scorer = make_scorer(fbeta_score, beta=0.5)

# instantiate a logistic regression classifer object
lr_clf = LogisticRegression(random_state=42, solver='liblinear')

# construct a params dict to tune the model
grid_params = {
    'penalty': ['l1', 'l2'],
    'C': [1.0, 0.1, 0.01]}

lr_random = RandomizedSearchCV(
    estimator = lr_clf, param_distributions = grid_params, 
    scoring=scorer, n_iter = 6, cv = 3, verbose=2, 
    random_state=42, n_jobs = 3)

# fit train data to the model
lr_random.fit(X_train, y_train)


# In[97]:


def evaluate(clf, X_train, y_train):
    class_name = re.sub("[<>']", '', str(clf.__class__))
    class_name = class_name.split(' ')[1]
    class_name = class_name.split('.')[-1]

    y_pred_rf = clf.predict(X_train)

    clf_accuracy = accuracy_score(y_train, y_pred_rf)
    clf_f1_score = f1_score(y_train, y_pred_rf)
    
    print("%s model accuracy: %.3f" % (class_name, clf_accuracy))
    print("%s model f1-score: %.3f" % (class_name, clf_f1_score))
    
    return clf_accuracy, clf_f1_score


# ### Evauating performance on training data

# In[98]:


evaluate(lr_random, X_train, y_train)


# ### Evaluating performance on test data

# In[99]:


evaluate(lr_random, X_test, y_test)


# The logistic regression classifier perfromance seems pretty good, but I would also try implementing RandomForestClassifier on our data and tune it using RandomizedSearchCV and see how it performs

# In[100]:


rf_clf = RandomForestClassifier(random_state=42)

# Number of trees in random forest
n_estimators = [10, 50, 100, 150, 200, 250, 300]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.arange(3, 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Create the random grid
grid_params = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# tune the classifer
rf_random = RandomizedSearchCV(estimator = rf_clf,
                               param_distributions = grid_params,
                               scoring=scorer,
                               n_iter = 100,
                               cv = 3,
                               verbose=2,
                               random_state=42,
                               n_jobs = 3)

# fit train data to the classifier
rf_random.fit(X_train, y_train)


# ### Evaluating performance on training data

# In[101]:


evaluate(rf_random.best_estimator_, X_train, y_train)


# ### Evaluating performance on test data

# In[102]:


evaluate(rf_random.best_estimator_, X_test, y_test)


# ## Conclusion
# 
# The problem that I tried to address was to build a model that predicts whether a customer will respond to an offer (offer being successful) based on multiple features of the offers themselves and the customers that are contacted. My strategy for solving this problem has mainly two steps. First, I combined offer portfolio, customer profile, and transaction data. Second, I assessed the accuracy and F1-score of a logistic regression model. Third, I compared the performance of logistic regression and random forest models. This analysis suggests that a random forest model has the best training data accuracy and F1-score. Analysis suggests that random forest model has a training data accuracy of 0.94 and an F1-score of 0.93. The test data set accuracy of 0.92 and F1-score of 0.915 suggests that the random forest model I constructed did not overfit the training data.
# 
# Better predictions may have been deducted if there were more customer metrics. For this analysis, I feel we had limited information about customer available to us — just age, gender, and income. To find optimal customer demographics, it would be nice to have a few more features of a customer. These additional features may aid in providing better classification model results.

# In[ ]:




