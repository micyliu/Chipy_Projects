
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sk
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[28]:

hr_data = pd.read_csv("D:/OneDrive/Business Science/Chipy_HR Analytics/WA_Fn-UseC_-HR-Employee-Attrition.csv")
hr_data = hr_data.drop(columns = 'EmployeeNumber') # remove EmployeeNumbe as it does not provide value

# EmployeeNumber: to be removed
# Education, RelationshipSatisfaction,StockOptionLevel,: to be converted to factors


# In[3]:


#pd.set_option('display.max_columns', -1)


# In[46]:


hr_data.dtypes
hr_data.columns.values
hr_data.shape
#hr_data.drop_duplicates(subset=None, keep='first', inplace=False)

# In[12]:


hr_data.isnull().sum().plot(kind='bar')
# no missing data from the data set


# In[17]:


# get column names

hr_data_num = hr_data


# In[50]:


# plot all numeric variables with attrition using scatterplot and smoothingline
hr_data.select_dtypes(['object'])


# In[51]:


# plot all categorical variables with attrition using boxplot
pd.__version__


# In[18]:


# the following variables are using categorical variables and we will convert them to numerical ones:Attrition, 
# BusinessTravel, Department, EducationField
#cat_vars = hr_data.select_dtypes(include=['object']).copy()
cat_vars = ['Attrition','BusinessTravel','Department','EducationField','Gender','JobRole',
            'MaritalStatus','Over18','OverTime']
for i in cat_vars:
    labels = hr_data_num[i].astype('category').cat.categories.tolist()
    replace_map_comp = {i : {k: v for k,v in zip(labels,list(range(0,len(labels))))}}
    hr_data_num.replace(replace_map_comp, inplace=True)


# In[26]:


hr_data_num.info()


# #### organize the 33 columns into 8 different featues with Attrition

# In[27]:


# 1) demographic features
demo_data = hr_data_num[['Attrition','Age','Gender','MaritalStatus']]
pair_plot = sns.pairplot(demo_data, hue = 'Attrition')


# In[29]:


# 2) Emlpoyment based features: department, job role, job level
demo_data = hr_data_num[['Attrition','Department','JobRole','JobLevel']]
pair_plot = sns.pairplot(demo_data, hue = 'Attrition')


# In[33]:


#3) compensation features: hourly rate, monthly income, stock option level
demo_data = hr_data_num[['Attrition','HourlyRate','MonthlyIncome','StockOptionLevel']]
pair_plot = sns.pairplot(demo_data, hue = 'Attrition')


# In[37]:


# 4) Survey Results: Satisfaction level, worklife balance
demo_data = hr_data_num[['Attrition','RelationshipSatisfaction','WorkLifeBalance']]
pair_plot = sns.pairplot(demo_data, hue = 'Attrition')


# In[ ]:


pair_plot = sns.pairplot(hr_data, hue = 'Attrition')


# - **Age_**: seems to be normal distribution. The younger age(such as younger than 35 years old) seem to churn way more often that the older ages.
# - **Daily Rate**: seems to be have the same amount of attrition across different daily rates ranges.
# - **DistanceFromHome**: different from my original assumption. People seem to be more likely to churn when its less than 10 miles or more than 20 miles. Left skewed.
# - **Education**: It seems that employees who have less than a master's degree are more likely to churn. A bit right skewed.
# - **EmployeeSastifaction**: It seems that employees with low sastifaction rates are most likely to leave. Kind of uniform distribution
# - **HourlyRate**: can't tell any  trends. might need a log transformation
# - **Job involvement**: the lower the job involvement, the more likely they will leave. Normal distribution.
# - **Job Level**: the lower the job level, the more likely they will leave.
# - **Job Satistifaction**: the lower the job satisfaction, the more likely they are the leave.
# - **Monthlyincome**: the lower the monthly income, the more likely they are to churn.
# - **MonthlyRate**: monthly rate around 10,000 seem to be more likely to churn.
# - **NumCompaniesWorked**: when the number of companies worked is around 2, the employee is more likely to churn.
# - **PerformanceRating**: More people are rated on 3 versus 4. 
# - **StandardHours**: this metric seem a bit strange. it seems that all are working 80 hours.
# - **StockOptionLevel**: lower stock option level seem to have more churn. left skewed.
# - **TotalWorkingYears**: less total working years seem to have more churn. left skewed data. 
# - **TrainingTimesLastYear**: too little or the medium amount of training are more likely to churn.
# - **WorkLifeBalance**:  
# - **YearsAtCompany**: it seems that there are less years a tthe company, more likely to churn.
# - **YearsInCurrentRole**: less years in current role more likely to churn.
# - **YearsSinceLastPromotion**: less years since last promotion more likely to churn
# - **YearsWithCurrManager**: less than 5 years are more liely to churn.
# 
# covert the following variables to factors:'Department','Education','EducationField','EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction','MaritalStatus', 'Over18', 'OverTime', 'PerformanceRating', 'PerformanceRating','StockOptionLevel','WorkLifeBalance'
# 
# 
# also consider the correlation among hourly rate, monthly income, monthly rate and consider leaving only one variable
# the following variables and the relationship with churn is not very easy to view "NumCompaniesWorked","HourlyRate",'PerformanceRating', "'RelationshipSatisfaction', "WorkLifeBalance'
# *another obervation* : HR data tend to left skewed in terms of compensation etc related. In the pyramid shape,, there are usually more employees at the bottom wit less years of experiences, less pay, less stock options etc.
# 

# In[ ]:


pair_plot.savefig("pair_plot.png")


# In[ ]:


# calculate correlations among  hourly rate, monthly income, monthly rate 
correlations_hr_data_rate = hr_data[['DailyRate','HourlyRate', 'MonthlyIncome','MonthlyRate']].corr(method = 'pearson')
print(correlations_hr_data_rate)
# very low amount of correlations among the income related variables


# In[ ]:


# the following variables are using categorical variables and we will convert them to numerical ones:Attrition, 
# BusinessTravel, Department, EducationField

#labels,unique = pd.factorize(hr_data.Attrition)
#hr_data.Attrition = labels
#hr_data.Attrition.head()


# In[ ]:


cat_hr_data = hr_data.select_dtypes(include = ['object'].copy())
cat_hr_data.head()


# In[ ]:


BusinessTravel_count = cat_hr_data['BusinessTravel'].value_counts()
sns.barplot(BusinessTravel_count.index, BusinessTravel_count.values,alpha = 0.9)
plt.title('Frequency Distribution of Business Travel')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('Business Travel Type', fontsize = 12)
plt.show()


# In[ ]:


BusinessTravel_count = cat_hr_data['Attrition'].value_counts()
sns.barplot(BusinessTravel_count.index, BusinessTravel_count.values,alpha = 0.9)
plt.title('Frequency Distribution of Attrition')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('Attrition Type', fontsize = 12)
plt.show()


# In[ ]:


Department_count = cat_hr_data['Department'].value_counts()
sns.barplot(Department_count.index, Department_count.values,alpha = 0.9)
plt.title('Frequency Distribution of Department')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('Department Type', fontsize = 12)
plt.show()


# In[ ]:


EducationField_count = cat_hr_data['EducationField'].value_counts()
sns.barplot(EducationField_count.index, EducationField_count.values,alpha = 0.9)
plt.title('Frequency Distribution of EducationField')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('EducationFieldl Type', fontsize = 12)
plt.show()


# In[ ]:


Gender_count = cat_hr_data['Gender'].value_counts()
sns.barplot(Gender_count.index, Gender_count.values,alpha = 0.9)
plt.title('Frequency Distribution of Gender')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('Gender Type', fontsize = 12)
plt.show()


# In[ ]:


JobRole_count = cat_hr_data['JobRole'].value_counts()
fig, ax = plt.subplots(figsize=(18,4))
grid = sns.barplot(JobRole_count.index, JobRole_count.values,alpha = 0.9)
plt.title('Frequency Distribution of JobRole')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('JobRole Type', fontsize = 12)

plt.show()


# In[ ]:


MaritalStatus_count = cat_hr_data['MaritalStatus'].value_counts()
sns.barplot(MaritalStatus_count.index, MaritalStatus_count.values,alpha = 0.9)
plt.title('Frequency Distribution of MaritalStatus')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('MaritalStatus Type', fontsize = 12)
plt.show()


# In[ ]:


Over18_count = cat_hr_data['Over18'].value_counts()
sns.barplot(Over18_count.index, Over18_count.values,alpha = 0.9)
plt.title('Frequency Distribution of Over18')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('Over18 Type', fontsize = 12)
plt.show()

# there is almost no variance in this variable. this variabel could be removed.


# In[ ]:


OverTime_count = cat_hr_data['OverTime'].value_counts()
sns.barplot(OverTime_count.index, OverTime_count.values,alpha = 0.9)
plt.title('Frequency Distribution of OverTime')
plt.ylabel('Number of Occurrences', fontsize = 12)
plt.xlabel('OverTime Type', fontsize = 12)
plt.show()


# In[ ]:


# understand feature importance

