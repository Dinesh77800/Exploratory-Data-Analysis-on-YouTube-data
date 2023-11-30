#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('youtube_dislike_dataset.csv')


# In[2]:


df


# 1. Import required libraries and read the provided dataset (youtube_dislike_dataset.csv) and retrieve top
# 5 and bottom 5 records.

# In[3]:


df.head(5)


# In[4]:


# Bottom 5 in the dataset
df.tail(5)


# 2. Check the info of the dataframe and write your inferences on data types and shape of the dataset.

# In[5]:


#info of the dataframe
df.info()


# # Inferences on data types and shape of the dataset
# 
# The dataset consists of 37,422 rows and 12 columns. It comprises both numerical and categorical data types. Numerical columns, such as "view_count," "likes," "dislikes," and "comment_count," likely hold quantitative metrics. Categorical columns, including "video_id," "title," and "tags," store textual information. The "published_at" column's successful conversion to a datetime data type enhances date-related analysis. Additionally, there are missing values in the "comments" column, suggesting potential data cleaning or handling requirements.

# 3. Check for the Percentage of the missing values and drop or impute them.

# In[6]:


# Calculate missing value percentages

missing_percentages = (df.isnull().sum() / len(df)) * 100
print(missing_percentages)


# In[7]:


# imputing the missing values
# impute numerical columns with mean
num_cols = df.select_dtypes(exclude = "O").columns
df[num_cols]=df[num_cols].fillna(df[num_cols].mean())

# impute categorical columns with mode
cat_cols = df.select_dtypes(include = "O").columns
df[cat_cols]=df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Check if any missing values are left
print(df.isnull().sum())


# 4. Check the statistical summary of both numerical and categorical columns and write your inferences.

# In[8]:


# Statistical summary of both numerical and categorical columns

num_summary=df.describe(include='all')
num_summary


# 5. Convert datatype of column published_at from object to pandas datetime

# In[9]:


pd.DataFrame(pd.to_datetime(df['published_at']))

# using to_datetime function given column is converted from object to datetime type


# 6. Create a new column as 'published_month' using the column published_at (display the months only)

# In[10]:


df['published_month']=df['published_at'].str[5:7]
df[['published_month']]


# 7. Replace the numbers in the column published_month as names of the months i,e., 1 as 'Jan', 2 as 'Feb' and so on.....

# In[11]:


month={'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
month


# In[12]:


df['published_month']=df['published_month'].map(month)
df['published_month']


# 8. Find the number of videos published each month and arrange the months in a decreasing order based on the video count.

# In[13]:


pd.DataFrame(df.groupby('published_month')['video_id'].count().sort_values(ascending=False))


# 9. Find the count of unique video_id, channel_id and channel_title.

# In[14]:


len(df['video_id'].unique()),len(df['channel_id'].unique()),len(df['channel_title'].unique())


# There are 37264 unique video id
# 10891 unique channel id, 
# 10813 unique channel title

# 10. Find the top10 channel names having the highest number of videos in the dataset and the bottom 10 having lowest number of videos.

# In[15]:


pd.DataFrame(df.groupby('channel_title')['video_id'].count().sort_values(ascending=False).head(10))


# In[16]:


pd.DataFrame(df.groupby('channel_title')['video_id'].count().sort_values(ascending=False).tail(10))


# 11. Find the title of the video which has the maximum number of likes and the title of the video having
# minimum likes and write your inferences.

# In[17]:


pd.DataFrame(df.groupby('title')['likes'].max().sort_values(ascending=False).head(1))


# In[18]:


pd.DataFrame(df.groupby('title')['likes'].max().sort_values(ascending=False).tail(1))


# 12. Find the title of the video which has the maximum number of dislikes and the title of the video having
# minimum dislikes and write your inferences.

# In[19]:


pd.DataFrame(df.groupby('title')['dislikes'].max().sort_values(ascending=False).tail(1))


# In[20]:


pd.DataFrame(df.groupby('title')['dislikes'].max().sort_values(ascending=False).head(1))


# 13. Does the number of views have any effect on how many people disliked the video? Support your
# answer with a metric and a plot.
# 

# In[21]:


df['dislikes rate']=df['dislikes']/df['view_count']*100


# In[22]:


pd.DataFrame(df[['dislikes rate','view_count']]).plot(x='view_count',y='dislikes rate',kind='scatter');


# In[23]:


import seaborn as sns
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Adjust data types as needed
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm');


# Scatter plot also follows a certain trend and pattern from left to right which shows there is a correlation using dislike rate
# The correlation between views and dislikes is [0.68] which can be considered as moderate to higher correlation. So from this we can conclude that views have a signnificance in determining the dislikes of a video

# 14. Display all the information about the videos that were published in January, and mention the count of
# videos that were published in January.
# 

# In[24]:


df[df['published_month']=='Jan']


# In[25]:


df[df['published_month']=='Jan']['video_id'].count()


# In[ ]:




