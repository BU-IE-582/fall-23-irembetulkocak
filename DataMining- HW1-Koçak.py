#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv ('/Users/irembetulkocak/Desktop/datamininghw1/all_ticks_wide.csv')
df.head()


# In[3]:


df.dropna()


# In[4]:


stock_names = df.columns
print (stock_names)


# In[5]:


pd.set_option('display.max_columns', None)
print(df.describe())


# Data type for timestamp is not datetime so we adjust it to become datetime data. 

# In[6]:


df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)


# In[7]:


print(df.index.dtype)


# # Descriptive Analysis

# Since we know the 5-number summary of each stock price, we will look at how their prices change over time. The line chart below, while too complicated and dirty, shows us that most stocks started from a real low point and continue that way until the end of our time. So I will look at the central tendency graph and pick 5 outstanding stocks and 5 stocks with lower prices.

# In[8]:


plt.figure(figsize=(12, 6))
for stock_name in df.columns:
    plt.plot(df.index, df[stock_name], label=stock_name, linewidth=0.5)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price Over Time')
plt.show()


# In[9]:


plt.figure(figsize=(20, 8))
df.mean().plot(label='Mean', marker='o')
df.median().plot(label='Median', marker='x')
df.mode().iloc[0].plot(label='Mode', marker='s')
plt.xlabel('Stocks')
plt.ylabel('Price')
plt.title('Central Tendency of Stock Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xticks(ticks = range(len(df.columns)), labels = df.columns, rotation=90) 
plt.show()


# From the graph above, we can observe that CCOLA, FROTO, OTKAR, PGSUS, and TUPRS have the highest means, modes, and medians. We've also selected five stocks with lower measurements that may reveal interesting price changes over time: AEFES, ARCLK, HALKB, THYAO, and TCELL.
# 
# To display all the stock names in this plot, I had to rotate them, a task I hadn't done before, making it quite an achievement for me.

# In[10]:


plt.figure(figsize=(20, 6))
plt.plot (df.index, df ['TUPRS'], linewidth = 0.75, label = 'TUPRS')
plt.plot (df.index, df ['OTKAR'], linewidth = 0.75, label = 'OTKAR')
plt.plot (df.index, df ['CCOLA'], linewidth = 0.75, label = 'CCOLA')
plt.plot (df.index, df ['FROTO'], linewidth = 0.75, label = 'FROTO')
plt.plot (df.index, df ['PGSUS'], linewidth = 0.75, label = 'PGSUS')
plt.xlabel('Date') 
plt.ylabel('Stock Price') 
plt.title ('Stock Price Comparison Over Time (H)')
plt.legend()
plt.grid(True)
plt.show


# We notice that OTKAR experiences more price spikes over time, while TUPRS shows a consistent upward price trend. Despite the higher mean and median of OTKAR, TUPRS appears to be a more consistent stock for investment.
# 
# FROTO follows a similar upward trend to TUPRS, as both companies are part of the same group. PGSUS went public in 2013 with a strong start but faced a loss of consistent growth in 2014, reaching its highest point in 2016. However, by the end of 2019, its price began to increase again. It would be interesting to monitor the price changes in 2020, considering the impact of the pandemic. Lastly, CCOLA doesn't exhibit any particularly remarkable trend; it has fluctuations but, as a part of a global brand, it is less affected by governmental policies compared to others.

# In[11]:


plt.figure(figsize=(20, 6))
plt.plot (df.index, df ['AEFES'], linewidth = 0.75, label = 'AEFES')
plt.plot (df.index, df ['ARCLK'], linewidth = 0.75, label = 'ARCLK')
plt.plot (df.index, df ['HALKB'], linewidth = 0.75, label = 'HALKB')
plt.plot (df.index, df ['THYAO'], linewidth = 0.75, label = 'THYO')
plt.plot (df.index, df ['TCELL'], linewidth = 0.75, label = 'TCELL')
plt.xlabel('Date') 
plt.ylabel('Stock Price') 
plt.title ('Stock Price Comparison Over Time (L)')
plt.legend()
plt.grid(True)
plt.show


# "We can observe that there was an increase in the prices of all stocks in the year 2017. In particular, ARCLK experienced a substantial increase, reaching its highest point in years, although it couldn't sustain this growth. On the other hand, AEFES temporarily lost its upward trend between 2016 and 2018, reaching its highest point in 2018. Meanwhile, THYAO showed a remarkable increase in 2017.
# 
# From the correlation matrix below, we can see that AEFES is one of the rare stocks that displays a positive correlation with HALKBK.

# In[12]:


correlation_matrix = df.corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5,  mask=mask)
plt.title('Correlation Matrix of Stock Prices')
plt.show()


# Before we proceed to the pairwise comparison, we will examine the correlation between the stocks. I omit the upper triangle in this graph to have a more clear image since the number of stocks is already high.
# 
# We can observe a negative correlation, whether it's high or low, between HALKB and nearly all other stocks, with a particularly high one with TUPRS. Given the increasing trend in the price of TUPRS, it's likely that HALKB is experiencing a decreasing trend in its price.
# 
# Finally, we notice that EREGL and FROTO generally exhibit high correlations with almost all other stocks and with each other. To gain a better understanding, we also examine the time graph of these four stocks in a single graph.

# In[13]:


plt.figure(figsize=(20, 6))
plt.plot (df.index, df ['TUPRS'], linewidth = 0.75, label = 'TUPRS')
plt.plot (df.index, df ['HALKB'], linewidth = 0.75, label = 'HALKB')
plt.plot (df.index, df ['FROTO'], linewidth = 0.75, label = 'FROTO')
plt.plot (df.index, df ['EREGL'], linewidth = 0.75, label = 'EREGL')
plt.plot (df.index, df ['SODA'], linewidth = 0.75, label = 'SODA')
plt.xlabel('Date') 
plt.ylabel('Stock Price') 
plt.title ('Stock Price Comparison Over Time (Cor1)')
plt.legend()
plt.grid(True)
plt.show


# We can clearly see that in the year of 2017, all stock prices are increasing and contunie this increase after 2018, as well. However, HALKB start to decrease after the half of the 2017.
# 
# So, I will choose to analyze two stocks based on my initial instincts: TUPRS and OTKAR. I'll also examine two negatively correlated stocks, SODA and HALKB. Lastly, I'll focus on two similar stocks, THYAO and ARCLK, which exhibit similar trends. I will also look at the two most expensive stocks of KOÇ Group: TUPRS and FROTO. 

# # Moving Window Correlation

# In[14]:


stock1 = df['TUPRS']
stock2 = df['OTKAR']

window_size = 90 #3-months window

rolling_corr = stock1.rolling(window=window_size).corr(stock2)
rolling_corr = rolling_corr.dropna()
plt.figure(figsize=(20, 6))
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Moving Window Correlation Between TUPRS and OTKAR')
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(y=0, color='red', linestyle='--') #zero correlation line

plt.show()


# In[15]:


stock3 = df['SODA']
stock4 = df['HALKB']

window_size = 90 

rolling_corr = stock3.rolling(window=window_size).corr(stock4)
rolling_corr = rolling_corr.dropna()
plt.figure(figsize=(20, 6))
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Moving Window Correlation Between SODA and HALKB')
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(y=0, color='red', linestyle='--') #zero correlation line

plt.show()


# In[16]:


stock5 = df['THYAO']
stock6 = df['ARCLK']

window_size = 90 

rolling_corr = stock5.rolling(window=window_size).corr(stock6)
rolling_corr = rolling_corr.dropna()
plt.figure(figsize=(20, 6))
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Moving Window Correlation Between THYAO and ARCLK')
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(y=0, color='red', linestyle='--') #zero correlation line

plt.show()


# In[17]:


stock7 = df['TUPRS']
stock8 = df['FROTO']

window_size = 90 

rolling_corr = stock7.rolling(window=window_size).corr(stock8)
rolling_corr = rolling_corr.dropna()
plt.figure(figsize=(20, 6))
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Moving Window Correlation Between TUPRS and FROTO (3-month)')
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(y=0, color='red', linestyle='--') #zero correlation line

plt.show()


# In[18]:


window_size = 180 

rolling_corr = stock7.rolling(window=window_size).corr(stock8)
rolling_corr = rolling_corr.dropna()
plt.figure(figsize=(20, 6))
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.title('Moving Window Correlation Between TUPRS and FROTO (6-month)')
plt.plot(rolling_corr.index, rolling_corr)
plt.axhline(y=0, color='red', linestyle='--') #zero correlation line

plt.show()


# ## Moving Window Correlation Discussion
# 
# I adjusted the window size from one month to one year in the graphs and found that using a one-month window makes it difficult to interpret the values. On the other hand, extending the window to six months or a year results in the loss of many changes over time. So, I decided to stick with the 3-month window size.
# 
# Looking at how correlation change over time and observing the dramatic changes, we can say that the correlation matrix provided above can be deceptive. For instance, we initially observed a strong negative correlation between HALKB and SODA. However, when using Moving Window Correlation, we discovered that between the second half of 2014 and 2016, there was a positive correlation between the prices of these stocks. After 2018, a negative correlation emerged, but it's important to note that we can't make general conclusions about an overall negative correlation between these two stock prices.
# 
# We know there is an increasing trend in all stock prices between 2014 and 2017, leading to mainly positive correlations during those years in any pair of stocks. However, in 2018, the devaluation of the Turkish Lira against the USD began, and this had varying effects on each stock over different time periods. This currency depreciation, coupled with the unbalanced state of the markets, led to dramatic ups and downs in some stocks, particularly in THYAO. Therefore, we can attribute the numerous correlation changes in 2018 and 2019 to these factors.
# 
# Finally, I examine the correlation between FROTO and TUPRS to understand how two companies from the same group react to currency devaluation. To provide a broader perspective and demonstrate that a 6-month period can be misleading at first, I use two different window sizes to analyze the stock prices of these two companies.
# 
# It becomes evident that when we employ the 6-month window size, some negative movements in these two stock prices may be overlooked. However, it also allows us to see that there is generally a strong correlation between these two stocks, a pattern that persists even after the events of the year 2018.

# # Principal Component Analysis

# In[19]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[20]:


scaler = StandardScaler() # First we scale the data.
data_scaled = scaler.fit_transform(df)


# In[21]:


from sklearn.impute import SimpleImputer # I had the error that input has missing NaN data so I did imputer method to get rid of the missing data.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data_scaled_imputed = imp.fit_transform(data_scaled) 


# In[65]:


n_components = 4 # I adjust this number by doing experiments between 2 and 6. Keep 4. ( > 82% explained variation)
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(data_scaled_imputed) 
eigenvalues = pca.explained_variance_
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index = df.columns)


# In[34]:


print (pca.explained_variance_ratio_ )


# We see that PC1 explains nearly half of the variance. This ratio decrease to %5 in PC4, so we can neglect it because 3 components can explain over 75% However, I would like to keep it since it does not make complex the model and increase the explained variance by 5 points.

# In[27]:


covariance_matrix = np.dot(loadings, loadings.T)
covariance_matrix = covariance_matrix * eigenvalues


# In[35]:


plt.figure(figsize=(5, 4))
sns.heatmap(covariance_matrix, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)
plt.title('Covariance Matrix Heatmap')
plt.show()


# Covariances of all pairs of different components are zero, which tells us that these components are orthogonal to each other and captures different aspects of the data. It shows that PCA is working correctly on the data. 
# 
# The diagonal values provide us with the eigenvalues for the principal components. The sum of the diagonal values gives us the overall variability. It's worth noting that PC1, represented as '0' in the heatmap, explains nearly half of the total variance, which was already obvious from explained_variance_ratio. 
# 
# 

# In[66]:


print(loadings)


# Finally, we will look at the loadings. We can observe that each principal component has loadings from every stock, with some being negative and others positive. When we focus on PC1, we notice that all stocks have negative loadings, except for TTKOM, SKBNK, GUBRF, CCOLA, BAGFS, and HALKB. The magnitudes of these loadings are generally greater than 10%. It's interesting to note that all components include all features from the data, so we cannot assert the dominance of any specific feature.
# 
# We can refer to these components as latent variables since they can account for over 80% of the variance, which is a substantial amount. Initially, we had 60 observable variables, and now we have reduced them to just 4 latent variables (principal components) while explaining over 80% of the variance. 

# # Google Trends

# ## OTKAR & TUPRS
# 

# ![OTKAR%20TUPRS%20Google%20Trend.png](attachment:OTKAR%20TUPRS%20Google%20Trend.png)

# People often search for these companies using their formal names, not with their stock names, and the most commonly searched word with these formal names is 'investing'. So, I choose to search them in Google Trends with their formal names since there is no results for IST: OTKAR and does not show anything under this name. 
# 
# However, the search trends do not align with their stock prices. For instance, TUPRS is more popular in Google Trends, despite having a significantly lower price than OTKAR. It shows that sometimes popularity of something in Google does not align with the value of that thing. 

# ## HALKB

# In this case, there are more results for HALKBANK in Google Trends compared to IST:HALKB. However, we need to consider that Halkbank has faced a negative public perception, particularly in the year 2018, which could potentially distort the data we are examining. For this reason, I choose to focus only on IST:HALKB.
# 
# When we examine the trend, we observe that the decreasing trend in Google Trend data is more dramatic than in the stock prices, but it generally aligns with them. The first noticeable declining trend appears in the stock prices is in 2016, and it is followed by a sharp decrease in the Google Trends search results.
# 
# ![IST%20HALKB%20Google%20Trends.png](attachment:IST%20HALKB%20Google%20Trends.png)

# ## THYAO
# 
# I excluded the THY and Turkish Airlines keywords because they are most closely associated with terms like 'flight number' or 'airport.' However, with IST: THYAO, it is primarily searched alongside ASELS, TUPRS, and EREGL. As a result, I've decided to analyze them together in the same graph and will also include the stock price over time graph for these three. I excluded TUPRS since this stock has a solid domination in terms of popularity among all and make it harder to understand the relationship between others in the graph.
# 
# 

# ![THYAO%20ASELS.png](attachment:THYAO%20ASELS.png)

# In[37]:


plt.figure(figsize=(20, 6))
plt.plot (df.index, df ['THYAO'], linewidth = 0.75, label = 'THYAO')
plt.plot (df.index, df ['ASELS'], linewidth = 0.75, label = 'ASELS')
plt.plot (df.index, df ['EREGL'], linewidth = 0.75, label = 'EREGL')
plt.xlabel('Date') 
plt.ylabel('Stock Price') 
plt.title ('Stock Price Comparison Over Time')
plt.legend()
plt.grid(True)
plt.show


# Between 2014 and 2015, there was no notable search activity for any of these keywords. During this period, ASELS stock prices experienced a steady increase, reaching a peak at the end of 2018, and people began searching for this stock more frequently. However, the peak in search activity coincided with a significant event in July 2018 when President Erdoğan introduced a new cabinet and treasury secretary, affecting ASELS as it is a governmental institution. As a result, it's not surprising to see ASELS as the most affected stock by government policies. Subsequently, interest in ASELS decreased, although the stock price recovered from the shock effect but followed a declining trend.
# 
# We observe a similar pattern with THYAO but on a smaller scale. There was a second peak in THYAO prices shortly after July 2018, in October 2018, coinciding with the highest search activity for this stock.
# 
# In the case of EREGL, there were no distinct peaks in Google Trends data, except for the last month of this dataset in July 2019. Notably, there were no dramatic shifts in the price of this stock during the same period. Therefore, the increased interest in this stock could be only a coincidence.
# 

# ### Remark:
# This was my first experience applying PCA to data using Python and I could not use the codes provided in the Moodle since it was in R. So, I had to search for and gather codes from various websites and forums. 
# 
# It was very fun to play with this data, thank you for this homework!
