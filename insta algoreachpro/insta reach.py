import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import joblib

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
#taking the null value column and dropping them
data = pd.read_csv("Instagram data.csv", encoding = 'latin1')
print(data.head())
data.isnull().sum()
data = data.dropna()

#data.info()

# to show bargraph and histogram

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show() 
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()

plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()

#defining labels=from dataset and values =defining
home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels,
             title='Impressions on Instagram Posts From Various Sources', hole=0.8)
fig.show()

#worldcloud from caption
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#world cloud,  from hashtag
text = " ".join(i for i in data.Hashtags)
#setting stopwords
stopwords = set(STOPWORDS)
#generating world cloud
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
#displaying worldcloud
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear') #interpolation bilinear sooths the image
plt.axis("off")
plt.show()

#to show scatter plot
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols",
                    title = "Relationship Between Likes and  Impressions")
figure.show()
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols",
                    title = "Relationship Between Comments and Total Impressions")
figure.show()
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares", trendline="ols",# tredline=ols use ols to show general trends
                    title = "Relationship Between Shares and Total Impressions")
figure.show()
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves", trendline="ols",
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()
#correlATION BETWEEN DATA



#correlation = data.corr() wrong  it includes all data including string 
numeric_data = data.select_dtypes(include=np.number)
correlation = numeric_data.corr()
print(correlation["Impressions"].sort_values(ascending=True))

#conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
#print(conversion_rate)


figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols",
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()




x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares',
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2,
                                                random_state=42)
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)

joblib_file=("instareach.pkl")
joblib.dump(model,joblib_file)
# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[253.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
reach=model.predict(features)
print("feartures",reach)