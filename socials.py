import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_excel('SocialsAnalysis.xlsx', sheet_name = 'LinkedIn Posts')



#---------DATA CLEANUP--------------------

#Change NaNs in Likers, Commenters, Reposters to empty strings
df[['Likers', 'Commenters', 'Reposters']] = df[['Likers', 'Commenters', 'Reposters']].fillna('')

#Change Likers, Commenters, Reposters columns to lists
df['Likers'] = df['Likers'].apply(lambda x: x.split(', '))
df['Commenters'] = df['Commenters'].apply(lambda x: x.split(', '))
df['Reposters'] = df['Reposters'].apply(lambda x: x.split(', '))

#print(df.head())

#Check unique values of Liker, Commenter, Reposter names - looks good!
names_list = []
for name in df['Likers']:
    names_list.extend(name)
for name in df['Commenters']:
    names_list.extend(name)
for name in df['Reposters']:
    names_list.extend(name)

names_df = pd.DataFrame(names_list, columns=['Name'])
names_count_df = pd.DataFrame(names_df.groupby(['Name']).size().reset_index(name='Count'))

#pd.set_option('display.max_rows', None)
#print(names_count_df.head())
#print(names_count_df['Count'].sum())
#pd.reset_option('display.max_rows')

#Convert Length of Video from float to times - looks good
df['Length of Video'] = df['Length of Video'].apply(lambda x: x if pd.isna(x) else time(0, int(modf(x)[1]), int(round(modf(x)[0] * 100))))

df['Engagement'] = df[['Likes', 'Comments', 'Reposts']].sum(axis=1)

#--------ANALYSIS 1: What % of engagement is from employees/prev. employees?-------

# Functions used
#If name is in empl_df and is current/prev, add to index 1/2, else add to index 0
def sort_empl(x):
    if x.Name != '':
        if x.Name in empl_df.Name.values:
            row = empl_df.loc[empl_df['Name'] == x.Name]
            if row.Current.values:
                empl_counts[1] += x.Count
            elif row.Prev.values:
                empl_counts[2] += x.Count
            else:
                empl_counts[0] += x.Count
        else:
            empl_counts[0] += x.Count

#Read employee/not employee sheet
empl_df = pd.read_excel('SocialsAnalysis.xlsx', sheet_name='Employees')
# 0 none, 1 current, 2 prev
empl_counts = [0, 0, 0]
names_count_df.apply(sort_empl, axis=1)
total = sum(empl_counts)
empl_counts = [(empl_counts[0]/total) * 100, (empl_counts[1]/total) * 100, (empl_counts[2]/total) * 100]

plt.pie(empl_counts, labels=['Non-employee', 'Current employee', 'Previous employee'], autopct='%.2f%%', colors=['#FFDB80', '#66B4B6', '#97A0CF'])
plt.suptitle('LinkedIn Engagement by Employee Status')
plt.title('Oct-Dec 2025')


#------------ANALYSIS 2: What topic of post gets the most views and engagement?-----------------

#Groupby topic, sum views, likes, and comments
topics_df = df.groupby(['Topic'])[['Impressions/Views', 'Likes', 'Comments', 'Reposts']].agg('sum')
topics_df['Num Posts'] = df.groupby(['Topic']).size().values
topics_df = topics_df.iloc[:, [4, 0, 1, 2, 3]]
#print(topics_df)

#Find ratio of views/engagement to posts per topic, rounded to 2 decimal places
topics_df['Engagement'] = topics_df[['Likes', 'Comments', 'Reposts']].sum(axis=1)
topics_df['Views to Posts'] = topics_df['Impressions/Views'] / topics_df['Num Posts']
topics_df['Engagement to Posts'] = topics_df['Engagement'] / topics_df['Num Posts']
topics_df[['Views to Posts', 'Engagement to Posts']] = topics_df[['Views to Posts', 'Engagement to Posts']].apply(lambda x: round(x, 2))

ax_views = topics_df.plot(y='Views to Posts', kind='bar', ylabel="Mean Views", legend=False, title='Mean Views of LinkedIn Posts by Topic', color='#00a6cb')
ax_views.set_xticklabels(ax_views.get_xticklabels(), rotation=35, ha='center')
plt.tight_layout()
ax_eng = topics_df.plot(y='Engagement to Posts', kind='bar', ylabel="Mean Likes/Comments/Reposts", legend=False, title='Mean Engagement With LinkedIn Posts by Topic', color='#2F419F')
ax_eng.set_xticklabels(ax_eng.get_xticklabels(), rotation=35, ha='center')
plt.tight_layout()


#Double Bar graph by topic - for fun :)
x_axis = np.arange(df['Topic'].nunique())
plt.bar(x_axis-0.2, topics_df['Views to Posts'], 0.4, color='#FFB700')
plt.bar(x_axis+0.2, topics_df['Engagement to Posts'], 0.4, color='#008286')
plt.title('Views and Engagement by Topic')
plt.xticks(x_axis, df.groupby('Topic').groups.keys(), rotation=45)
plt.ylabel('Count')
plt.legend(['Views', 'Likes/Comments/Reposts'])
plt.tight_layout()
plt.show()

#----------ANALYSIS 3: What is the relationship between engagement and views?-----------------------

#Scatter plot - all
df.plot.scatter('Engagement', 'Impressions/Views', c='#2F419F', xlabel='Num Likes/Comments/Reposts', ylabel='Num Views', title='Engagement vs Views')
#Line of best fit
beta, alpha = np.polyfit(df['Engagement'], df['Impressions/Views'], 1)
#print('Alpha: ' + str(round(alpha, 3)) + ', Beta: ' + str(round(beta, 3)))
xseq = np.linspace(0, 55, num=100)
plt.plot(xseq, alpha + beta*xseq)

#Scatter plot - minus outliers
df_min_outliers = df[df['Impressions/Views'] < 1000]
df_min_outliers.plot.scatter('Engagement', 'Impressions/Views', c='#00a6cb', xlabel='Num Likes/Comments/Reposts', ylabel='Num Views', title='Outliers Removed')
plt.suptitle('Engagement vs Views')
beta, alpha = np.polyfit(df_min_outliers['Engagement'], df_min_outliers['Impressions/Views'], 1)
#print('Alpha: ' + str(round(alpha, 3)) + ', Beta: ' + str(round(beta, 3)))
xseq = np.linspace(0, 16, num=100)
plt.plot(xseq, alpha + beta*xseq)
plt.show()

#-----------ANALYSIS 4: What are the characteristics of top-performing videos?-----------
#Length, engagement, etc.
