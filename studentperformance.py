# import necessary modules
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# store the data in a variable
data190315 = open("C:\\Users\\Xin\\PycharmProjects\\untitled\\StudentsPerformance.csv")
# Read in the data with `read_csv()`
S_data = pd.read_csv(data190315, encoding='utf-8')
# Using .head() method to view the first few records of the data set
print(S_data.head())
# using the dtypes() method to display the different datatypes available
print(S_data.dtypes)
# import the necessary module
from sklearn import preprocessing

# create the Labelencoder object
le = preprocessing.LabelEncoder()

print("gender' : ", S_data['gender'].unique())
print("race/ethnicity : ", S_data['race/ethnicity'].unique())
print("parental level of education : ", S_data['parental level of education'].unique())
print("lunch : ", S_data['lunch'].unique())
print("test preparation course : ", S_data['test preparation course'].unique())

# convert the categorical columns into numeric
S_data['gender'] = le.fit_transform(S_data['gender'])
S_data['race/ethnicity'] = le.fit_transform(S_data['race/ethnicity'])
S_data['parental level of education'] = le.fit_transform(S_data['parental level of education'])
S_data['lunch'] = le.fit_transform(S_data['lunch'])
S_data['test preparation course'] = le.fit_transform(S_data['test preparation course'])

# select columns other than 'race/ethnicity','math score', 'test preparation course', 'reading score' and 'writing score'
cols = [col for col in S_data.columns if
        col not in ['race/ethnicity', 'math score', 'test preparation course' 'reading score', 'writing score', 'gender']]
# dropping the 'race/ethnicity', 'reading score', 'writing score', 'gender' and 'math score' columns
data = S_data[cols]
ms = S_data['math score']
print(ms)
for index in ms.index:
    if ms[index] < 60:
        ms[index] = 0
    elif 60 <= ms[index] < 70:
        ms[index] = 1
    elif 70 <= ms[index] < 80:
        ms[index] = 2
    elif 80 <= ms[index] < 90:
        ms[index] = 3
    else:
        ms[index] = 4
print(ms)
# assigning the 'ms' column as target
target = ms

print(data.head(n=2))
print(ms.head(n=5))

# split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.20, random_state=10)
print(data_train, data_test, target_train, target_test)


# create an object of the type MultinomialNB
mtn = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)



# train the algorithm on training data and predict using the testing data
pred = mtn.fit(data_train, target_train).predict(data_test)
pred1 = mtn.fit(data_train, target_train).predict_proba(data_test)

print("\n==Predict result by multinomial predict==")
print(pred.tolist())
print("\n==Predict result by multinomial predict_proba==")
print(pred1.tolist())

# print the accuracy score of the model
print("Multinomial Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize=True))

# create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
Gau = gnb.fit(data_train, target_train).predict(data_test)
Gau1 = gnb.fit(data_train, target_train).predict_proba(data_test)
print("\n==Predict result by multinomial predict==")
print(Gau.tolist())
print("\n==Predict result by multinomial predict_proba==")
print(Gau1.tolist())

#print the accuracy score of the model
print("Gaussian Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))