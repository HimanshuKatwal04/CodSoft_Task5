'''CREDIT CARD FRAUD DETECTION'''

#IMPORTING LIBRARIES
import sys
import pandas as pd
import numpy as np
import seaborn as sns 
import scipy
import matplotlib
import sklearn

print('python : {}'. format(sys.version))
print('Numpy : {}'. format(np.__version__))
print('Pandas : {}'. format(pd.__version__))
print('Seaborn: {}'. format(sns.__version__))
print('Sklearn : {}'. format(sklearn.__version__))
print('Matplotlib : {}'. format(matplotlib.__version__))

import sys
import pandas as pd
import numpy as np
import seaborn as sns 
import scipy
import matplotlib.pyplot as plt 

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()

df.shape

df.describe()

count_1 = (df['Class'] == 1).sum()
count_0 = (df['Class'] == 0).sum()

# Print the counts
print(f"Count of 1: {count_1}")
print(f"Count of 0: {count_0}")

data=df.sample(frac = 0.1 , random_state=1)
print(data.shape)

#plot histogram
df.hist(figsize = (20 , 20))
plt.show()

fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]

outliers_frc = len(fraud) / float(len(valid))
print(outliers_frc)

print('Fraud Cases: {}' .format(len(fraud)))
print('valid Cases: {}' .format(len(valid)))

# Correlation metrics
corrmat = df.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat , vmax= .8 ,square = True)
plt.show()

# Get all the coluimns in dataframe
columns = df.columns.tolist()

# Filter the columns to remove data we do not want
columns = [data for data in columns if data not in ['Class']]
target = "Class"

X = df[columns]
y = df[target]

#Print the shape opf x and y

print(X.shape)
print(y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor   

'''MODEL'''

# define a random state
state = 1

Classifier = {
    "Isolation forest ": IsolationForest(max_samples=len(X),
                                        contamination = outliers_frc,
                                        random_state = state),
    
    "Local Outliers Factor" : LocalOutlierFactor(
    n_neighbors = 20 ,
    contamination = outliers_frc,
    novelty=True )
}

n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(Classifier.items()):
    #fit the data and tag outliers
    if clf_name == "Local Outliers Factors":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_  # Use negative_outlier_factor_ instead of decision_function
        
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)  # This will work for Isolation Forest
        y_pred = clf.predict(X)
        
    # reshape the prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != y).sum()
    
    # Run Classification Metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
