# Reason or cause of failure in machine learing identification .
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('ML_DATA_Reviews.csv', header=None);
dataset = dataset.fillna(0);
all_data_subset = dataset.iloc[1:,2:30].values;

#getting all the incorrect values
listEntry=[];
i=0;
while(i<len(all_data_subset)):
    temp=all_data_subset[i];
    valid=temp[26]
    if(valid=='INCORRECT'):
      listEntry.append(temp);
    i+=1;
    
all_incorrect_enteries = np.array(listEntry)

#taking out only the last colum for data processing to nlp
corpus=[];
nlpProcessing=all_incorrect_enteries[:,27];

#replacing not necessary string to blank
for i in range(0,len(nlpProcessing)):
    review=nlpProcessing[i];
    review=review.replace('INCORRECT',"");
    review=review.replace('TAX',"");
    review=review.replace('FULLINCOMETAX',"");
    review=review.replace('FULLINCOME',"");
    review=review.replace('/',' ');
    review=review.strip();
    corpus.append(review);

#creating sparse matrix for data processing 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray();


#converting index to int binary -> decimal conversion
nlpInt=[];
for i in range(0,len(X)):
    sum='';
    values=X[i];
    for j in range(0,len(values)):
        if(values[j]==1):
            sum+='1';
        else:
            sum+='0';
    nlpInt.append(int(sum,2));

y=np.array(nlpInt);
X_data=all_incorrect_enteries[:,:26];


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_data,y,random_state=1,test_size=0.30);


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier();
classifier=classifier.fit(X_train,y_train);
y_pred=classifier.predict(X_test);
classifier.score(X_test,y_test)

list=[];
dist=cv.vocabulary_;
for i in dist.keys():
    val=dist.get(i);
    list.append(str(i));
    


reason=[];
for i in range(0,len(y_pred)):
    values=[];
    binary_val=("{0:b}".format(y_pred[i]));
    binary_val_new=binary_val
    for j in range(len(binary_val),len(dist)):
          binary_val_new="0"+binary_val_new;
    for k in range(0,len(binary_val_new)):
          if(int(binary_val_new[k])==1):
            values.append(list[int(k)])
    reason.append(values);







    
    





