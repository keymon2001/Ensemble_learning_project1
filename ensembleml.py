
# library 
from pathlib import Path
import os.path
import urllib         #allows you to access, and interact with, websites using their URL's (Uniform Resource Locator)
import wget          # wget is a URL network downloader it helps in downloading files directly from the main server
import zipfile       # for manipulating ZIP files
import pandas as pd  # for data manipulation and analysis
import numpy as np               # linear algebra
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns            # data visualization library and can perform exploratory analysis
from sklearn.model_selection import train_test_split  #Split arrays or matrices into random train and test subsets.
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB     
from sklearn.ensemble import BaggingClassifier      
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics  import  confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore") 
#----------------------------------------------------

# read data from URL network
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip'
infotext = "datainformation.txt"

class emsrmbleML:
    def __init__(self,url,infotext):
        self.url = url
        self.df0 = pd.DataFrame()
        self.dinfo = infotext
        self.result = pd.DataFrame()
        
    def variablepro(self):
        if os.path.exists('bank-additional.zip'):
            zf = zipfile.ZipFile('bank-additional.zip')
            df0 = pd.read_csv(zf.open('bank-additional/bank-additional-full.csv'), sep=';')
        else:
            wget.download(self.url)
            zf = zipfile.ZipFile('bank-additional.zip')
            df0 = pd.read_csv(zf.open('bank-additional/bank-additional-full.csv'), sep=';')
        df0.drop(["duration"],inplace=True,axis=1)
        df0["pdays"]=df0["pdays"].astype("category")
        df0["y"]=df0["y"].astype("category")
        # file1 = open(self.dinfo,"a")
        # file1.write("\nData Information: \n")
        # df0.info(buf=file1)
        # file1.close() #to change file access modes

        plt.figure(figsize=(15,5))  #used create a new figure (15,5) are the width and height in inches
        sns.boxplot(x=df0["age"],data=df0)  # view age column has some outliers,median about age 40
        # Saving figure by changing parameter values
        plt.savefig("static/boxplot.png", bbox_inches="tight",pad_inches=0.3, transparent=False)
        
        df0["job"].value_counts() # total number of particular job
        plt.figure(figsize=(15,5))  #used create a new figure (15,5) are the width and height in inches
        sns.countplot(df0["job"])  # countplot on job
        # Saving figure by changing parameter values
        plt.savefig("static/job.png", bbox_inches="tight",pad_inches=0.3, transparent=False)
        
        sns.countplot(df0["marital"])  # view marital status
        # Saving figure by changing parameter values
        plt.savefig("static/marital.png", bbox_inches="tight",pad_inches=0.3, transpare=False)
        # View education status
        plt.figure(figsize=(12,5))
        sns.countplot(df0["education"])
        # Saving figure by changing parameter values
        plt.savefig("static/education.png", bbox_inches="tight",pad_inches=0.3, transpare=False)
        
        sns.countplot(df0["housing"])   # view has housing loan
        # Saving figure by changing parameter values
        plt.savefig("static/house.png", bbox_inches="tight",pad_inches=0.3, transpare=False)
        #Rename the dependant column from 'y ' to 'Target'
        df0.rename(columns={'y':'Target'}, inplace=True)
        #Group numerical variables by mean for the classes of Y variable
        np.round(df0.groupby(["Target"]).mean() ,1)
        
        pd.crosstab(df0['job'], df0['Target'], normalize='index').sort_values(by='yes',ascending=False )
        ct = pd.crosstab(df0['job'], df0['Target'], normalize='index')
        # pie plot with respect Jobs with the client subscribed
        ct.plot.pie(subplots=True, figsize=(15, 10),autopct='%1.1f%%')
        plt.legend(title='Jobs with the client subscribed')
        # Saving figure by changing parameter values
        plt.savefig("static/pie.png", bbox_inches="tight",pad_inches=0.3, transpare=False)
        
        sns.countplot(df0["Target"])   # has the client subscribed a term deposit
        plt.savefig("static/Target.png", bbox_inches="tight",pad_inches=0.3, transpare=False)
        self.df0 = df0
        
        

    def calculationem(self):
        df0 = self.df0
        x = df0.drop("Target" , axis=1)
        y = df0["Target"]   # select all rows and the 17 th column which is the classification "Yes", "No"
        x = pd.get_dummies(x, drop_first=True)
        test_size = 0.30 # taking 70:30 training and test set
        seed = 7  # Random numbmer seeding for reapeatability of the code
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
                
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
        clf_entropy.fit(x_train, y_train)
        preds_entropy = clf_entropy.predict(x_test)
        # Model Accuracy, how often is the classifier correct?
        acc_DT = accuracy_score(y_test,preds_entropy)
        recall_DT = recall_score(y_test, preds_entropy, average="binary", pos_label="yes")
        precision = precision_score(y_test, preds_entropy,average="binary", pos_label="yes")
        f1 = f1_score(y_test, preds_entropy,average="binary", pos_label="yes")
        #Store the accuracy results for each model in a dataframe for final comparison
        resultsDf = pd.DataFrame({'Method':['Decision Tree'], 'accuracy': [acc_DT], 'recall': [recall_DT],'precision':[precision],'f1_score':[f1]})
        resultsDf = resultsDf[['Method', 'accuracy', 'recall','precision','f1_score']]
        features = list(x_train.columns)
        plt.figure(figsize=(20, 12))
        tree.plot_tree(clf_entropy,feature_names=features,rounded=True, filled=True, proportion=True); 
        # Saving figure by changing parameter values
        plt.savefig("static/decission_tree.png", bbox_inches="tight",pad_inches=0.3, transpare=False)

        rfcl = RandomForestClassifier(n_estimators = 50)
        rfcl = rfcl.fit(x_train, y_train)
        #Predict the response for test dataset
        pred_RF = rfcl.predict(x_test)
        # Model Accuracy, how often is the classifier correct?
        acc_RF = accuracy_score(y_test, pred_RF)
        recall_RF = recall_score(y_test, pred_RF, average="binary", pos_label="yes")
        precision = precision_score(y_test, pred_RF,average="binary", pos_label="yes")
        f1 = f1_score(y_test, pred_RF,average="binary", pos_label="yes")
        #Store the accuracy results for each model in a dataframe for final comparison
        tempResultsDf = pd.DataFrame({'Method':['Random Forest'], 'accuracy': [acc_RF], 'recall': [recall_RF],'precision':[precision],'f1_score':[f1]})
        resultsDf = pd.concat([resultsDf, tempResultsDf])
            
        # Build a BernoulliNB Classifier
        NBclf=BernoulliNB()
        NBclf.fit(x_train, y_train)
        y_pred=NBclf.predict(x_test)
        ## Making the Confusion Matrix   
        acc_NB = accuracy_score(y_test,y_pred)
        recall_NB = recall_score(y_test, y_pred, average="binary", pos_label="yes")
        precision_NB = precision_score(y_test, y_pred, average="binary", pos_label="yes")
        f1score_NB=f1_score(y_test, y_pred, average="binary", pos_label="yes")
        #Store the accuracy results for each model in a dataframe for final comparison
        resultsDf1 = pd.DataFrame({'Method':['neive Bayes'], 'accuracy': [acc_NB], 'recall': [recall_NB],'precision':[precision_NB],'f1_score':[f1score_NB]})
        ##The concat() function is used to concatenate (or join together) two or more Pandas objects such as dataframes or series.
        resultsDf2 = pd.concat([resultsDf,resultsDf1 ])
               
        #We use bagging for combining weak learners of high variance. 
        bgcl = BaggingClassifier(n_estimators=100, max_samples= .7, bootstrap=True, oob_score=True, random_state=22)
        ## Model training
        bgcl = bgcl.fit(x_train, y_train)
        #Predict the response for test dataset
        pred_BG =bgcl.predict(x_test)
        # Model Accuracy, how often is the classifier correct?
        acc_BG = accuracy_score(y_test, pred_BG)
        recall_BG = recall_score(y_test, pred_BG, pos_label='yes')
        precision_BG = precision_score(y_test, pred_BG, average="binary", pos_label="yes")
        f1score_BG=f1_score(y_test, pred_BG, average="binary", pos_label="yes")
        #Store the accuracy results for each model in a dataframe for final comparison
        resultsDf1 = pd.DataFrame({'Method':['Bagging Classifier'], 'accuracy': [acc_BG], 'recall': [recall_BG],'precision':[precision_BG],'f1_score':[f1score_BG]})
        ##The concat() function is used to concatenate (or join together) two or more Pandas objects such as dataframes or series.
        resultsDf3 = pd.concat([resultsDf2,resultsDf1 ])       

        # We use boosting for combining weak learners with high bias
        gbcl=GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,random_state=22)
        ## Model training
        gbcl=gbcl.fit(x_train,y_train)
        #Predict the response for test dataset
        pred_BG=bgcl.predict(x_test)
        #Predict the response for test dataset
        pred_GB=gbcl.predict(x_test)
        # Model Accuracy, how often is the classifier correct?
        acc_GB=accuracy_score(y_test,pred_GB)
        recall_GB = recall_score(y_test, pred_GB, pos_label='yes')
        #Store the accuracy results for each model in a dataframe for final comparison
        precision_GB = precision_score(y_test, pred_GB, average="binary", pos_label="yes")
        f1score_GB=f1_score(y_test, pred_GB, average="binary", pos_label="yes")
        ##The concat() function is used to concatenate (or join together) two or more Pandas objects such as dataframes or series.
        resultsDf1=pd.DataFrame({"Method":["GradientBoostingClassifier"],"accuracy":[acc_GB],"recall":[recall_GB],'precision':[precision_GB],'f1_score':[f1score_GB] })
        resultsDf4=pd.concat([resultsDf3,resultsDf1])

        classifier= LogisticRegression(random_state=0) 
        classifier.fit(x_train, y_train) 
        #create a  LogisticRegression using Scikit-learn.
        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None, max_iter=100,multi_class='warn',n_jobs=None,penalty='l2',random_state=0,solver='warn',tol=0.0001,verbose=0,warm_start=False) 
        y_pred= classifier.predict(x_test)  
        # Model Accuracy, how often is the classifier correct?
        acc_LR = accuracy_score(y_test, y_pred)
        recall_LR = recall_score(y_test, y_pred, average="binary", pos_label="yes")
        precision_LR = precision_score(y_test, y_pred, average="binary", pos_label="yes")
        f1score_LR=f1_score(y_test, y_pred, average="binary", pos_label="yes")
        #Store the accuracy results for each model in a dataframe for final comparison
        resultsDf1=pd.DataFrame({"Method":["LogisticRegression"],"accuracy":[acc_LR],"recall":[recall_LR],'precision':[precision_LR],'f1_score':[f1score_LR]})
        ##The concat() function is used to concatenate (or join together) two or more Pandas objects such as dataframes or series.
        resultsDf5=pd.concat([resultsDf4,resultsDf1])
        #---------------------
        self.result=resultsDf5