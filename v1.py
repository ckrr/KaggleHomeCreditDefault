import pandas as pd
from sklearn import linear_model

#Read data
trainOrig=pd.read_csv("application_train.csv")
trainX=trainOrig.drop(["TARGET"],axis=1)
trainY=trainOrig.loc[:,"TARGET"]
testOrig=pd.read_csv("application_test.csv")
testX=testOrig

#Handle non-numeric columns and nulls
trainX=pd.get_dummies(trainX)
testX=pd.get_dummies(testX)
[trainX,testX]=trainX.align(testX, join='inner', axis=1)

def setZero(x):
    return 0

for colName in trainX.columns:
    if (trainX.loc[:,colName].isnull().sum()>0):
        trainX.loc[:,colName]=trainX.loc[:,colName].apply(setZero, 1)
for colName in testX.columns:
    if (testX.loc[:,colName].isnull().sum()>0):
        testX.loc[:,colName]=testX.loc[:,colName].apply(setZero, 1)

regression=linear_model.LinearRegression()
regression.fit(trainX,trainY)

testY=regression.predict(testX)

def normalize(x):
    return min(max(0,x),1)

for i in range(len(testY)):
    testY[i]=normalize(testY[i])

finalOutput=pd.DataFrame()
finalOutput["SK_ID_CURR"]=testX.loc[:,"SK_ID_CURR"]
finalOutput["TARGET"]=testY

writer=pd.ExcelWriter("submission.xlsx")
finalOutput.to_excel(writer,index=False)
writer.save()














