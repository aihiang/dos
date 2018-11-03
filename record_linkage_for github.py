import pandas as pd
import recordlinkage
import numpy as np
import csv
from recordlinkage.preprocessing import clean, phonenumbers
from sklearn import model_selection #to save model
from sklearn.linear_model import LogisticRegression  #to save model
import pickle #to save model


# no need for deduplication, already checked for unique records for data

a = pd.read_csv('C:.csv')
b = pd.read_csv(r'C:\csv') #names already cleaned
c = pd.read_csv(r'C:\.csv') #used excel to generate training data, get correct matches

KOMPASS = a.copy()
TIMES = b.copy()
MATCH = c.copy()

KOMPASS.set_index(['KOMPASSRECID'], inplace=True)
TIMES.set_index(['TIMESRECID'], inplace=True)
#MATCH.head()

matchList=[]

for row in MATCH.iterrows():
    index, data = row
    matchList.append(data.tolist())
    
KOMPASS.fillna(0, inplace=True) #put inplace=True so it stays throughout the entire analysis
TIMES.fillna(0, inplace=True)

print(len(KOMPASS)) #10606
print(len(TIMES)) #6714

#standardizing obj type, else return errors. Not documented in record linkage toolkit
TIMES['REG_UNIT_NO'] = TIMES['REG_UNIT_NO'].astype(object)                  #REG_UNIT_NO
KOMPASS['REG_UNIT_NO'] = KOMPASS['REG_UNIT_NO'].astype(object)
TIMES['REG_POSTAL_CODE'] = TIMES['REG_POSTAL_CODE'].astype(object)          #REG_POSTAL_CODE
KOMPASS['REG_POSTAL_CODE'] = KOMPASS['REG_POSTAL_CODE'].astype(object)
TIMES['REG_BLKHSE_NO'] = TIMES['REG_BLKHSE_NO'].astype(object)              #REG_BLKHSE_NO
KOMPASS['REG_BLKHSE_NO'] = KOMPASS['REG_BLKHSE_NO'].astype(object)
TIMES['REG_STR_NM'] = TIMES['REG_STR_NM'].astype(str)                       #REG_STR_NM
KOMPASS['REG_STR_NM'] = KOMPASS['REG_STR_NM'].astype(str)
TIMES['REG_BLDG_NM'] = TIMES['REG_BLDG_NM'].astype(str)                     #REG_BLDG_NM
KOMPASS['REG_BLDG_NM'] = KOMPASS['REG_BLDG_NM'].astype(str)

#print(KOMPASS.dtypes)

#SORTED NEIGHBOUR
#Window must be an odd integer
#indexer = recordlinkage.FullIndex()
indexer = recordlinkage.SortedNeighbourhoodIndex(on='COMPANYNAME', window = 3)
pairs = indexer.index(TIMES, KOMPASS)
print("number of record pairs created")
print (len(pairs)) #11907 record pairs

#Getting degree of match
compare_cl = recordlinkage.Compare()
compare_cl.string('COMPANYNAME', 'COMPANYNAME', method='jarowinkler',  label = 'COMPANYNAME')
compare_cl.string('REG_STR_NM', 'REG_STR_NM',  method='jarowinkler',  label = 'REG_STR_NM')
compare_cl.string('REG_BLDG_NM', 'REG_BLDG_NM',  method='jarowinkler', label = 'REG_BLDG_NM')
compare_cl.exact('REG_BLKHSE_NO', 'REG_BLKHSE_NO',  label = 'REG_BLKHSE_NO')
compare_cl.exact('REG_LVL_NO', 'REG_LVL_NO',  label = 'REG_LVL_NO')
compare_cl.exact('REG_UNIT_NO', 'REG_UNIT_NO',  label = 'REG_UNIT_NO')
compare_cl.exact('REG_POSTAL_CODE', 'REG_POSTAL_CODE',  label = 'REG_POSTAL_CODE')

linkResult = compare_cl.compute(pairs, TIMES, KOMPASS)

#print(len(linkResult)) #11907
#linkResult.head()

#make a copy of the pairs, then tag the pairs true match or not
matchResult=linkResult.copy()
matchResult['MatchStatus'] = False
for i in range(0, len(linkResult)):
    tempList = list(list(linkResult.index)[i]) #convert tuple to list
    if((tempList in matchList)==True):
        matchResult.set_value(list(linkResult.index)[i], 'MatchStatus', True)
        


matchTRUE =matchResult[(matchResult['MatchStatus']== True)]
matchTRUE=matchTRUE.drop(['MatchStatus'], axis=1)
print("number of record pairs that are true")
print(len(matchTRUE))


# Shuffle
from sklearn.utils import shuffle
linkResult = shuffle(linkResult)
# linkResult.head()

# 80/20 ratio. change accordingly
# Not necessary for 80/20 split -> if this, use enture linkResult will do 
golden_pairs = linkResult[0:9001]
golden_matches_trng_index = golden_pairs.index & matchTRUE.index 
#golden_pairs.index.get_values()
print("number of true matches in training data")
print(len(golden_matches_trng_index))

#Logistic Regression Classifier
# Initialize the classifier
logreg = recordlinkage.LogisticRegressionClassifier()
# Train the classifier
logreg.learn(golden_pairs, golden_matches_trng_index)
result_logreg = logreg.predict(linkResult[0:9001])
print("training set predicted number of true pairs")
print(len(result_logreg))
# print ("Intercept: ", logreg.intercept)
# print ("Coefficients: ", logreg.coefficients)

# Get confusion matrix
conf_logreg = recordlinkage.confusion_matrix(golden_matches_trng_index, result_logreg, 9000)
print(conf_logreg)

# Get F-score for classification of the 80%
print ("F Score Results for Training Set")
print(recordlinkage.fscore(conf_logreg))
# Returns 0.9538850284270373


# Now work on test set
golden_pairs2 = linkResult[9001:]
golden_matches_test_index = golden_pairs2.index & matchTRUE.index
result_logreg2 = logreg.predict(linkResult[9001:])
# Get confusion matrix
conf_logreg2 = recordlinkage.confusion_matrix(golden_matches_test_index, result_logreg2, len(linkResult)-9000)
print(conf_logreg2)

# Get F-score for classification
print ("F Score Results for Testing Set")
print(recordlinkage.fscore(conf_logreg2))
# returns error due to 0 cases


# SAVE THE TRAINING MODEL using pickle
filename = 'finalized_model.sav'
pickle.dump(logreg, open(filename,'wb'))


#Deploying onto dataset
#a = pd.read_csv('C:\DOS Internship\Data Sets\KOMPASS n TIMES\Kompass.csv')
#d = pd.read_csv(r'C:\DOS Internship\Data Sets\5. Record Linkages\unpopulatedbase.csv')
d = pd.read_csv(r'C:\DOS Internship\Data Sets\7. FINAL\Base RL Match_urlrerun_url2.csv')
e = pd.read_csv(r'C:\DOS Internship\Data Sets\5. Record Linkages\11. rerun2_to_check_linkages.csv')
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
logreg = loaded_model


#split the data up
BASE = d.copy()
DATA = e.copy()
BASE.fillna(0, inplace=True)
DATA.fillna(0, inplace=True)
popBASE = BASE[(BASE['URL']!=0)] #those already with urls
unpopBASE = BASE[(BASE['URL']==0)]

print("size of populated base is " + str(len(popBASE)))
print("size of unpopulated base is " + str(len(unpopBASE)))

unpopBASE.is_copy = False   
unpopBASE.loc[:,'BASERECID'] = "BASE - " + (unpopBASE.index).astype(str)
DATA.loc[:,'MASRECID'] = "MAS - " + (DATA.index).astype(str) ##to change
#print(unpopBASE[:5])
#print(DATA[:5])

unpopBASE.set_index(['BASERECID'], inplace=True)
DATA.set_index(['MASRECID'], inplace=True) #to change


#standardizing obj type
DATA['ENTP_NEW'] = DATA['ENTP_NEW'].astype(str)                    #ENTP_NEW
unpopBASE['ENTP_NEW'] = unpopBASE['ENTP_NEW'].astype(str)
DATA['REG_STR_NM'] = DATA['REG_STR_NM'].astype(str)                #REG_STR_NAME
unpopBASE['REG_STR_NM'] = unpopBASE['REG_STR_NM'].astype(str)
DATA['REG_BLDG_NM'] = DATA['REG_BLDG_NM'].astype(str)              #REG_BLDG_NM
unpopBASE['REG_BLDG_NM'] = unpopBASE['REG_BLDG_NM'].astype(str)
DATA['REG_BLKHSE_NO'] = DATA['REG_BLKHSE_NO'].astype(object)       #REG_BLKHSE_NO
unpopBASE['REG_BLKHSE_NO'] = unpopBASE['REG_BLKHSE_NO'].astype(object)
DATA['REG_LVL_NO'] = DATA['REG_LVL_NO'].astype(object)             #REG_LVL_NO
unpopBASE['REG_LVL_NO'] = unpopBASE['REG_LVL_NO'].astype(object)
DATA['REG_UNIT_NO'] = DATA['REG_UNIT_NO'].astype(object)           #REG_UNIT_NO
unpopBASE['REG_UNIT_NO'] = unpopBASE['REG_UNIT_NO'].astype(object)
DATA['REG_POSTAL_CODE'] = DATA['REG_POSTAL_CODE'].astype(object)   #REG_POSTAL_CODE
unpopBASE['REG_POSTAL_CODE'] = unpopBASE['REG_POSTAL_CODE'].astype(object)
#print(KOMPASS.dtypes)

print("CONVERTING TYPE OF DATA DONE!")


#Sorted Neighbour
indexer = recordlinkage.SortedNeighbourhoodIndex(on='ENTP_NEW', window = 3)
pairs = indexer.index(unpopBASE, DATA)
print("number of record pairs from base and data is ")
print (len(pairs))

#Getting degree of match
compare_cl = recordlinkage.Compare()
compare_cl.string('ENTP_NEW', 'ENTP_NEW', method='jarowinkler',  label = 'COMPANYNAME')
compare_cl.string('REG_STR_NM', 'REG_STR_NM',  method='jarowinkler',  label = 'REG_STR_NM')
compare_cl.string('REG_BLDG_NM', 'REG_BLDG_NM',  method='jarowinkler', label = 'REG_BLDG_NM')
compare_cl.exact('REG_BLKHSE_NO', 'REG_BLKHSE_NO',  label = 'REG_BLKHSE_NO')
compare_cl.exact('REG_LVL_NO', 'REG_LVL_NO',  label = 'REG_LVL_NO')
compare_cl.exact('REG_UNIT_NO', 'REG_UNIT_NO',  label = 'REG_UNIT_NO')
compare_cl.exact('REG_POSTAL_CODE', 'REG_POSTAL_CODE',  label = 'REG_POSTAL_CODE')

linkResult = compare_cl.compute(pairs, unpopBASE, DATA) 
print("CALCULATING DEGREE OF MATCH DONE!")


finalResults = logreg.predict(linkResult)
resultArray = finalResults.get_values()
print("predicted number of linkage pairs")
print(len(finalResults))

df = pd.DataFrame(resultArray)

df['ID1'] = df[0].apply(lambda x: x[0]) #BASE ID
df['ID2'] = df[0].apply(lambda x: x[1]) #DATA ID
df.drop(df.columns[0],axis=1,inplace=True)

df.to_csv("positivematched.csv")
unpopBASE.to_csv("base_check_duplicates.csv")
DATA.to_csv("data_check_duplicates.csv")


#to check manually if there are one-to-many pairs

f = pd.read_csv(r'C.csv')
f.drop(f.columns[[0]], axis=1, inplace=True) #axis=1 for columns, axis=0 for rows #remov unnamed row
#print(DATA.index.values)

for row in range(0, len(f)):
    print("processing matches " + str(row) + " out of a total of " + str(len(f)))
    baseID = f.iloc[row]['ID1']
    dataID = f.iloc[row]['ID2']
    tempURL = DATA.loc[dataID]['URL']
    unpopBASE.loc[baseID, 'URL']= tempURL

print("DONE UPDATING UNPOP BASE, NOW MERGE")
#unpopBASE.drop(unpopBASE.columns[0],axis=1, inplace=True) #drop index
#BASE = pd.concat
#print(unpopBASE[:5])
#print(popBASE[:5])
updatedBASE = pd.concat([popBASE, unpopBASE])
#updatedBASE.drop(updatedBASE.columns[[0]], axis=1, inplace=True)
updatedBASE.to_csv("repopulated_base.csv")
print("PROBABILISTIC MERGE DONE!")