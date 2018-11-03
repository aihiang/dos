import pandas as pd
import numpy as np
import csv
from collections import Counter
import collections
import re


b = pd.read_csv(r'C:\Data Sets\3. Clean URL\times_add_url_clean.csv')

data=b.copy()
data.fillna('', inplace=True)

unwanted_list = [' PTE', ' LTD', 'LTD', 'PRIVATE', 'LIMITED', ' CO ', 'PTE.', 'LTD.']
# 'CO' and 'SINGAPORE' not removed
def cleanName(string):
    string = string.replace(' AND ', ' ')
    string1 = re.sub("|".join(unwanted_list), "", string)
    string2 = re.sub('[^ a-zA-Z0-9]', '', string1) #removes symbols
    string3 = re.sub(r'\s+', ' ',   string2).rstrip() #rstrip removes empty space at the end
    return string3
    
new_col = data['ENTP_NM'] #do once will do
data.insert(loc=2, column='ENTP_NEW', value=new_col)


for i in range(0, len(data)):
    print("processing link " + str(i)+ " " + data['ENTP_NM'][i])
    tempName = data['ENTP_NM'][i]
    newName = cleanName(tempName)
    data.loc[i,'ENTP_NEW'] = newName


data.to_csv('rerun2_add_url_name_clean.csv')
print(len(b))