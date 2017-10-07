import re
import pandas as pd
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


########################################################################

def pandas2arff(df,filename,wekaname = "pandasdata",cleanstringdata=True,cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka. 
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values. 
              To suppress this, set this to False
    """
    import re
    
    def cleanstring(s):
        if s!="?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"
            
    dfcopy = df #all cleaning operations get done on this copy

    
    if cleannan!=False:
        dfcopy = dfcopy.fillna(-999999999) #this is so that we can swap this out for "?"
        #this makes sure that certain numerical columns with missing values don't get stuck with "object" type
 
    f = open(filename,"w")
    arffList = []
    arffList.append("@relation " + wekaname + "\n")
    #look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        if dfcopy.dtypes[i]=='O' or (df.columns[i] in ["Class","CLASS","class"]):
            if cleannan!=False:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].replace(to_replace=-999999999, value="?")
            if cleanstringdata!=False:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].apply(cleanstring)
            _uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:,i])]
            _uniqueNominalVals = ",".join(_uniqueNominalVals)
            _uniqueNominalVals = _uniqueNominalVals.replace("[","")
            _uniqueNominalVals = _uniqueNominalVals.replace("]","")
            _uniqueValuesString = "{" + _uniqueNominalVals +"}" 
            arffList.append("@attribute " + df.columns[i] + _uniqueValuesString + "\n")
        else:
            arffList.append("@attribute " + df.columns[i] + " real\n") 
            #even if it is an integer, let's just deal with it as a real number for now
    arffList.append("@data\n")           
    for i in range(dfcopy.shape[0]):#instances
        _instanceString = ""
        for j in range(df.shape[1]):#features
                if dfcopy.dtypes[j]=='O':
                    _instanceString+="\"" + str(dfcopy.iloc[i,j]) + "\""
                else:
                    _instanceString+=str(dfcopy.iloc[i,j])
                if j!=dfcopy.shape[1]-1:#if it's not the last feature, add a comma
                    _instanceString+=","
        _instanceString+="\n"
        if cleannan!=False:
            _instanceString = _instanceString.replace("-999999999.0","?") #for numeric missing values
            _instanceString = _instanceString.replace("\"?\"","?") #for categorical missing values
        arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True





#######################################################################


def get_text_from_file(path):
    list_of_classes = []
    list_of_tweets = []
    f_open = open(path,'r')
    for line in f_open.readlines():
        temp = line.split(",")
        num_parts = int(len(temp))
        the_class = temp[0]
        text_temp = temp[4:num_parts]
        tweet_string = ' '.join(text_temp)
        tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
        string = re.sub(r'[^\x00-\x7F]', ' ', tweet_string) #remove unicodes
        string = string.replace(","," ")
        string = string.replace(".", " ")
        string = string.replace("!", " ")
        string = string.replace('*', " ")
        string = string.replace('"', " ")
        string  =string.replace("\n"," ")
        string = string.replace("\t"," ")
        string = re.sub('\s+', ' ', string)
        string = re.sub('[^0-9a-zA-Z]+', ' ', string)
        #tokens = word_tokenize(string)
        #string_as_tokens = [stemmer.stem(i.lower()) for i in tokens if i not in stop_words_list]
        list_of_classes.append(the_class)
        list_of_tweets.append(string)
    f_open.close()
    return list_of_classes, list_of_tweets



###########################################################################
## main()

path_train = 'input/train.txt'
path_test = 'input/test.txt'

list_classes_train, list_tweets_train = get_text_from_file(path_train)
list_classes_test, list_tweets_test = get_text_from_file(path_test) 

vectorizer = CountVectorizer(min_df=1, stop_words='english')
dtm = vectorizer.fit(list_tweets_train)
dtm_train = dtm.transform(list_tweets_train)
dtm_test  = dtm.transform(list_tweets_test)

#X_train =  pd.DataFrame(dtm_train.toarray(), index=list_tweets_train, columns=dtm.get_feature_names())


X_train =  pd.DataFrame(dtm_train.toarray(),index=list_classes_train)
X_test = pd.DataFrame(dtm_test.toarray(),index=list_classes_test)
#print X_train


#X_train.to_csv('output/train.txt',sep='\t')
#X_test.to_csv('output/test.txt',sep='\t')



#state = pandas2arff(X_train,'output/train.arff',wekaname = "train",cleanstringdata=True,cleannan=True)
#state = pandas2arff(X_test,'output/test.arff',wekaname = "test",cleanstringdata=True,cleannan=True)


lsa = TruncatedSVD(300)
dtm_lsa = lsa.fit(dtm_train)
dtm_lsa_train = dtm_lsa.transform(dtm_train)
dtm_lsa_test = dtm_lsa.transform(dtm_test)


X_train_lsa =  pd.DataFrame(dtm_lsa_train,index=list_classes_train)
X_test_lsa = pd.DataFrame(dtm_lsa_test,index=list_classes_test)

X_train_lsa.to_csv('output/train_lsa.txt',sep='\t')
X_test_lsa.to_csv('output/test_lsa.txt',sep='\t')



###########################################################################

print '<<<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>>>>>>'
