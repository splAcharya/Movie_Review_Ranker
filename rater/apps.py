from django.apps import AppConfig
import os, joblib
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords #for stopwords
from nltk.stem import PorterStemmer #for word stemming
from nltk.tokenize import TweetTokenizer #for toekinizing string to list of words
import string #for punctuation
import re #for regular expression


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_file_0 = os.path.join(BASE_DIR,'saved_models/0_TfidfVectorizer.joblib')
model_file_1 = os.path.join(BASE_DIR,'saved_models/1_BernoulliNB.joblib')
model_file_2 = os.path.join(BASE_DIR,'saved_models/2_DecisionTreeClassifier.joblib')
model_file_3 = os.path.join(BASE_DIR,'saved_models/3_KNeighborsClassifier.joblib')
model_file_4 = os.path.join(BASE_DIR,'saved_models/4_LogisticRegression.joblib')
model_file_5 = os.path.join(BASE_DIR,'saved_models/5_LinearSVC.joblib')
model_file_6 = os.path.join(BASE_DIR,'saved_models/6_BaggingClassifier.joblib')
model_file_7 = os.path.join(BASE_DIR,'saved_models/7_StackingClassifier.joblib')
model_file_8 = os.path.join(BASE_DIR,'saved_models/8_RandomForestClassifier.joblib')
model_file_9 = os.path.join(BASE_DIR,'saved_models/9_AdaBoostClassifier.joblib')
model_file_10 = os.path.join(BASE_DIR,'saved_models/10_ExtraTreesClassifier.joblib')



class RaterConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rater'

    model_tfidf = joblib.load(model_file_0)
    model_nb = joblib.load(model_file_1)
    model_dtc = joblib.load(model_file_2)
    model_knn = joblib.load(model_file_3)
    model_lr = joblib.load(model_file_4)
    model_svm = joblib.load(model_file_5)
    model_bc = joblib.load(model_file_6)
    model_sc = joblib.load(model_file_7)
    model_rfc = joblib.load(model_file_8)
    model_abc = joblib.load(model_file_9)
    model_etc = joblib.load(model_file_10)
    
    ml_model_list = [model_nb,model_dtc,model_knn,model_lr,model_svm,model_bc,model_sc,model_rfc,model_abc,model_etc]

    ml_model_names = [model.__class__.__name__ for model in ml_model_list]


    #initilze tweet tokenizer 
    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)

    #intizlize porter stemmer
    porter_stemmer = PorterStemmer()
    
    #get english stopwords
    english_stopwords = stopwords.words("english")

    english_punctuations = list(string.punctuation)

    print("Models Loaded\n" * 10)


