from django.shortcuts import render
#from django.http import HttpResponse
from rater.apps import RaterConfig
import numpy as np
import re

# Create your views here.
def home(request):
    return render(request,'rater/home.html')


#https://stackoverflow.com/questions/42359112/access-form-data-of-post-method-in-django/42359355
def results(request):
    if request.method == "POST":
        review_text = request.POST.get("review_text")

        processed_text = process_string(review_text)

        print("asd",processed_text)

        #generate feature matrix for the review using TF-IDF Scheme
        feature_matrix = RaterConfig.model_tfidf.transform([processed_text])

        ##get predictions from all 10 classifiers
        predictions = [model.predict(feature_matrix) for model in RaterConfig.ml_model_list]

        #convert predictions form list to numpy array
        predictions_ar = np.array(predictions)

        #sum predictions from all models to get rating
        final_rating = str(np.sum(predictions_ar))

        sentiments = ["Positive" if pred == 1 else "Negative" for pred in predictions]

        #create_string to be printed
        name_pred = [ [name,sentiment]  for name,sentiment in zip(RaterConfig.ml_model_names,sentiments) ]


        #create a context to pass along
        context = {
            "name_pred": name_pred,
            "final_rating": final_rating
        }

        #print(rating)


    return render(request,'rater/results.html',context)


def process_string(text):
    """This function returns a processed list of words from the given text
    
    This function removes html elements and urls using regular expression, then
    converts string to list of workds, them find the stem of words in the list of words and
    finally removes stopwords and punctuation marks from list of words.
    
    Args:
        text(string): The text from which hrml elements, urls, stopwords, punctuation are removed and stemmed
        
    Returns:
        clean_text(string): A text formed after text preprocessing.
    """
    
    #remove any urls from the text
    text = re.sub(r"https:\/\/.*[\r\n]*","",text)
    
    #remove any urls starting from www. in the text
    text = re.sub(r"www\.\w*\.\w\w\w","",text)
    
    #remove any html elements from the text
    text = re.sub(r"<[\w]*[\s]*/>","",text)
    
    #remove prediods  marks
    text = re.sub(r"[\.]*","",text)
    
    #tokenize text
    text_tokens = RaterConfig.tokenizer.tokenize(text)
    
    
    cleaned_text_tokens = [] # a list to hold cleaned text tokens
    
    for word in text_tokens:
        if((word not in RaterConfig.english_stopwords) and #remove stopwords
            (word not in RaterConfig.english_punctuations)): #remove punctuation marks
                
                stemmed_word = RaterConfig.porter_stemmer.stem(word) #get stem of the current word
                cleaned_text_tokens.append(stemmed_word) #appened stemmed word to list of cleaned list
    
    #combine list into single string
    clean_text = " ".join(cleaned_text_tokens)
    
    return clean_text