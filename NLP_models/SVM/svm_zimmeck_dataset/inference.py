import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pickle
import ast
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

def match_after_label_mapping(pred,target):
    # uncomment if test is other set
    # pred,target=target,pred
    
    # # from zimmeck to our corpus
    if pred==0 and target==0:
        return True
    if pred!=0 and target==1:
        return True
    if pred>=1 and pred<=16 and target==14:
        return True
    if pred>=17 and target<=22 and target==1:
        return True
    if pred>=23 and pred<=46 and target==13:
        return True
    if pred>=47 and pred<=58 and target==15:
        return True
    return False

    # from liushaung to our corpus
    # if pred==target:
    #     return True
    # if pred==1 and target>=11 and target<=19:
    #     return True
    # if pred==0 and target==20:
    #     return True
    # return False

def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)

if __name__=='__main__':
    # Loading Label encoder
    labelencode = pickle.load(open('result/labelencoder_fitted.pkl', 'rb'))

    # Loading TF-IDF Vectorizer
    Tfidf_vect = pickle.load(open('result/Tfidf_vect_fitted.pkl', 'rb'))

    # Loading models
    SVM = pickle.load(open('result/svm_trained_model.sav', 'rb'))
    Naive = pickle.load(open('result/nb_trained_model.sav', 'rb'))


    # Inference
    infer_dataset=pd.read_csv('../policy_all_in_one_filter_purged.csv', encoding='utf-8')
    infer_dataset.dropna(inplace=True)

    predictions_SVM=[]
    svm_pred_labels=[]
    svm_correct=0
    nb_pred_labels=[]
    nb_correct=0
    target=infer_dataset['label']
    for sentence in tqdm(infer_dataset['review']):
        sample_text_processed = text_preprocessing(sentence)
        sample_text_processed_vectorized = Tfidf_vect.transform([sample_text_processed])

        prediction_SVM = SVM.predict(sample_text_processed_vectorized)
        prediction_Naive = Naive.predict(sample_text_processed_vectorized)
        
        target_label=target[infer_dataset.index[infer_dataset['review'] == sentence].tolist()[0]]

        svm_label=labelencode.inverse_transform(prediction_SVM)[0]
        svm_pred_labels.append(svm_label)
        if match_after_label_mapping(svm_label,target_label):
            svm_correct+=1

        nb_label=labelencode.inverse_transform(prediction_Naive)[0]
        nb_pred_labels.append(nb_label)
        if match_after_label_mapping(nb_label,target_label):
            nb_correct+=1
        # print("Prediction from SVM Model:", svm_label)
        # print("Prediction from NB Model:", nb_label)

    print("SVM Accuracy Score -> ", svm_correct/len(infer_dataset['review']))
    report =classification_report(svm_pred_labels,infer_dataset['label'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('zimmeck_our_report_svm.csv',index=True)
    print(report)

    print("NB Accuracy Score -> ", nb_correct/len(infer_dataset['review']))
    report =classification_report(nb_pred_labels,infer_dataset['label'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('zimmeck_our_report_nb.csv',index=True)
    print(report)
