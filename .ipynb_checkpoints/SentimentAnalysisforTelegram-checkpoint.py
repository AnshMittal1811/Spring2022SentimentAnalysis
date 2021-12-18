import json
# import glob
from tqdm import tqdm
import numpy as np
import unidecode
import re
import contractions
from bs4 import BeautifulSoup
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, words
from nltk.metrics.distance import edit_distance
import emoji

def dataset_extraction(data):
    df = []
    crypto = ['shib', 'doge', 'shiba', 'dogecoin']
    print("Extracting Messages from Telegram Data...")
    for i in tqdm(range(len(data['messages']))):
        frame = []
        st = data['messages'][i]
        
        if type(st['text']) == str:
            if any(curr in st['text'].lower() for curr in crypto):
                frame.append(st['date'].split("T")[0])
                frame.append(st['text'])

        if type(st['text']) == list: 
            b = str()
            
            for j in st['text']: 
                if type(j) == dict:
                    a = j['text']
                    b += a
                else: 
                    b += j
                    
            if any(curr in b.lower() for curr in crypto):
                frame.append(st['date'].split("T")[0])
                frame.append(b)
        
        if len(frame): 
            df.append(frame)
        
    return df

def convert_data_to_df(messages): 
    df = pd.DataFrame(messages, columns = ['Date', 'Messages'])
    return df

def preprocessing(df):
    df = demojizing(df)
    df = convert_accented_chars(df)
    df = remove_case_sensitive(df)
    df = remove_htmls_and_urls(df)
    df = remove_extra_spaces_between_words(df)
    df = expand_contractions(df)
    df = remove_stop_words(df)
    df = lemmatization(df)
    df = removing_non_english(df)
    df = spelling_corrections(df)
    return df

def demojizing(dataframe): 
    tqdm.pandas()
    print("Decoding Emojis in Text...")
    dataframe["Messages"] = dataframe["Messages"].progress_apply(lambda txt: emoji.demojize(txt))
    return dataframe

def convert_accented_chars(dataframe): 
    tqdm.pandas()
    print("Converting Accented Characters...")
    dataframe["Messages"] = dataframe["Messages"].progress_apply(lambda txt: unidecode.unidecode(txt))
    return dataframe

def remove_case_sensitive(dataframe):
    tqdm.pandas()
    print("Removing Case Sensitive Characters...")
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: str(txt).lower())
    return dataframe

def remove_htmls_and_urls(dataframe):
    tqdm.pandas()
    print("Removing HTMLs and URLs...")
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: re.sub(r"http\S+", "", txt))
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: BeautifulSoup(txt, 'lxml').get_text())
    return dataframe
    
def remove_extra_spaces_between_words(dataframe):
    tqdm.pandas()
    print("Removing Extra Whitespaces...")
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: re.sub(" +"," ", txt))
    return dataframe

def expand_contractions(dataframe):
    tqdm.pandas()
    print("Expanding Contractions...")
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: contractions.fix(txt))
    return dataframe

def spelling_corrections(dataframe):
    
    def spell_check(message, correct_words):
        new_message = ""
        for word in message.split(" "):
            if word.isalpha() and (word not in correct_words) and (word.lower() not in ["doge", "dogecoin", "shibe", "shiba", "shib", "shiba inu"]):
                temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
                new_message = new_message + sorted(temp, key = lambda val:val[0])[0][1] + " "
            else:
                new_message =  new_message + word + " "
        return new_message

    tqdm.pandas()
    print("Performing Spelling Corrections...")
    
    slangs = ["doge", "dogecoin", "dogecoins", "shib", "shiba", "shiba inu", "shibe inu", 
          "dollar", "dolar", "$", "ps", "p.s.", "app", "money", "tarde", "telegram", "whatsapp", 
          "buy", "issue", "crypto", "usdc", "bank", "account", "portfolio", "Elon", "Musk", "shibaa",
          "profit", "cro", "€", "inr", "mill", "cdc", "tbh", "hi", "hey", "plz", "wbu", "%",
          "crypto.com", "email", "usdt", "cent", "ct", "mil", "ppl", "btc", "curr"]

    a = [w for w in wordnet.all_lemma_names()]
    a = list(set(a).union(set(slangs)))
    correct_words = list(set(words.words()).union(set(a)))    
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: spell_check(txt, correct_words))
    return dataframe

def removing_non_english(dataframe): 
    def word_in_english(message, correct_words): 
        st = []
        for word in message.split(" "): 
            if wordnet.synsets(word) or word.lower() in correct_words:
                st.append(1)
            else: 
                st.append(0)

        cutoff_value = 0.45
        
        if sum(st)/len(st) >= cutoff_value: 
            return message
        else: 
            return ""
        
    tqdm.pandas()
    print("Checking English Messages...")
    slangs = ["doge", "dogecoin", "dogecoins", "shib", "shiba", "shiba inu", "shibe inu", 
              "dollar", "dolar", "$", "ps", "p.s.", "app", "money", "tarde", "telegram", "whatsapp", 
              "buy", "issue", "crypto", "usdc", "bank", "account", "portfolio", "Elon", "Musk", "shibaa",
              "profit", "cro", "€", "inr", "mill", "cdc", "tbh", "hi", "hey", "plz", "wbu", "%",
              "crypto.com", "email", "usdt", "cent", "ct", "mil", "ppl", "btc", "curr"]
    
    correct_words = list(set(words.words()).union(set(slangs)))    
    dataframe["Messages"] = dataframe["Messages"].progress_apply(lambda txt: word_in_english(txt, correct_words))
    return dataframe[dataframe['Messages'] > ""]

def remove_stop_words(dataframe): 

    def Remove_Stopwords(message, stop_words_list):
        tokens = message.split(" ")
        clean_message = [word for word in tokens if not word in stop_words_list]
        return [(" ").join(clean_message)]

    tqdm.pandas()
    print("Removing Stop Words...")
    deselect_stop_words = ['not', 'nor', 'no', 'against', 'don', "don't", 
          'should', "should've", 'aren', "aren't", 'couldn', 
          "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
          'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
          'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 
          'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
          'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
          'wouldn', "wouldn't"]

    stop_words_list = set([stopwords.words('english').remove(word) for word in deselect_stop_words])
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: Remove_Stopwords(txt, stop_words_list))
    return dataframe

def lemmatization(dataframe): 
    def lemmatize(message):
        word_lemma = WordNetLemmatizer()
        lemmatize_word = [word_lemma.lemmatize(word) for word in message]
        return (" ").join(lemmatize_word)
    
    tqdm.pandas()
    print("Lemmatizing Words in Messages...")
    dataframe['Messages']= dataframe['Messages'].progress_apply(lambda txt: lemmatize(txt))
    return dataframe


def main(): 
    
    messages = open('./Data/result.json', 'r', encoding='utf8')
    data = json.load(messages)

    messages = dataset_extraction(data)
    df = convert_data_to_df(messages)
    
    df_pre = preprocessing(df)
    
if __name__ == "__main__": 
    main()