import json
from tqdm import tqdm
import numpy as np
import unidecode
import nltk
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords, words
from nltk.metrics.distance  import edit_distance
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import contractions
from bs4 import BeautifulSoup
import pandas as pd
import emoji
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go

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
            if (word.isalpha() 
                    and (word not in correct_words) 
                    and (word.lower() not in ["doge", "dogecoin", "shibe", "shiba", 
                                              "shib", "shiba inu"])):
                temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
                new_message = new_message + sorted(temp, key = lambda val:val[0])[0][1] + " "
            else:
                new_message =  new_message + word + " "
        return new_message

    tqdm.pandas()
    print("Performing Spelling Corrections...")
    
    slangs = ["doge", "dogecoin", "dogecoins", "shib", "shiba", "shiba inu", "shibe inu",
              "dollar", "dolar", "$", "ps", "p.s.", "app", "money", "tarde", "telegram",
              "whatsapp", "buy", "issue", "crypto", "usdc", "bank", "account", "portfolio",
              "Elon", "Musk", "shibaa", "profit", "cro", "€", "inr", "mill", "cdc", "tbh", 
              "hi", "hey", "plz", "wbu", "%", "crypto.com", "email", "usdt", "cent", "ct", 
              "mil", "ppl", "btc", "curr"]

    a = [w for w in wordnet.all_lemma_names()]
    a = list(set(a).union(set(slangs)))
    correct_words = list(set(words.words()).union(set(a)))    
    dataframe['Messages'] = dataframe['Messages'].progress_apply(lambda txt: spell_check(txt, correct_words))
    return dataframe

def removing_non_english(dataframe): 
    def word_in_english(message, correct_words): 
        st = []
        for word in message.split(" "): 
            if (wordnet.synsets(word)
                    or word.lower() in correct_words):
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
              "dollar", "dolar", "$", "ps", "p.s.", "app", "money", "tarde", "telegram",
              "whatsapp", "buy", "issue", "crypto", "usdc", "bank", "account", "portfolio",
              "Elon", "Musk", "shibaa", "profit", "cro", "€", "inr", "mill", "cdc", "tbh", 
              "hi", "hey", "plz", "wbu", "%", "crypto.com", "email", "usdt", "cent", "ct", 
              "mil", "ppl", "btc", "curr"]
    
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

def make_simple_predictions(dataframe):
    def predict(messages): 
        sia = SentimentIntensityAnalyzer()
        if (sia.polarity_scores(messages)['compound'] > 0.25): 
            return "positive"
        elif (sia.polarity_scores(messages)['compound'] < -0.25):
            return "negative"
        else:
            return "neutral"
        
    tqdm.pandas()
    print("Predicting Sentiments in the Text...")
    dataframe['Sentiment'] = dataframe['Messages'].progress_apply(lambda txt: predict(txt))
    return dataframe

def average_sentiment(dataframe): 
    def cal_sentiment(day):
        val = (1.0*day['positive'] + (-1.0)*day['negative'])/day['Total Messages']
        thresh = 0.1
        if val > thresh: 
            return "positive"
        elif val < -thresh:
            return "negative"
        else: 
            return "neutral"
        
    def total_messages(day): 
        return day['negative'] + day['neutral'] + day['positive']
    
    tqdm.pandas()
    print("Calculating Average Sentiment...")
    dataframe['Total Messages'] = dataframe.progress_apply(total_messages, axis=1)
    dataframe['Average_sent'] = dataframe.progress_apply(cal_sentiment, axis=1)
    return dataframe

def graph_1(dataframe):
    fig = px.bar(dataframe, x='Date', y='Total Messages', color='Total Messages')
    fig.show()
    fig.write_image("Images/Fig_1")
    
def graph_2(dataframe):
    fig = px.bar(dataframe, x = 'Date', y = ['neutral','negative','positive'],
                 color_discrete_sequence=px.colors.qualitative.D3,
                 title="Sentiment-based for Telegram Cryptocoin")
    fig.show()
    fig.write_image("Images/Fig_2.png")

def final_graph(dataframe): 
    fig = fig2 = px.bar(final_df, x='Date', 
                        y='Total Messages', color = 'Average_sent',
                        title = 'Average Sentiment (Cryptocoin) over time')

    fig.show()
    fig.write_image("Images/Fig_3.png")
    
def main(): 
    # Opening Telegram JSON file
    messages = open('./Data/result.json', 'r', encoding='utf8')
    data = json.load(messages)
    
    # Extracting Dataset
    messages = dataset_extraction(data)
    
    df = convert_data_to_df(messages)
    # Preprocessing Data (Removing non-English sentences here, 
    # Removing words not having SHIB, or DOGE), Performing spell-check here
    # Demojizing texts, Stop Words Removal, Lemmatizing, etc. 
    df_pre = preprocessing(df)
    
    # For Graph 1
    modified_df = df_pre.groupby(['Date']).size().to_frame('Total Messages').reset_index()
    graph_1(modified_df)
    
    # Making Predictions
    mod_df = make_simple_predictions(df)
    
    # For Graph 2
    mod_df = df.groupby(['Date', 'Sentiment']).size().to_frame('Total Sentiment').reset_index()
    mod_df_1 = pd.pivot_table(mod_df, index = 'Date', columns='Sentiment', values='Total Sentiment').reset_index()
    mod_df_1 = mod_df_1.fillna(0)
    graph_2(mod_df_1)
    
    # Final Graph for Average Sentiment
    final_df = average_sentiment(mod_df_1)
    final_graph(final_df)
    
    
if __name__ == "__main__": 
    main()