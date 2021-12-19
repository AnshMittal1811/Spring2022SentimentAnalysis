# Sentiment Analysis

This is a brief summary of the work done for this assignment. To create the environment for running this program please type in the following in your command prompt: 

```shell
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```

Or you can create the environment for this repository using the ```YML config``` file for the environment on anaconda prompt. This can be done as follows. 

```shell
  conda env create -f environment.yml
```

This repository contains a brief summary of the assignment where the The specific task was to collect a dataset and perform an analysis on it. To build this application, we were supposed to crawl Telegram messages, filter non-English messages, and compute the average sentiment over time.

This README will cover the following steps of our own data pipeline. 
  
  * Requirements and Environment
  * Introduction
  * Data Extraction
  * Data Preprocessing and Cleaning
  * Sentiment Classification (using NLTK)
  * Making Predictions
  * Summary of Result
  * Further Scope
    1. Fine-tuning a model
    2. Sentiment Classification (using Advanced Models)

## Requirements and Environment

#### Clone the repository

First clone the repository onto your local machine. The command to do this is given below: 

```shell
   git clone https://github.com/sociometrik/planet-rep.git 
```

This will create a folder called **Spring2022SentimentAnalysis** on your machine with the Jupyter notebook scripts and other python files and graphs along with dataframe that was used for the work here.

#### Folder structure

The folder structure for the scripts is the same as in the GitHub repository. The structure diagram is given below: 


    .
    ├── ...
    ├── Data                    		            # Contains used for NLP task
    │   ├── result.json         		            # Main file for the data
    │   ├── contacts            		            # Contact Information extracted from telegram
    │   	├──  contact_4.vcard 
    │── Images            		                  # Images Folder
    │   ├── Fig_1.png         		 	            # Total no of english messages from May 1 to 15
    │   ├── Fig_1_before_non_english.png        # Total no of english messages from May 1 to 15 (inc non English)
    │   ├── Fig_2.png         		              # Sentiments from May 1 to 15
    │   ├── Fig_2_before_non_english.png        # Sentiments from May 1 to 15 (inc non-English)
    │   ├── Fig_3.png         	                # Average Sentiments from May 1 to 15
    ├── AdvancedTrainingSentimentAnalysisModels.ipynb
    ├── environment_ind.yml
    ├── environment.yml
    ├── GraphPlotting.ipynb
    ├── Preprocessed_data.csv
    ├── Preprocessing+Predictions.ipynb
    ├── README.md
    ├── requirements.txt
    ├── SentimentAnalysisforTelegram.py
    └── ...


## Introduction 

Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them.

The requirements for this project were as follows. 
  * **Create a GitHub repo** with a **Python 3.6+** environment for this project and start a requirements.txt file to capture the packages required to run your code. This repository is where you can upload all the files pertaining to your submission.
  * **Export the Telegram messages** from [here](https://t.me/CryptoComOfficial) from May 1 to and including May 15, 2021. Export this data as JSON. No coding is required here, but include the JSON file with your submission. _**Warning:** Telegram may make you wait a period of time before allowing you to export the chat messages. You should start the process of downloading these messages immediately upon receiving this assignment._
  * **Pre-process** the data. Remove non-English messages. From these, keep only messages that mention either “SHIB” or “DOGE.” Use the [tqdm](https://tqdm.github.io/) package to display progress on the terminal. Use [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/) for your python code.
  * **Compute the sentiment** of each message. We encourage you to use an off-the-shelf library, but you may create your own if you feel it is appropriate. Document your choice of sentiment analysis approach in your summary.
  * **Plot** the number of messages per day and the average sentiment per day using the [plotly](https://plotly.com/python/) visualization library. Please include a screenshot of this plot in your deliverable.
  * **Create a README.md file with a summary of your results** and provide high-level documentation of your code as well as instructions on how to run your code in order to reproduce the results.


## Data Extraction

Data was extracted from the telegram desktop app and it gave a ```.json``` file after 24 hours as it was the required waiting time for the data to get downloaded from the Telegram Desktop App. First, the json file was opened and loaded using the ```json``` package of python. This has been depicted below.

```python
   messages = open('./Data/result.json', 'r', encoding='utf8')
   data = json.load(messages) 
```

Then, the data was extracted from the JSON dictionary according to 2 keys (i.e. ```Date``` and ```Text```) using the ```dataset_extraction()``` method.

## Data Preprocessing and Cleaning
The data preprocessing and data cleaning was divided into several phases mentioned below.
 * Demojizing the text
 * Converting accented characters
 * Removing Case Sensitivity
 * Removing HTMLs and URLs
 * Removing Extra Spaces between words
 * Expansion of Contractions in words
 * Removing Stop words
 * Lemmatizing the words present
 * Removing non-English messages
 * Spelling Corrections (using Levenshtein Distance)

### Demojizing the text
Since this text was based on telegram messages, there were several words which could have been included in non-English words and hence lead us to misconception that there were actually more number of words, and hence messages which weren't from the English language. And to resolve this, we used the following function from the library.

```python
  import emoji
  emoji.demojize(txt)
```

### Converting accented characters
There are several words like **Café** which can result in our language being considered as non-English (for e.g., German, or French). Hence, we convert the accented characters which lead to such misunderstandings using the following.

```python
  import unidecode
  unidecode.unidecode(txt)
```

### Removing Case Sensitivity
Case Sensitivity is one of the issues that can rise when we try to recognize the words from pre-build corpus. This can be resolved very easily as has been given below.

```python
  dataframe['col1'] = dataframe['col1'].apply(lambda txt: str(txt).lower())
```

### Removing HTMLs and URLs
We further remove HTMLs and URLs to make our messages text more easier for the algorithm to decide if it falls under the `English' language. This has been done as follows. 

```python
    from bs4 import BeautifulSoup
    dataframe['col1'] = dataframe['col1'].apply(lambda txt: re.sub(r"http\S+", "", txt))
    dataframe['col1'] = dataframe['col1'].apply(lambda txt: BeautifulSoup(txt, 'lxml').get_text())
```

### Removing Extra Spaces between words
We remove extra spaces coming in the words, so, that it doesn't lead to any inconsistencies while trying to predict sentiments and removing the messages from English language. This has been done as follows. 

```python
    import re
    dataframe['col1'] = dataframe['col1'].apply(lambda txt: re.sub(" +"," ", txt))
```

### Expansion of Contractions in words
We expand the words such as  ```I'll``` to ```I will```. This is again done to compare the words easily to our corpus of words in NLTK. The following represents the method of doing the same in our pipeline. 

```python
    import contractions
    dataframe['col1'] = dataframe['col1'].apply(lambda txt: contractions.fix(txt))
```

### Removing Stop words
We remove certain stop words but some of the stop words have been excluded from teh list of stop words that NLTK provides us with. This has been depicted in the code snippet below. 

```python
  from nltk.corpus import stopwords

  def Remove_Stopwords(message, stop_words_list):
    tokens = message.split(" ")
    clean_message = [word for word in tokens if not word in stop_words_list]
    return [(" ").join(clean_message)]

  deselect_stop_words = ['not', 'nor', 'no', 'against', 'don', "don't", 
        'should', "should've", 'aren', "aren't", 'couldn', 
        "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
        'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 
        'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
        'wouldn', "wouldn't"]

  stop_words_list = set([stopwords.words('english').remove(word) for word in deselect_stop_words])
  dataframe['col1'] = dataframe['col1'].apply(lambda txt: Remove_Stopwords(txt, stop_words_list))
```

### Lemmatizing the words present
Finally, we lemmatize the words so that we can get a good match in the corpus from ```NLTK```. This is one of the most impotant steps as there might be some forms of words that are there in the ```wordnet.all_lemma_names()``` or ```words.words()``` in ```NLTK``` but there equivalent forms in  the sentences aren't there in the corpus. An instance of this can be the word `inestment' which was there in our corpus but not in these word lists and hence it was getting removed repeatedly. The following code snippet takes care of lemmatizing the corpus. 

```python
  def lemmatize(message):
    word_lemma = WordNetLemmatizer()
    lemmatize_word = [word_lemma.lemmatize(word) for word in message]
    return (" ").join(lemmatize_word)

  tqdm.pandas()
  print("Lemmatizing Words in Messages...")
  dataframe['Messages']= dataframe['Messages'].progress_apply(lambda txt: lemmatize(txt))
```

### Removing non-English messages
Now, this is one of the vital part of the pipeline where we were supposed to remove all those sentences that were not there in the english language. We specifically did this step after extracting messages such as ```DOGE``` or ```SHIB``` from our corpus. This step has been done by using the function ```wordnet.synsets()``` and checking if the word lies in the ```words.words()``` and an extension of finance related words list. If majority of the words aren't present in the list, then the sentence is considered non-English and dropped. The majority is taken as there can be words which were misspelled while typing the messages on telegram.

### Spelling Corrections (using Levenshtein Distance)
After removing all non-English messages from our dataframe, we start correcting the grammatical mistakes made while writing messages word-by-word. This is done using levenshtein-distance and not Jaccard distance with N-grams (as there can be sentences where Intersection over union in J-distance gives us divide-by-zero error which had been encountered multiple times). The following code-snippet depicts the grammatical corrections made. 

```python
  def spell_check(message, correct_words):
    new_message = ""
    for word in message.split(" "):
      if word.isalpha() and (word not in correct_words) and (word.lower() not in ["doge", "dogecoin", "shibe", "shiba", "shib", "shiba inu"]):
        temp = [(edit_distance(word, w),w) for w in correct_words if w[0]==word[0]]
        new_message = new_message + sorted(temp, key = lambda val:val[0])[0][1] + " "
      else:
        new_message =  new_message + word + " "
    return new_message
```

## Sentiment Classification (using NLTK)
For the sentiment classification, we were trying to take two different approaches which have been mentioned as follows. 
1. Using ```NLTK``` pretrained ```SentimentIntensityAnalyzer()```
2. Using ```Machine Learning``` Models
3. Using ```BERT``` and ```Transformer-based``` models
The results depicted right now are those based on the 1st part and have been summarized as follows.

## Summary of Result
These are the results obtained after running the preprocessing pipeline and Sentiment Analysis model (from NLTK) on the preprocessed data. Figure 1 represents the total number os messages per day from May 1st to May 15th on a bar graph

![Figure 1](https://github.com/AnshMittal1811/Spring2022SentimentAnalysis/blob/master/Images/Fig_1.png)

Figure 2 represents the distribution of different sentiment over the messages (only english) throughout the duration from May 1st to May 15th on a stack bar plot. Here, the value for the threshold was taken 0.2 for the ```compound``` key of the dictional obtained from ```SentimentIntensityAnalyser().polarity_scores()``` and hence obtained the following graph.

![Figure 2](https://github.com/AnshMittal1811/Spring2022SentimentAnalysis/blob/master/Images/Fig_2.png)

Figure 3 represents the average sentiment calculated over the messages  (only english) throughout the duration from May 1st to May 15th on a bar plot where the height of the bar is the total number of messages on that day. Here, the threshold for the positive sentiment was taken to be 0.15 and for the negative sentiment it was taken as -0.15. Any values that were lying between these 2 ranges were taken to be Neutral and hence obtained the following graph.

![Figure 3](https://github.com/AnshMittal1811/Spring2022SentimentAnalysis/blob/master/Images/Fig_3.png)

 
## Further Scope
Obtaining the results from 2 and 3.

