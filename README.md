# Spring2022SentimentAnalysis

This is a brief summary of the work done for this assignment. To create the environment for running this program please type in the following in your command prompt: 

''''''

This repository contains a brief summary of the assignment where the The specific task was to collect a dataset and perform an analysis on it. To build this application, we wee supposed to crawl Telegram messages, filter non-English messages, and compute the average sentiment over time.

This README will cover the following steps of our own data pipeline. 
  
  * Requirements and Environment
  * Introduction
  * Data Extraction
  * Data Cleaning
  * Data Preprocessing
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

## Data Cleaning

## Data Preprocessing

## Sentiment Classification (using NLTK)

## Summary of Result

## Further Scope


