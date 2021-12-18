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
    ├── Data                    		# Contains used for NLP task
    │   ├── result.json         		# Main file for the data
    │   ├── contacts            		# Contact Information extracted from telegram
    │   	├──  contact_4.vcard 
    │── Images            			# Images Folder
    │   ├── Fig_1.png         			# Total no of english messages from May 1 to 15
    │   ├── Fig_1_before_non_english.png 	# Total no of english messages from May 1 to 15 (inc non English)
    │   ├── Fig_2.png         			# Sentiments from May 1 to 15
    │   ├── Fig_2_before_non_english.png        # Sentiments from May 1 to 15 (inc non-English)
    │   ├── Fig_3.png         			# Average Sentiments from May 1 to 15
    ├── AdvancedTrainingSentimentAnalysisModels.ipynb
    ├── environment_ind.yml
    ├── environment.yml
    ├── GraphPlotting.ipynb
    ├── Preprocessed_data.csv
    ├── Preprocessing+Predictions.ipynb
    ├── README.md
    ├── requirements.txt
    └── SentimentAnalysisforTelegram.py


## Introduction 

## Data Extraction

## Data Cleaning

## Data Preprocessing

## Sentiment Classification (using NLTK)

## Further Scope


