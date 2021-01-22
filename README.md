# Disaster-Response-Pipeline-Project

## Installation:
Necessary libraries to work with the project are:
- numpy
- pandas
- sys
- sqlalchemy
- re
- nltk
- scikit-learn
- pickle
- flask
- plotly
- python version 3

## Project Motivation:
In this project, there is a data set containing real messages that were sent during disaster events. My objective is creating a machine learning pipeline to categorize these events and showing some visualisations of the data and the analysis of performance of the models in the web application. Moreover, the user can write a new message and make classfication of it.

## File Descriptions:
The dataset is provided by [Figure Eight] (https://appen.com) consisting of two csv files (disaster_messages.csv and disaster_categories.csv).

- app/template/master.html  # main page of web app
- app/template/go.html  # classification result page of web app
- app/run.py  # Flask file that runs app

- data/disaster_categories.csv  # categories data to process 
- data/disaster_messages.csv  # messages data to process
- data/process_data.py  # ETL pipeline
- data/DisasterResponse.db  # database to save clean data to

- models/train_classifier.py  # ML pipeline
- models/classifier.pkl  # saved model 

- README.md

## Instructions:
- ETL Pipeline: run code "python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db"
- ML Pipeline: run code "python train_classifier.py ../data/DisasterResponse.db classifier.pkl"
- Web App: run code "python run.py"

## Results and Findings:

![Image](https://drive.google.com/uc?export=view&id=1lS9Ro4QRcBdgFfgcGhsHWf2vZOzifkeT)

## Licensing, Authors, Acknowledgements:
- [Figure Eight] (https://appen.com)
- [scikit-learn] (https://scikit-learn.org/stable/index.html)
