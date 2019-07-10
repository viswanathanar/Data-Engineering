# Disaster Response Pipeline Project

## Project Description
This is project is from Figure Eight, which provided us the real data that were sent during disaster. Data contains messages sent during the disasters and the responses categorized into 36 pre-defined categories. With the given data, we need to first build an ETL pipeline which pre-processes, cleans the data and stores them into a SQL database. Then we build a machine learning pipeline to classify messages. There are 36 pre-defined categories, which includes Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency.

## Folder Structure

![](/FolderStructure.PNG)


## Instruction
*To run ETL pipeline that cleans data and stores in database: 
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db*
*To run ML pipeline that trains classifier and saves model:
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl*

## Installation
Provide code examples and explanations of how to get the project.

## Tests
Describe and show how to run the tests with code examples.

## References
