---


---

<h1 id="disaster-response-pipeline-project">Disaster Response Pipeline Project</h1>
<h2 id="project-description">Project Description</h2>
<p>This is project is from Figure Eight, which provided us the real data that were sent during the disaster. Data contains messages sent during the disasters and the responses categorized into 36 pre-defined categories. With the given data, we need to first build an ETL pipeline which pre-processes, cleans the data and stores them into a SQL database. Then we build a machine learning pipeline to classify messages. There are 36 pre-defined categories, which includes Aid Related, Medical Help, Search And Rescue, etc. By classifying these messages, we can allow these messages to be sent to the appropriate disaster relief agency.</p>
<h2 id="folder-structure">Folder Structure</h2>
<p>![File Structure](file:///C:/Users/viswanathan.a/Desktop/FolderStructure.PNG)</p>
<h2 id="file-description">File Description</h2>
<ol>
<li>disaster_message.csv: Contains the original disaster messages</li>
<li>disaster_categories.csv: Contains the labels of the disaster messages</li>
<li>process_data.py: Runs the ETL pipeline to process data from both disaster_message.csv and disaster_categories.csv and load them into an SQLite database, DisasterResponse.db.</li>
<li>train_classifier.py: Runs the ML pipeline to classify the messages. The pipeline will build the model, optimize it using grid search and print the model’s evaluation. It will then save the classifier model.</li>
<li><a href="http://run.py">run.py</a>: Script to run the web app.</li>
</ol>
<h2 id="instruction">Instruction</h2>
<ol>
<li>
<p>Run the following commands in the project’s root directory to set up your database and model.<br>
- To run ETL pipeline that cleans data and stores in database<br>
<code>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db</code></p>
<ul>
<li>To run ML pipeline that trains classifier and saves<br>
<code>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl</code></li>
</ul>
</li>
<li>
<p>Run the following command in the app’s directory to run your web app.<br>
<code>python run.py</code></p>
</li>
<li>
<p>Go to <a href="http://127.0.0.1:5000/">http://127.0.0.1:5000/</a></p>
</li>
</ol>
<h2 id="screenshots-from-webapp">Screenshots from Webapp</h2>
<p><img alt="enter image description here" src="C:%5CWork%5CLearnings%5COnline%20Courses%5CData%20Scientist%20Nano%20Degree%20Program%5CData%20Engineering%5Cresults%5Cnewplot%20%281%29.png"></p>
<p><img alt="enter image description here" src="C:%5CWork%5CLearnings%5COnline%20Courses%5CData%20Scientist%20Nano%20Degree%20Program%5CData%20Engineering%5Cresults%5Cnewplot%20%282%29.png"></p>
<p><img alt="enter image description here" src="C:%5CWork%5CLearnings%5COnline%20Courses%5CData%20Scientist%20Nano%20Degree%20Program%5CData%20Engineering%5Cresults%5Cnewplot%20%283%29.png"></p>
<p><img alt="enter image description here" src="C:%5CWork%5CLearnings%5COnline%20Courses%5CData%20Scientist%20Nano%20Degree%20Program%5CData%20Engineering%5Cresults%5CWebapp-Screenshot.PNG"></p>
<h2 id="references">References</h2>

