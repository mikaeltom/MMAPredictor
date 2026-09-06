# Current Trends in Artificial Intelligence

Authors : Thomas Josephy, Arkadiusz Zaleski, Mikael Tom

Date : 10 June 2026

# Project Video Demonstration
A short video demonstration is available, showing the system in action along with a brief overview of the installation process.

Here is the link : [https://youtu.be/Q](https://youtu.be/Q)

# Description of the project
We present a Multi-Agent system to predict MMA fight outcomes, examine historical data, analyze the news, evaluate bookmakers advices and generate explainable predictions using Diverse Counterfactual Explanations (DiCE).

# Features
The user can either ask a chatbot for the prediction or observe the calendar and click for a particular fight to predict it.

# Before running the code

## Prerequisites
Python 3.10+

## Install librairies
This system rely on multiple standard libraries : `streamlit`, `spacy`, `openai`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `dice-ml`, `requests`. Please install them using the following command :
```bash
pip install -r requirements.txt
```
OR : 
```bash
pip install streamlit spacy openai pandas numpy xgboost scikit-learn dice-ml requests
```

## Install Spacy Model
In addition, we use Spacy model for word extractions. Please be sure installing it by using the following command : 
### On Linux/Mac
```bash
python3 -m spacy download en_core_web_sm
```
### On Windows
```bash
python -m spacy download en_core_web_sm
```

# How to run the program ?
Given the context, this app is not online so API tokens are needed. While passing raw tokens via the terminal is not a standard best practice for production security, it is the only way for you to easily run this code locally using our provided API keys. For safety reasons, these keys are only **free API keys** that **expire in about a month** and are **not linked** to any **bank account** or **personal data** (they are created with dummy emails). So, sharing them does not pose important risks, but it is still not a good practice and should be avoided in real-world scenarios.

## First : Set up the key 
We use the APIs : [Tavily](https://www.tavily.com) & [The Odds API](https://the-odds-api.com) and `gpt-4o-mini` for the LLM calls.
### Linux/Mac
```bash
export GITHUB_TOKEN=<YOUR TOKEN>
export TAVILY_API_KEY=<YOUR TOKEN>
export BET_TOKEN=<YOUR TOKEN>
```
### Windows
```bash
set GITHUB_TOKEN=<YOUR TOKEN>
set TAVILY_API_KEY=<YOUR TOKEN>
set BET_TOKEN=<YOUR TOKEN>
```

## Then : Run the code
```bash
streamlit run main.py
```

Or if it is not working (depending on the system), please try :

```bash
python3 -m streamlit run main.py
```

(When it is the first time you run it, it will ask you for an email to subscribe to their newsletter, just press enter to ignore that.)

After that, **open your browser** at [http://localhost:8501](http://localhost:8501)

## Problems ? 

### The API key is not being picked up ?
Make sure to set the environment variable in the **same terminal session** where you run `streamlit run main.py` or `python3 -m streamlit run main.py`.

### Port already in used ?
Please run the following command instead :
```bash
streamlit run main.py --server.port 8502
```

OR 

```bash
python3 -m streamlit run main.py --server.port 8502
```

You can specify another port (of your choice) instead of `<port>` :
```bash
streamlit run main.py --server.port <port>
```

OR 

```bash
python3 -m streamlit run main.py --server.port <port>
```

## Dataset : 
As it is important to give credit, the datasets we use are the following : 

* [large_dataset.csv](Data/large_dataset.csv) : [https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024](https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024)
* [fighter_stats.csv](Data/fighter_stats.csv) : [https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024](https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024)
* [fighter_recent.csv](Data/fighter_recent.csv) : [https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025?select=fighter_details.csv](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025?select=fighter_details.csv)
* [historical_data.csv](Data/historical_data.csv) : [https://www.kaggle.com/datasets/rajaisrarkiani/ufc-fights-and-fighter-stats-dataset?select=Fights+Data.csv](https://www.kaggle.com/datasets/rajaisrarkiani/ufc-fights-and-fighter-stats-dataset?select=Fights+Data.csv)
