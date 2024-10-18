# Final Project - Spotify Streams 2023 




## Description
This project aims to predict which songs are likely to perform well in terms of streaming numbers across different platforms, such as Spotify, Apple Music, and Shazam. By analyzing various features like chart rankings, streams, and playlist appearances, the project uses machine learning algorithms to identify patterns that make certain songs more popular than others.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

- [Features](#features)
- [Tests](#tests)
- [Contact](#contact)

## Installation
import pandas as pd
import itertools
import random
import datetime as dt
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, inspect, text
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score

## Usage
Machine learning algorithms and statistical tests are applied on a basket of features used to predict a target.

## Credits
Paola Van De Weyer, Leomar Crowal,Noah Woltman, Sadaf Hakim


## Features
 Machine learning :Models Linear Regression, Random Forest, K-Means, PCA 9:37 
 Metrics Use: R-Square Score, Confusion Matrix, Accuracy Score, Cross Validation Score, Mean Squared Error

## Next Steps
Revisit our data source by exploring Spotify API requests. Examine historical stream data for deeper trend analysis. Additionally, analyze correlations between Spotify streams and social media engagement to identify any underlying impact.

## Contact
If there are any questions or concerns, I can be reached at:
##### [email: paola.guigou@gmail.com, lcrowalcrowal@gmail.com, npwoltman@gmail.com,sadaf.hakim99@gmail.com]
https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023
https://www.forbes.com/sites/conormurray/2023/11/28/spotify-wrapped-2023-comes-soon-heres-how-it-became-a-viral-and-widely-copied-marketing-tactic/
https://www.musicnotes.com/blog/25-quotes-from-musicians-for-musicians/


