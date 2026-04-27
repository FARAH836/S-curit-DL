"""
=============================================================================
  Projet : Détection d'anomalies IoT par approche hybride ML + Deep Learning
  Datasets : N-BaIoT 2018 & UNSW-NB15
  Modèles  : H1 = RF + AutoEncoder | H2 = LSTM + RF
=============================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ─── Scikit-learn ───────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.impute import SimpleImputer
