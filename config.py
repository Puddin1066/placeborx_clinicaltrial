import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'PlaceboRx_Validation_Bot/1.0')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PATENTSVIEW_API_KEY = os.getenv('PATENTSVIEW_API_KEY', 'noCNxhqh.63XyW1sKoK9tFQ9xPBa7nenbaIJppIcP')

# Clinical Trials API
CLINICAL_TRIALS_API = "https://classic.clinicaltrials.gov/api/query/study_fields"

# PatentsView API
PATENTSVIEW_API_BASE = "https://api.patentsview.org/patents/query"

# Target subreddits for market validation
SUBREDDITS = [
    'chronicpain', 'CFS', 'fibromyalgia', 'anxiety', 'depression',
    'mentalhealth', 'wellness', 'meditation', 'mindfulness'
]

# Keywords for filtering relevant posts
KEYWORDS = [
    'nothing works', 'need help', 'non-pharma', 'alternative', 'desperate',
    'tried everything', 'chronic', 'symptom', 'treatment', 'therapy'
]

# OLP search terms for clinical trials
OLP_SEARCH_TERMS = [
    'open-label placebo', 'open label placebo', 'digital placebo',
    'app placebo', 'online placebo', 'digital therapeutic'
]

# Patent search terms for digital therapeutics
PATENT_SEARCH_TERMS = [
    'digital therapeutic', 'digital placebo', 'mobile health',
    'telemedicine', 'remote monitoring', 'digital intervention',
    'app-based treatment', 'online therapy', 'digital medicine',
    'mobile application', 'software therapeutic', 'digital health'
] 