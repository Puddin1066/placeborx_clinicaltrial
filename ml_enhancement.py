import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime, timedelta
import joblib
import os

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# NLP libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob

# Statistical libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

from enhanced_config import CONFIG

class MLEnhancementEngine:
    """Advanced machine learning and analytics engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.vectorizers = {}
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
    def _initialize_nlp_models(self):
        """Initialize pre-trained NLP models"""
        try:
            # Sentiment analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=CONFIG.market.sentiment_model,
                return_all_scores=True
            )
            
            # Emotion detection
            self.emotion_analyzer = pipeline(
                "text-classification",
                model=CONFIG.market.emotion_model,
                return_all_scores=True
            )
            
            self.logger.info("NLP models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize all NLP models: {e}")
            # Fallback to simpler models
            self.sentiment_analyzer = None
            self.emotion_analyzer = None
    
    def enhance_clinical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance clinical trials data with ML predictions"""
        enhanced_df = df.copy()
        
        # Predict trial success probability
        enhanced_df = self._predict_trial_success(enhanced_df)
        
        # Classify trial types
        enhanced_df = self._classify_trial_types(enhanced_df)
        
        # Extract semantic features from text
        enhanced_df = self._extract_clinical_text_features(enhanced_df)
        
        # Detect trial clusters
        enhanced_df = self._cluster_trials(enhanced_df)
        
        return enhanced_df
    
    def enhance_market_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance market data with advanced sentiment and predictive analytics"""
        enhanced_df = df.copy()
        
        # Advanced sentiment analysis
        enhanced_df = self._analyze_advanced_sentiment(enhanced_df)
        
        # Extract emotional patterns
        enhanced_df = self._analyze_emotions(enhanced_df)
        
        # Predict engagement potential
        enhanced_df = self._predict_engagement(enhanced_df)
        
        # Identify user personas
        enhanced_df = self._identify_user_personas(enhanced_df)
        
        # Extract linguistic features
        enhanced_df = self._extract_linguistic_features(enhanced_df)
        
        return enhanced_df
    
    def _predict_trial_success(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict trial success probability using ML"""
        if df.empty:
            return df
        
        try:
            # Create features for prediction
            features = []
            feature_names = []
            
            # Numerical features
            if 'enrollment' in df.columns:
                features.append(df['enrollment'].fillna(0))
                feature_names.append('enrollment')
            
            # Phase encoding
            if 'phase' in df.columns:
                phase_encoded = pd.get_dummies(df['phase'], prefix='phase').fillna(0)
                features.extend([phase_encoded[col] for col in phase_encoded.columns])
                feature_names.extend(phase_encoded.columns.tolist())
            
            # Text-based features
            if 'title' in df.columns and 'intervention' in df.columns:
                # Combine text for analysis
                text_data = (df['title'].fillna('') + ' ' + df['intervention'].fillna('')).tolist()
                
                # TF-IDF features
                if len(text_data) > 1:
                    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
                    text_features = vectorizer.fit_transform(text_data).toarray()
                    
                    for i in range(text_features.shape[1]):
                        features.append(text_features[:, i])
                        feature_names.append(f'text_feature_{i}')
                    
                    self.vectorizers['clinical_text'] = vectorizer
            
            if features:
                # Combine all features
                X = np.column_stack(features)
                
                # Create synthetic target based on existing evidence
                y = self._create_synthetic_success_target(df)
                
                if len(np.unique(y)) > 1:
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    
                    # Predict success probability
                    success_prob = model.predict_proba(X)[:, 1]
                    df['predicted_success_probability'] = success_prob
                    
                    # Save model
                    self.models['trial_success'] = model
                    
                    self.logger.info("Trial success prediction completed")
                else:
                    df['predicted_success_probability'] = 0.5  # Default neutral
                    
        except Exception as e:
            self.logger.warning(f"Could not predict trial success: {e}")
            df['predicted_success_probability'] = 0.5
        
        return df
    
    def _classify_trial_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify trials into categories using ML"""
        if df.empty or 'intervention' not in df.columns:
            return df
        
        try:
            # Define intervention categories
            intervention_categories = {
                'digital': ['app', 'online', 'digital', 'mobile', 'web', 'platform', 'telemedicine'],
                'pharmaceutical': ['drug', 'medication', 'pill', 'tablet', 'injection', 'infusion'],
                'behavioral': ['therapy', 'counseling', 'training', 'education', 'behavioral'],
                'device': ['device', 'stimulation', 'implant', 'monitor', 'sensor'],
                'placebo': ['placebo', 'sham', 'control', 'dummy']
            }
            
            # Classify interventions
            df['intervention_category'] = 'other'
            
            for category, keywords in intervention_categories.items():
                mask = df['intervention'].str.lower().str.contains('|'.join(keywords), na=False)
                df.loc[mask, 'intervention_category'] = category
            
            # Create confidence scores
            df['category_confidence'] = 1.0  # Could be enhanced with ML model
            
        except Exception as e:
            self.logger.warning(f"Could not classify trial types: {e}")
        
        return df
    
    def _extract_clinical_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract semantic features from clinical trial text"""
        if df.empty:
            return df
        
        try:
            text_columns = ['title', 'condition', 'intervention']
            existing_columns = [col for col in text_columns if col in df.columns]
            
            for column in existing_columns:
                text_data = df[column].fillna('')
                
                # Basic text statistics
                df[f'{column}_length'] = text_data.str.len()
                df[f'{column}_word_count'] = text_data.str.split().str.len()
                
                # Keyword presence
                placebo_keywords = ['placebo', 'sham', 'control', 'dummy']
                digital_keywords = ['app', 'online', 'digital', 'mobile', 'web']
                
                df[f'{column}_has_placebo_terms'] = text_data.str.lower().str.contains(
                    '|'.join(placebo_keywords), na=False
                ).astype(int)
                
                df[f'{column}_has_digital_terms'] = text_data.str.lower().str.contains(
                    '|'.join(digital_keywords), na=False
                ).astype(int)
                
        except Exception as e:
            self.logger.warning(f"Could not extract text features: {e}")
        
        return df
    
    def _cluster_trials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cluster similar trials using unsupervised learning"""
        if df.empty or len(df) < 3:
            return df
        
        try:
            # Select numerical features for clustering
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numerical_cols if col not in ['nct_id']]
            
            if len(feature_cols) > 0:
                # Prepare data
                cluster_data = df[feature_cols].fillna(0)
                
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(cluster_data)
                
                # Determine optimal number of clusters
                n_clusters = min(5, len(df) // 2)
                
                if n_clusters >= 2:
                    # Perform clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    df['trial_cluster'] = clusters
                    df['cluster_confidence'] = kmeans.transform(scaled_data).min(axis=1)
                    
                    # Save models
                    self.models['trial_clustering'] = kmeans
                    self.scalers['trial_clustering'] = scaler
                    
                    self.logger.info(f"Clustered trials into {n_clusters} groups")
                    
        except Exception as e:
            self.logger.warning(f"Could not cluster trials: {e}")
        
        return df
    
    def _analyze_advanced_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced sentiment analysis on market data"""
        if df.empty or 'title' not in df.columns:
            return df
        
        try:
            # Combine title and body for analysis
            text_data = df['title'].fillna('') + ' ' + df.get('body', '').fillna('')
            
            # Advanced sentiment analysis
            if self.sentiment_analyzer:
                sentiments = []
                for text in text_data:
                    if len(text.strip()) > 0:
                        result = self.sentiment_analyzer(text[:512])  # Limit text length
                        # Get the highest scoring sentiment
                        best_sentiment = max(result[0], key=lambda x: x['score'])
                        sentiments.append({
                            'sentiment_label': best_sentiment['label'],
                            'sentiment_score': best_sentiment['score']
                        })
                    else:
                        sentiments.append({
                            'sentiment_label': 'NEUTRAL',
                            'sentiment_score': 0.5
                        })
                
                df['advanced_sentiment'] = [s['sentiment_label'] for s in sentiments]
                df['sentiment_confidence'] = [s['sentiment_score'] for s in sentiments]
            else:
                # Fallback to TextBlob
                df['advanced_sentiment'] = text_data.apply(self._textblob_sentiment)
                df['sentiment_confidence'] = 0.7  # Default confidence
            
            # Calculate desperation and openness scores
            df = self._calculate_psychological_scores(df, text_data)
            
        except Exception as e:
            self.logger.warning(f"Could not perform advanced sentiment analysis: {e}")
        
        return df
    
    def _analyze_emotions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract emotional patterns from text"""
        if df.empty or 'title' not in df.columns:
            return df
        
        try:
            text_data = df['title'].fillna('') + ' ' + df.get('body', '').fillna('')
            
            if self.emotion_analyzer:
                emotions = []
                for text in text_data:
                    if len(text.strip()) > 0:
                        result = self.emotion_analyzer(text[:512])
                        # Get top emotion
                        best_emotion = max(result[0], key=lambda x: x['score'])
                        emotions.append({
                            'primary_emotion': best_emotion['label'],
                            'emotion_confidence': best_emotion['score']
                        })
                    else:
                        emotions.append({
                            'primary_emotion': 'neutral',
                            'emotion_confidence': 0.5
                        })
                
                df['primary_emotion'] = [e['primary_emotion'] for e in emotions]
                df['emotion_confidence'] = [e['emotion_confidence'] for e in emotions]
            
        except Exception as e:
            self.logger.warning(f"Could not analyze emotions: {e}")
        
        return df
    
    def _predict_engagement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict post engagement potential"""
        if df.empty:
            return df
        
        try:
            # Create features for engagement prediction
            features = []
            feature_names = []
            
            # Text features
            if 'title' in df.columns:
                features.extend([
                    df['title'].str.len(),
                    df['title'].str.split().str.len(),
                    df['title'].str.count(r'[!?]'),  # Emotional punctuation
                ])
                feature_names.extend(['title_length', 'title_words', 'emotional_punct'])
            
            # Timing features (if timestamp available)
            if 'created_utc' in df.columns:
                df['hour'] = pd.to_datetime(df['created_utc'], unit='s').dt.hour
                df['day_of_week'] = pd.to_datetime(df['created_utc'], unit='s').dt.dayofweek
                features.extend([df['hour'], df['day_of_week']])
                feature_names.extend(['hour', 'day_of_week'])
            
            # Sentiment features
            if 'sentiment_confidence' in df.columns:
                features.append(df['sentiment_confidence'])
                feature_names.append('sentiment_confidence')
            
            if features and 'score' in df.columns and 'num_comments' in df.columns:
                # Combine features
                X = np.column_stack(features)
                
                # Create engagement target
                y = df['score'] + df['num_comments']  # Simple engagement metric
                
                # Train regression model
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Predict engagement
                predicted_engagement = model.predict(X)
                df['predicted_engagement'] = predicted_engagement
                
                # Save model
                self.models['engagement_prediction'] = model
                
                self.logger.info("Engagement prediction completed")
                
        except Exception as e:
            self.logger.warning(f"Could not predict engagement: {e}")
        
        return df
    
    def _identify_user_personas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify user personas based on posting patterns"""
        if df.empty:
            return df
        
        try:
            # Create persona features
            persona_features = []
            
            # Desperation indicators
            if 'title' in df.columns:
                text_data = df['title'].fillna('') + ' ' + df.get('body', '').fillna('')
                
                desperation_words = CONFIG.market.desperation_keywords
                openness_words = CONFIG.market.openness_keywords
                
                df['desperation_score'] = text_data.str.lower().str.count(
                    '|'.join(desperation_words)
                )
                df['openness_score'] = text_data.str.lower().str.count(
                    '|'.join(openness_words)
                )
                
                persona_features.extend(['desperation_score', 'openness_score'])
            
            # Engagement patterns
            if 'score' in df.columns and 'num_comments' in df.columns:
                df['engagement_ratio'] = df['num_comments'] / (df['score'] + 1)
                persona_features.append('engagement_ratio')
            
            # Cluster users into personas
            if persona_features:
                cluster_data = df[persona_features].fillna(0)
                
                if len(cluster_data) > 3:
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    personas = kmeans.fit_predict(cluster_data)
                    
                    # Assign persona labels
                    persona_labels = {0: 'seeker', 1: 'skeptic', 2: 'advocate'}
                    df['user_persona'] = [persona_labels.get(p, 'unknown') for p in personas]
                    
                    self.models['persona_clustering'] = kmeans
                    
        except Exception as e:
            self.logger.warning(f"Could not identify user personas: {e}")
        
        return df
    
    def _extract_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract linguistic patterns and features"""
        if df.empty or 'title' not in df.columns:
            return df
        
        try:
            text_data = df['title'].fillna('') + ' ' + df.get('body', '').fillna('')
            
            # Linguistic features
            df['readability_score'] = text_data.apply(self._calculate_readability)
            df['formality_score'] = text_data.apply(self._calculate_formality)
            df['urgency_score'] = text_data.apply(self._calculate_urgency)
            
        except Exception as e:
            self.logger.warning(f"Could not extract linguistic features: {e}")
        
        return df
    
    def _create_synthetic_success_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create synthetic success labels for training"""
        # This is a simplified approach - in practice, you'd use actual outcome data
        success_indicators = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Higher enrollment suggests more confidence
            if pd.notna(row.get('enrollment')) and row.get('enrollment', 0) > 50:
                score += 1
            
            # Phase 3 trials are more likely to succeed
            if row.get('phase') in ['PHASE3', 'PHASE4']:
                score += 1
            
            # Trials with results published
            if row.get('results') == 'Yes' or pd.notna(row.get('results')):
                score += 1
            
            # Digital interventions might have different success patterns
            if row.get('is_digital') == True:
                score += 0.5
            
            success_indicators.append(1 if score >= 2 else 0)
        
        return np.array(success_indicators)
    
    def _textblob_sentiment(self, text: str) -> str:
        """Fallback sentiment analysis using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'POSITIVE'
            elif polarity < -0.1:
                return 'NEGATIVE'
            else:
                return 'NEUTRAL'
        except:
            return 'NEUTRAL'
    
    def _calculate_psychological_scores(self, df: pd.DataFrame, text_data: pd.Series) -> pd.DataFrame:
        """Calculate psychological indicators from text"""
        try:
            # Desperation indicators
            desperation_patterns = [
                r'\b(desperate|hopeless|last resort|nothing works|tried everything)\b',
                r'\b(can\'t take|unbearable|suffering|agony)\b',
                r'\b(running out of|no other choice|at my wit\'s end)\b'
            ]
            
            # Openness indicators
            openness_patterns = [
                r'\b(willing to try|open to|alternative|natural|holistic)\b',
                r'\b(anything helps|any suggestions|open minded)\b',
                r'\b(complementary|integrative|non-traditional)\b'
            ]
            
            # Calculate scores
            desperation_score = text_data.str.lower().str.count('|'.join(desperation_patterns))
            openness_score = text_data.str.lower().str.count('|'.join(openness_patterns))
            
            df['desperation_intensity'] = desperation_score
            df['openness_level'] = openness_score
            
        except Exception as e:
            self.logger.warning(f"Could not calculate psychological scores: {e}")
        
        return df
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score"""
        try:
            words = text.split()
            sentences = text.split('.')
            
            if len(sentences) == 0 or len(words) == 0:
                return 0.0
            
            # Simplified readability metric
            avg_words_per_sentence = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            
            # Flesch Reading Ease approximation
            score = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables_per_word)
            return max(0, min(100, score)) / 100  # Normalize to 0-1
            
        except:
            return 0.5
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        vowels = 'aeiouy'
        word = word.lower()
        count = sum(1 for char in word if char in vowels)
        
        # Adjust for common patterns
        if word.endswith('e'):
            count -= 1
        if count == 0:
            count = 1
            
        return count
    
    def _calculate_formality(self, text: str) -> float:
        """Calculate text formality level"""
        formal_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'consequently']
        informal_indicators = ['gonna', 'wanna', 'can\'t', 'don\'t', 'won\'t', '!']
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        total_indicators = formal_count + informal_count
        if total_indicators == 0:
            return 0.5
        
        return formal_count / total_indicators
    
    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency level in text"""
        urgency_indicators = ['urgent', 'emergency', 'immediate', 'asap', 'help!', '!!!', 'now']
        
        text_lower = text.lower()
        urgency_count = sum(text_lower.count(indicator) for indicator in urgency_indicators)
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0.0
        
        return min(urgency_count / words * 10, 1.0)  # Cap at 1.0
    
    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk"""
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f"{name}.pkl"))
        
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(model_dir, f"{name}_scaler.pkl"))
        
        for name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, os.path.join(model_dir, f"{name}_vectorizer.pkl"))
        
        self.logger.info(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk"""
        if not os.path.exists(model_dir):
            self.logger.warning(f"Model directory {model_dir} does not exist")
            return
        
        # Load models, scalers, and vectorizers
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(model_dir, filename)
                name = filename.replace('.pkl', '')
                
                if 'scaler' in name:
                    self.scalers[name.replace('_scaler', '')] = joblib.load(filepath)
                elif 'vectorizer' in name:
                    self.vectorizers[name.replace('_vectorizer', '')] = joblib.load(filepath)
                else:
                    self.models[name] = joblib.load(filepath)
        
        self.logger.info(f"Models loaded from {model_dir}")
    
    def generate_ml_insights_report(self, clinical_df: pd.DataFrame, 
                                   market_df: pd.DataFrame) -> str:
        """Generate comprehensive ML insights report"""
        report = []
        report.append("# Machine Learning Enhancement Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Clinical ML insights
        if not clinical_df.empty:
            report.append("## Clinical Trials ML Analysis")
            
            if 'predicted_success_probability' in clinical_df.columns:
                avg_success_prob = clinical_df['predicted_success_probability'].mean()
                high_success_trials = (clinical_df['predicted_success_probability'] > 0.7).sum()
                report.append(f"**Average Success Probability**: {avg_success_prob:.2f}")
                report.append(f"**High Success Probability Trials**: {high_success_trials}")
            
            if 'trial_cluster' in clinical_df.columns:
                n_clusters = clinical_df['trial_cluster'].nunique()
                report.append(f"**Trial Clusters Identified**: {n_clusters}")
            
            report.append("")
        
        # Market ML insights
        if not market_df.empty:
            report.append("## Market Analysis ML Insights")
            
            if 'advanced_sentiment' in market_df.columns:
                sentiment_dist = market_df['advanced_sentiment'].value_counts()
                report.append("**Sentiment Distribution**:")
                for sentiment, count in sentiment_dist.items():
                    report.append(f"- {sentiment}: {count} ({count/len(market_df)*100:.1f}%)")
            
            if 'primary_emotion' in market_df.columns:
                emotion_dist = market_df['primary_emotion'].value_counts().head(3)
                report.append("**Top Emotions**:")
                for emotion, count in emotion_dist.items():
                    report.append(f"- {emotion}: {count}")
            
            if 'user_persona' in market_df.columns:
                persona_dist = market_df['user_persona'].value_counts()
                report.append("**User Personas**:")
                for persona, count in persona_dist.items():
                    report.append(f"- {persona}: {count}")
            
            if 'desperation_intensity' in market_df.columns:
                avg_desperation = market_df['desperation_intensity'].mean()
                high_desperation = (market_df['desperation_intensity'] > 2).sum()
                report.append(f"**Average Desperation Level**: {avg_desperation:.2f}")
                report.append(f"**High Desperation Posts**: {high_desperation}")
        
        # Model performance
        report.append("")
        report.append("## Model Performance")
        report.append(f"**Models Trained**: {len(self.models)}")
        for model_name in self.models.keys():
            report.append(f"- {model_name}")
        
        return '\n'.join(report)