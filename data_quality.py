import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from enhanced_config import CONFIG, ValidationLevel

@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    cleaned_data: Optional[pd.DataFrame] = None
    quality_score: float = 0.0
    
class DataQualityValidator:
    """Comprehensive data quality validation and cleaning"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
    def validate_clinical_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate clinical trials data"""
        issues = []
        warnings = []
        cleaned_df = df.copy()
        
        # Check required fields
        required_fields = CONFIG.quality.required_clinical_fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Validate NCT IDs
        if 'nct_id' in df.columns:
            invalid_nct = df[~df['nct_id'].str.match(r'^NCT\d{8}$', na=False)]
            if not invalid_nct.empty:
                issues.append(f"{len(invalid_nct)} invalid NCT IDs found")
                if self.validation_level == ValidationLevel.STRICT:
                    cleaned_df = cleaned_df[cleaned_df['nct_id'].str.match(r'^NCT\d{8}$', na=False)]
        
        # Validate enrollment numbers
        if 'enrollment' in df.columns:
            cleaned_df['enrollment'] = pd.to_numeric(cleaned_df['enrollment'], errors='coerce')
            low_enrollment = cleaned_df[cleaned_df['enrollment'] < CONFIG.clinical.min_enrollment]
            if not low_enrollment.empty:
                warnings.append(f"{len(low_enrollment)} trials below minimum enrollment threshold")
                if self.validation_level != ValidationLevel.BASIC:
                    cleaned_df = cleaned_df[cleaned_df['enrollment'] >= CONFIG.clinical.min_enrollment]
        
        # Validate completion dates
        if 'completion_date' in df.columns:
            cleaned_df = self._validate_dates(cleaned_df, 'completion_date')
        
        # Check for duplicates
        if 'nct_id' in cleaned_df.columns:
            duplicates = cleaned_df.duplicated(subset=['nct_id'])
            if duplicates.sum() > 0:
                warnings.append(f"{duplicates.sum()} duplicate NCT IDs found")
                cleaned_df = cleaned_df.drop_duplicates(subset=['nct_id'])
        
        # Validate text fields
        text_fields = ['title', 'condition', 'intervention']
        for field in text_fields:
            if field in cleaned_df.columns:
                cleaned_df = self._clean_text_field(cleaned_df, field)
        
        # Calculate quality score
        quality_score = self._calculate_clinical_quality_score(cleaned_df)
        
        is_valid = len(issues) == 0 and quality_score > 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            cleaned_data=cleaned_df,
            quality_score=quality_score
        )
    
    def validate_market_data(self, df: pd.DataFrame) -> ValidationResult:
        """Validate market analysis (Reddit) data"""
        issues = []
        warnings = []
        cleaned_df = df.copy()
        
        # Check required fields
        required_fields = CONFIG.quality.required_market_fields
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
        
        # Remove deleted/removed posts
        if CONFIG.quality.remove_deleted_posts and 'title' in df.columns:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df[
                (~cleaned_df['title'].isin(['[deleted]', '[removed]'])) &
                (~cleaned_df.get('body', '').isin(['[deleted]', '[removed]']))
            ]
            removed_count = before_count - len(cleaned_df)
            if removed_count > 0:
                warnings.append(f"Removed {removed_count} deleted/removed posts")
        
        # Validate text length
        if 'title' in cleaned_df.columns:
            cleaned_df['text_length'] = cleaned_df['title'].str.len() + cleaned_df.get('body', '').str.len()
            
            # Filter by text length
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df[
                (cleaned_df['text_length'] >= CONFIG.quality.min_text_length) &
                (cleaned_df['text_length'] <= CONFIG.quality.max_text_length)
            ]
            filtered_count = before_count - len(cleaned_df)
            if filtered_count > 0:
                warnings.append(f"Filtered {filtered_count} posts due to text length constraints")
        
        # Validate scores and engagement metrics
        numeric_fields = ['score', 'num_comments', 'upvote_ratio']
        for field in numeric_fields:
            if field in cleaned_df.columns:
                cleaned_df[field] = pd.to_numeric(cleaned_df[field], errors='coerce')
                
        # Remove posts with minimal engagement
        if 'score' in cleaned_df.columns and 'num_comments' in cleaned_df.columns:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df[
                (cleaned_df['score'] >= CONFIG.market.min_post_score) &
                (cleaned_df['num_comments'] >= CONFIG.market.min_comment_count)
            ]
            filtered_count = before_count - len(cleaned_df)
            if filtered_count > 0:
                warnings.append(f"Filtered {filtered_count} low-engagement posts")
        
        # Validate timestamps
        if 'created_utc' in cleaned_df.columns:
            cleaned_df = self._validate_timestamps(cleaned_df, 'created_utc')
        
        # Detect and remove duplicates
        cleaned_df = self._remove_duplicate_posts(cleaned_df)
        
        # Calculate quality score
        quality_score = self._calculate_market_quality_score(cleaned_df)
        
        is_valid = len(issues) == 0 and quality_score > 0.3
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            cleaned_data=cleaned_df,
            quality_score=quality_score
        )
    
    def _validate_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Validate and standardize date fields"""
        if date_column not in df.columns:
            return df
        
        # Try to parse dates
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Remove future dates (likely errors)
        future_mask = df[date_column] > datetime.now()
        if future_mask.sum() > 0:
            self.logger.warning(f"Removed {future_mask.sum()} entries with future dates")
            df = df[~future_mask]
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
        """Validate and filter timestamps"""
        if timestamp_column not in df.columns:
            return df
        
        # Convert to datetime
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit='s', errors='coerce')
        
        # Filter to recent posts within time window
        cutoff_date = datetime.now() - timedelta(days=CONFIG.market.time_window_days)
        before_count = len(df)
        df = df[df[timestamp_column] >= cutoff_date]
        filtered_count = before_count - len(df)
        
        if filtered_count > 0:
            self.logger.info(f"Filtered {filtered_count} posts outside time window")
        
        return df
    
    def _clean_text_field(self, df: pd.DataFrame, field: str) -> pd.DataFrame:
        """Clean and standardize text fields"""
        if field not in df.columns:
            return df
        
        # Remove null/empty values
        df = df[df[field].notna() & (df[field] != '')]
        
        # Basic text cleaning
        df[field] = df[field].astype(str)
        df[field] = df[field].str.strip()
        
        # Remove entries that are too short or clearly invalid
        df = df[df[field].str.len() >= 10]  # Minimum meaningful length
        
        return df
    
    def _remove_duplicate_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate posts using similarity detection"""
        if len(df) < 2 or 'title' not in df.columns:
            return df
        
        # Simple exact duplicate removal first
        before_count = len(df)
        df = df.drop_duplicates(subset=['title'])
        exact_dupes = before_count - len(df)
        
        if exact_dupes > 0:
            self.logger.info(f"Removed {exact_dupes} exact duplicate posts")
        
        # Advanced similarity-based duplicate detection
        if CONFIG.quality.check_cross_platform_duplicates and len(df) > 1:
            df = self._remove_similar_duplicates(df)
        
        return df
    
    def _remove_similar_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove similar posts using TF-IDF similarity"""
        try:
            # Combine title and body for similarity analysis
            text_content = df['title'] + ' ' + df.get('body', '')
            
            # Use TF-IDF to find similar posts
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(text_content)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find duplicates above threshold
            duplicate_indices = set()
            threshold = CONFIG.quality.similarity_threshold
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > threshold:
                        # Keep the post with higher engagement
                        if df.iloc[i].get('score', 0) >= df.iloc[j].get('score', 0):
                            duplicate_indices.add(j)
                        else:
                            duplicate_indices.add(i)
            
            if duplicate_indices:
                df = df.drop(df.index[list(duplicate_indices)])
                self.logger.info(f"Removed {len(duplicate_indices)} similar duplicate posts")
                
        except Exception as e:
            self.logger.warning(f"Could not perform similarity-based duplicate detection: {e}")
        
        return df
    
    def _calculate_clinical_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall quality score for clinical data"""
        if df.empty:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Completeness score (30%)
        required_fields = CONFIG.quality.required_clinical_fields
        present_fields = [field for field in required_fields if field in df.columns]
        completeness = len(present_fields) / len(required_fields)
        score += completeness * 0.3
        max_score += 0.3
        
        # Data validity score (25%)
        if 'enrollment' in df.columns:
            valid_enrollment = df['enrollment'].notna().sum() / len(df)
            score += valid_enrollment * 0.25
        max_score += 0.25
        
        # Recency score (20%)
        if 'completion_date' in df.columns:
            recent_trials = df[df['completion_date'] > datetime.now() - timedelta(days=365*5)]
            recency = len(recent_trials) / len(df) if len(df) > 0 else 0
            score += recency * 0.2
        max_score += 0.2
        
        # Diversity score (25%)
        if 'condition' in df.columns:
            unique_conditions = df['condition'].nunique()
            diversity = min(unique_conditions / 10, 1.0)  # Normalize to max 10 conditions
            score += diversity * 0.25
        max_score += 0.25
        
        return score / max_score if max_score > 0 else 0.0
    
    def _calculate_market_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall quality score for market data"""
        if df.empty:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Engagement quality (40%)
        if 'score' in df.columns and 'num_comments' in df.columns:
            avg_score = df['score'].mean()
            avg_comments = df['num_comments'].mean()
            engagement = min((avg_score + avg_comments) / 20, 1.0)  # Normalize
            score += engagement * 0.4
        max_score += 0.4
        
        # Content quality (30%)
        if 'title' in df.columns:
            avg_length = df['title'].str.len().mean()
            length_quality = min(avg_length / 100, 1.0)  # Normalize to 100 chars
            score += length_quality * 0.3
        max_score += 0.3
        
        # Diversity (30%)
        if 'subreddit' in df.columns:
            unique_subreddits = df['subreddit'].nunique()
            diversity = min(unique_subreddits / len(CONFIG.market.subreddits), 1.0)
            score += diversity * 0.3
        max_score += 0.3
        
        return score / max_score if max_score > 0 else 0.0
    
    def generate_quality_report(self, clinical_result: ValidationResult, 
                               market_result: ValidationResult) -> str:
        """Generate comprehensive data quality report"""
        report = []
        report.append("# Data Quality Assessment Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Validation Level: {self.validation_level.value}")
        report.append("")
        
        # Clinical data quality
        report.append("## Clinical Trials Data Quality")
        report.append(f"**Overall Quality Score**: {clinical_result.quality_score:.2f}/1.00")
        report.append(f"**Validation Status**: {'✅ PASSED' if clinical_result.is_valid else '❌ FAILED'}")
        
        if clinical_result.cleaned_data is not None:
            report.append(f"**Records Processed**: {len(clinical_result.cleaned_data)}")
        
        if clinical_result.issues:
            report.append("**Issues Found**:")
            for issue in clinical_result.issues:
                report.append(f"- ❌ {issue}")
        
        if clinical_result.warnings:
            report.append("**Warnings**:")
            for warning in clinical_result.warnings:
                report.append(f"- ⚠️ {warning}")
        
        report.append("")
        
        # Market data quality
        report.append("## Market Analysis Data Quality")
        report.append(f"**Overall Quality Score**: {market_result.quality_score:.2f}/1.00")
        report.append(f"**Validation Status**: {'✅ PASSED' if market_result.is_valid else '❌ FAILED'}")
        
        if market_result.cleaned_data is not None:
            report.append(f"**Records Processed**: {len(market_result.cleaned_data)}")
        
        if market_result.issues:
            report.append("**Issues Found**:")
            for issue in market_result.issues:
                report.append(f"- ❌ {issue}")
        
        if market_result.warnings:
            report.append("**Warnings**:")
            for warning in market_result.warnings:
                report.append(f"- ⚠️ {warning}")
        
        # Overall assessment
        report.append("")
        report.append("## Overall Data Quality Assessment")
        
        overall_score = (clinical_result.quality_score + market_result.quality_score) / 2
        report.append(f"**Combined Quality Score**: {overall_score:.2f}/1.00")
        
        if overall_score >= 0.8:
            report.append("✅ **EXCELLENT**: Data quality is excellent for analysis")
        elif overall_score >= 0.6:
            report.append("✅ **GOOD**: Data quality is good with minor issues")
        elif overall_score >= 0.4:
            report.append("⚠️ **FAIR**: Data quality is fair, consider additional cleaning")
        else:
            report.append("❌ **POOR**: Data quality is poor, significant issues detected")
        
        return '\n'.join(report)