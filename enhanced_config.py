import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class AnalysisMode(Enum):
    QUICK = "quick"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class ValidationLevel(Enum):
    BASIC = "basic"
    STRICT = "strict"
    RESEARCH_GRADE = "research_grade"

@dataclass
class ClinicalTrialsConfig:
    """Enhanced clinical trials analysis configuration"""
    search_terms: List[str] = field(default_factory=lambda: [
        'open-label placebo', 'open label placebo', 'digital placebo',
        'app placebo', 'online placebo', 'digital therapeutic',
        'mobile health', 'mhealth', 'telemedicine', 'digital intervention'
    ])
    
    # Enhanced search parameters
    target_conditions: List[str] = field(default_factory=lambda: [
        'chronic pain', 'anxiety', 'depression', 'fibromyalgia',
        'irritable bowel syndrome', 'migraine', 'insomnia',
        'PTSD', 'chronic fatigue', 'arthritis'
    ])
    
    # Quality filters
    min_enrollment: int = 10
    required_phases: List[str] = field(default_factory=lambda: ['PHASE1', 'PHASE2', 'PHASE3', 'PHASE4'])
    require_published_results: bool = False
    min_follow_up_months: int = 1
    
    # Statistical significance thresholds
    significance_p_value: float = 0.05
    clinical_relevance_threshold: float = 0.2  # 20% improvement
    effect_size_threshold: float = 0.3  # Cohen's d
    
    # API rate limiting
    requests_per_minute: int = 30
    max_trials_per_search: int = 1000

@dataclass
class MarketAnalysisConfig:
    """Enhanced market analysis configuration"""
    subreddits: List[str] = field(default_factory=lambda: [
        'chronicpain', 'CFS', 'fibromyalgia', 'anxiety', 'depression',
        'mentalhealth', 'wellness', 'meditation', 'mindfulness',
        'IBS', 'migraine', 'insomnia', 'PTSD', 'BipolarReddit'
    ])
    
    # Enhanced keyword categories
    desperation_keywords: List[str] = field(default_factory=lambda: [
        'nothing works', 'tried everything', 'desperate', 'last resort',
        'running out of options', 'at my wit\'s end', 'no hope left'
    ])
    
    openness_keywords: List[str] = field(default_factory=lambda: [
        'alternative', 'natural', 'holistic', 'non-pharma', 'complementary',
        'integrative', 'mind-body', 'willing to try anything'
    ])
    
    efficacy_keywords: List[str] = field(default_factory=lambda: [
        'placebo', 'psychological', 'mindset', 'belief', 'expectation',
        'psychosomatic', 'mind over matter', 'power of positive thinking'
    ])
    
    # Sampling parameters
    posts_per_subreddit: int = 100
    time_window_days: int = 30
    min_post_score: int = 1
    min_comment_count: int = 2
    
    # Sentiment analysis
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"

@dataclass
class DataQualityConfig:
    """Data quality and validation controls"""
    validation_level: ValidationLevel = ValidationLevel.STRICT
    
    # Clinical data validation
    required_clinical_fields: List[str] = field(default_factory=lambda: [
        'nct_id', 'title', 'condition', 'intervention', 'enrollment', 'status'
    ])
    
    # Market data validation
    required_market_fields: List[str] = field(default_factory=lambda: [
        'subreddit', 'title', 'score', 'num_comments', 'created_utc'
    ])
    
    # Data cleaning parameters
    remove_deleted_posts: bool = True
    remove_removed_posts: bool = True
    min_text_length: int = 50
    max_text_length: int = 10000
    
    # Duplicate detection
    similarity_threshold: float = 0.85
    check_cross_platform_duplicates: bool = True

@dataclass
class OutputConfig:
    """Output format and reporting configuration"""
    output_dir: str = "outputs"
    include_raw_data: bool = True
    include_visualizations: bool = True
    
    # Report formats
    generate_pdf: bool = True
    generate_html: bool = True
    generate_json: bool = True
    
    # Visualization settings
    plot_style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    figure_dpi: int = 300
    
    # Advanced analytics
    include_statistical_tests: bool = True
    include_ml_predictions: bool = True
    include_trend_analysis: bool = True

@dataclass
class EnhancedConfig:
    """Master configuration class"""
    # Analysis mode
    mode: AnalysisMode = AnalysisMode.COMPREHENSIVE
    
    # Sub-configurations
    clinical: ClinicalTrialsConfig = field(default_factory=ClinicalTrialsConfig)
    market: MarketAnalysisConfig = field(default_factory=MarketAnalysisConfig)
    quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # API credentials
    reddit_client_id: str = field(default_factory=lambda: os.getenv('REDDIT_CLIENT_ID', ''))
    reddit_client_secret: str = field(default_factory=lambda: os.getenv('REDDIT_CLIENT_SECRET', ''))
    reddit_user_agent: str = field(default_factory=lambda: os.getenv('REDDIT_USER_AGENT', 'PlaceboRx_Enhanced/2.0'))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    
    # Advanced features
    enable_ml_enhancement: bool = True
    enable_real_time_monitoring: bool = False
    enable_cross_validation: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.reddit_client_id:
            issues.append("Missing REDDIT_CLIENT_ID")
        if not self.reddit_client_secret:
            issues.append("Missing REDDIT_CLIENT_SECRET")
            
        if self.clinical.min_enrollment < 1:
            issues.append("Minimum enrollment must be >= 1")
            
        if not self.clinical.search_terms:
            issues.append("Clinical search terms cannot be empty")
            
        if not self.market.subreddits:
            issues.append("Target subreddits cannot be empty")
            
        return issues
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'EnhancedConfig':
        """Load configuration from JSON file"""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        import json
        from dataclasses import asdict
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

# Global configuration instance
CONFIG = EnhancedConfig()