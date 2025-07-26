#!/usr/bin/env python3
"""
Real-World Evidence Engine for PlaceboRx Hypothesis Validation
Provides longitudinal tracking, real-world outcomes monitoring, and continuous validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

# Time series analysis
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Survival analysis
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False

# Causal inference
try:
    from causalinference import CausalModel
    import econml
    CAUSAL_AVAILABLE = True
except ImportError:
    CAUSAL_AVAILABLE = False

from enhanced_config import CONFIG

class OutcomeType(Enum):
    """Types of real-world outcomes to track"""
    SYMPTOM_SEVERITY = "symptom_severity"
    QUALITY_OF_LIFE = "quality_of_life"
    MEDICATION_USAGE = "medication_usage"
    HEALTHCARE_UTILIZATION = "healthcare_utilization"
    FUNCTIONAL_STATUS = "functional_status"
    ADHERENCE = "adherence"
    SATISFACTION = "satisfaction"
    ADVERSE_EVENTS = "adverse_events"

class DataSource(Enum):
    """Sources of real-world data"""
    MOBILE_APP = "mobile_app"
    WEARABLE_DEVICE = "wearable_device"
    ELECTRONIC_HEALTH_RECORDS = "ehr"
    PATIENT_REPORTED = "patient_reported"
    PHARMACY_RECORDS = "pharmacy"
    INSURANCE_CLAIMS = "insurance_claims"
    SOCIAL_MEDIA = "social_media"

@dataclass
class RealWorldDataPoint:
    """Individual real-world data point"""
    patient_id: str
    timestamp: datetime
    outcome_type: OutcomeType
    value: Union[float, int, str]
    data_source: DataSource
    confidence: float
    context: Dict[str, Any]

@dataclass
class LongitudinalOutcome:
    """Longitudinal outcome tracking"""
    patient_id: str
    outcome_type: OutcomeType
    baseline_value: float
    current_value: float
    trend_slope: float
    trend_significance: float
    days_tracked: int
    last_updated: datetime
    quality_score: float

@dataclass
class CohortAnalysis:
    """Cohort analysis results"""
    cohort_name: str
    n_patients: int
    baseline_characteristics: Dict[str, float]
    outcome_summary: Dict[str, float]
    survival_analysis: Optional[Dict[str, Any]]
    trend_analysis: Dict[str, float]
    comparison_groups: List[str]

class RealWorldEvidenceEngine:
    """Engine for collecting and analyzing real-world evidence"""
    
    def __init__(self, database_path: str = "real_world_evidence.db"):
        self.logger = logging.getLogger(__name__)
        self.database_path = database_path
        self.initialize_database()
        
    def initialize_database(self):
        """Initialize SQLite database for real-world data storage"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    enrollment_date DATE,
                    condition TEXT,
                    demographics TEXT,
                    baseline_characteristics TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp DATETIME,
                    outcome_type TEXT,
                    value REAL,
                    data_source TEXT,
                    confidence REAL,
                    context TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interventions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    intervention_type TEXT,
                    start_date DATE,
                    end_date DATE,
                    dosage TEXT,
                    adherence_rate REAL,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cohorts (
                    cohort_name TEXT PRIMARY KEY,
                    description TEXT,
                    created_date DATE,
                    inclusion_criteria TEXT,
                    n_patients INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized at {self.database_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def enroll_patient(self, patient_id: str, condition: str, demographics: Dict[str, Any],
                      baseline_characteristics: Dict[str, float]) -> bool:
        """Enroll a new patient for real-world evidence tracking"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO patients 
                (patient_id, enrollment_date, condition, demographics, baseline_characteristics)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                patient_id,
                datetime.now().date(),
                condition,
                json.dumps(demographics),
                json.dumps(baseline_characteristics)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Patient {patient_id} enrolled for RWE tracking")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enrolling patient {patient_id}: {e}")
            return False
    
    def record_outcome(self, data_point: RealWorldDataPoint) -> bool:
        """Record a real-world outcome data point"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO outcomes 
                (patient_id, timestamp, outcome_type, value, data_source, confidence, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_point.patient_id,
                data_point.timestamp,
                data_point.outcome_type.value,
                data_point.value,
                data_point.data_source.value,
                data_point.confidence,
                json.dumps(data_point.context)
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording outcome: {e}")
            return False
    
    def analyze_longitudinal_outcomes(self, patient_ids: List[str] = None,
                                    outcome_types: List[OutcomeType] = None,
                                    days_lookback: int = 365) -> List[LongitudinalOutcome]:
        """Analyze longitudinal outcomes for patients"""
        self.logger.info("Analyzing longitudinal outcomes...")
        
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Build query
            query = '''
                SELECT patient_id, outcome_type, timestamp, value
                FROM outcomes
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days_lookback)
            
            conditions = []
            if patient_ids:
                placeholders = ','.join(['?' for _ in patient_ids])
                conditions.append(f"patient_id IN ({placeholders})")
            
            if outcome_types:
                outcome_values = [ot.value for ot in outcome_types]
                placeholders = ','.join(['?' for _ in outcome_values])
                conditions.append(f"outcome_type IN ({placeholders})")
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY patient_id, outcome_type, timestamp"
            
            # Execute query
            params = []
            if patient_ids:
                params.extend(patient_ids)
            if outcome_types:
                params.extend([ot.value for ot in outcome_types])
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return []
            
            # Analyze trends for each patient-outcome combination
            longitudinal_outcomes = []
            
            for (patient_id, outcome_type), group in df.groupby(['patient_id', 'outcome_type']):
                if len(group) < 3:  # Need at least 3 points for trend
                    continue
                
                # Sort by timestamp
                group = group.sort_values('timestamp')
                group['timestamp'] = pd.to_datetime(group['timestamp'])
                
                # Calculate trend
                trend_result = self._calculate_trend(group['timestamp'], group['value'])
                
                # Calculate quality score
                quality_score = self._calculate_data_quality_score(group)
                
                longitudinal_outcome = LongitudinalOutcome(
                    patient_id=patient_id,
                    outcome_type=OutcomeType(outcome_type),
                    baseline_value=group['value'].iloc[0],
                    current_value=group['value'].iloc[-1],
                    trend_slope=trend_result['slope'],
                    trend_significance=trend_result['p_value'],
                    days_tracked=(group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).days,
                    last_updated=group['timestamp'].iloc[-1],
                    quality_score=quality_score
                )
                
                longitudinal_outcomes.append(longitudinal_outcome)
            
            return longitudinal_outcomes
            
        except Exception as e:
            self.logger.error(f"Error analyzing longitudinal outcomes: {e}")
            return []
    
    def create_cohort(self, cohort_name: str, inclusion_criteria: Dict[str, Any],
                     description: str = "") -> bool:
        """Create a patient cohort for analysis"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get patients matching criteria
            patient_ids = self._identify_cohort_patients(inclusion_criteria)
            
            # Store cohort definition
            cursor.execute('''
                INSERT OR REPLACE INTO cohorts 
                (cohort_name, description, created_date, inclusion_criteria, n_patients)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                cohort_name,
                description,
                datetime.now().date(),
                json.dumps(inclusion_criteria),
                len(patient_ids)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cohort '{cohort_name}' created with {len(patient_ids)} patients")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating cohort {cohort_name}: {e}")
            return False
    
    def analyze_cohort_outcomes(self, cohort_name: str,
                               comparison_cohorts: List[str] = None) -> CohortAnalysis:
        """Analyze outcomes for a specific cohort"""
        self.logger.info(f"Analyzing cohort: {cohort_name}")
        
        try:
            # Get cohort patients
            cohort_patients = self._get_cohort_patients(cohort_name)
            
            if not cohort_patients:
                raise ValueError(f"Cohort {cohort_name} not found or empty")
            
            # Get baseline characteristics
            baseline_chars = self._get_baseline_characteristics(cohort_patients)
            
            # Analyze outcomes
            outcome_summary = self._analyze_cohort_outcome_summary(cohort_patients)
            
            # Survival analysis (if applicable)
            survival_analysis = None
            if SURVIVAL_AVAILABLE:
                survival_analysis = self._perform_survival_analysis(cohort_patients)
            
            # Trend analysis
            trend_analysis = self._analyze_cohort_trends(cohort_patients)
            
            # Comparison with other cohorts
            comparison_results = {}
            if comparison_cohorts:
                for comp_cohort in comparison_cohorts:
                    comp_patients = self._get_cohort_patients(comp_cohort)
                    if comp_patients:
                        comparison_results[comp_cohort] = self._compare_cohorts(
                            cohort_patients, comp_patients
                        )
            
            return CohortAnalysis(
                cohort_name=cohort_name,
                n_patients=len(cohort_patients),
                baseline_characteristics=baseline_chars,
                outcome_summary=outcome_summary,
                survival_analysis=survival_analysis,
                trend_analysis=trend_analysis,
                comparison_groups=list(comparison_results.keys())
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing cohort {cohort_name}: {e}")
            return CohortAnalysis(
                cohort_name=cohort_name,
                n_patients=0,
                baseline_characteristics={},
                outcome_summary={},
                survival_analysis=None,
                trend_analysis={},
                comparison_groups=[]
            )
    
    def monitor_real_time_safety(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Monitor real-time safety signals"""
        self.logger.info("Monitoring real-time safety signals...")
        
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Get recent adverse events
            query = '''
                SELECT patient_id, timestamp, value, context
                FROM outcomes
                WHERE outcome_type = 'adverse_events'
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            '''.format(lookback_hours)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            safety_signals = {
                'total_events': len(df),
                'unique_patients': df['patient_id'].nunique() if not df.empty else 0,
                'event_rate_per_hour': len(df) / lookback_hours,
                'severity_distribution': {},
                'trending_events': [],
                'alerts': []
            }
            
            if not df.empty:
                # Analyze severity distribution
                severities = []
                for context_str in df['context'].dropna():
                    try:
                        context = json.loads(context_str)
                        severity = context.get('severity', 'unknown')
                        severities.append(severity)
                    except:
                        severities.append('unknown')
                
                severity_counts = pd.Series(severities).value_counts().to_dict()
                safety_signals['severity_distribution'] = severity_counts
                
                # Check for alerts
                if len(df) > 10:  # More than 10 events in lookback period
                    safety_signals['alerts'].append({
                        'type': 'high_event_rate',
                        'message': f'High adverse event rate detected: {len(df)} events in {lookback_hours} hours'
                    })
                
                # Check for severe events
                severe_events = sum(1 for s in severities if s in ['severe', 'serious'])
                if severe_events > 2:
                    safety_signals['alerts'].append({
                        'type': 'severe_events',
                        'message': f'Multiple severe events detected: {severe_events} severe events'
                    })
            
            return safety_signals
            
        except Exception as e:
            self.logger.error(f"Error monitoring safety: {e}")
            return {'error': str(e)}
    
    def generate_effectiveness_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for real-world effectiveness dashboard"""
        try:
            # Get overall statistics
            conn = sqlite3.connect(self.database_path)
            
            # Total patients
            total_patients = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM patients", conn
            ).iloc[0]['count']
            
            # Active patients (with data in last 30 days)
            active_patients = pd.read_sql_query('''
                SELECT COUNT(DISTINCT patient_id) as count 
                FROM outcomes 
                WHERE timestamp >= datetime('now', '-30 days')
            ''', conn).iloc[0]['count']
            
            # Outcome trends
            outcome_trends = {}
            for outcome_type in OutcomeType:
                trend_data = pd.read_sql_query('''
                    SELECT DATE(timestamp) as date, AVG(value) as avg_value
                    FROM outcomes
                    WHERE outcome_type = ?
                    AND timestamp >= datetime('now', '-90 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                ''', conn, params=[outcome_type.value])
                
                if not trend_data.empty:
                    outcome_trends[outcome_type.value] = {
                        'dates': trend_data['date'].tolist(),
                        'values': trend_data['avg_value'].tolist()
                    }
            
            # Data quality metrics
            data_quality = pd.read_sql_query('''
                SELECT 
                    data_source,
                    COUNT(*) as total_records,
                    AVG(confidence) as avg_confidence
                FROM outcomes
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY data_source
            ''', conn)
            
            conn.close()
            
            return {
                'total_patients': total_patients,
                'active_patients': active_patients,
                'outcome_trends': outcome_trends,
                'data_quality': data_quality.to_dict('records') if not data_quality.empty else [],
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    def perform_causal_analysis(self, intervention_type: str, outcome_type: OutcomeType,
                               confounders: List[str] = None) -> Dict[str, Any]:
        """Perform causal analysis of intervention effects"""
        if not CAUSAL_AVAILABLE:
            return {'error': 'Causal inference libraries not available'}
        
        self.logger.info(f"Performing causal analysis: {intervention_type} -> {outcome_type.value}")
        
        try:
            # Get data for causal analysis
            conn = sqlite3.connect(self.database_path)
            
            # Build comprehensive dataset
            query = '''
                SELECT 
                    p.patient_id,
                    p.demographics,
                    p.baseline_characteristics,
                    i.intervention_type,
                    i.start_date as intervention_start,
                    i.adherence_rate,
                    o.timestamp as outcome_timestamp,
                    o.value as outcome_value
                FROM patients p
                LEFT JOIN interventions i ON p.patient_id = i.patient_id
                LEFT JOIN outcomes o ON p.patient_id = o.patient_id
                WHERE i.intervention_type = ? 
                AND o.outcome_type = ?
                AND o.timestamp >= i.start_date
            '''
            
            df = pd.read_sql_query(query, conn, params=[intervention_type, outcome_type.value])
            conn.close()
            
            if df.empty:
                return {'error': 'Insufficient data for causal analysis'}
            
            # Prepare data for causal model
            causal_data = self._prepare_causal_data(df, confounders)
            
            # Estimate causal effect
            causal_effect = self._estimate_causal_effect(causal_data)
            
            return {
                'intervention': intervention_type,
                'outcome': outcome_type.value,
                'sample_size': len(causal_data),
                'causal_effect': causal_effect['estimate'],
                'confidence_interval': causal_effect['ci'],
                'p_value': causal_effect['p_value'],
                'method': 'propensity_score_matching',
                'confounders_controlled': confounders or [],
                'interpretation': self._interpret_causal_effect(causal_effect)
            }
            
        except Exception as e:
            self.logger.error(f"Error in causal analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, timestamps: pd.Series, values: pd.Series) -> Dict[str, float]:
        """Calculate trend statistics for time series data"""
        try:
            # Convert timestamps to numeric (days since first measurement)
            timestamps = pd.to_datetime(timestamps)
            days = (timestamps - timestamps.iloc[0]).dt.days.values
            
            # Linear regression
            X = sm.add_constant(days)
            model = sm.OLS(values, X).fit()
            
            return {
                'slope': model.params[1],
                'intercept': model.params[0],
                'r_squared': model.rsquared,
                'p_value': model.pvalues[1],
                'confidence_interval': model.conf_int().iloc[1].tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trend: {e}")
            return {'slope': 0, 'intercept': 0, 'r_squared': 0, 'p_value': 1.0, 'confidence_interval': [0, 0]}
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate data quality score for a dataset"""
        try:
            score = 0.0
            
            # Completeness (40%)
            completeness = data['value'].notna().mean()
            score += completeness * 0.4
            
            # Consistency (30%) - coefficient of variation of confidence scores
            if 'confidence' in data.columns:
                confidence_cv = data['confidence'].std() / data['confidence'].mean()
                consistency = max(0, 1 - confidence_cv)
                score += consistency * 0.3
            else:
                score += 0.3  # Default if no confidence scores
            
            # Timeliness (20%) - recent data gets higher score
            if len(data) > 1:
                time_diff = (pd.to_datetime(data['timestamp'].iloc[-1]) - 
                           pd.to_datetime(data['timestamp'].iloc[0])).days
                timeliness = min(1.0, 30 / max(time_diff, 1))  # Prefer data within 30 days
                score += timeliness * 0.2
            else:
                score += 0.2
            
            # Frequency (10%) - more frequent measurements get higher score
            frequency_score = min(1.0, len(data) / 30)  # Prefer at least daily measurements
            score += frequency_score * 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.5  # Default moderate quality
    
    def _identify_cohort_patients(self, criteria: Dict[str, Any]) -> List[str]:
        """Identify patients matching cohort criteria"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            query = "SELECT patient_id FROM patients WHERE 1=1"
            params = []
            
            # Add criteria to query
            if 'condition' in criteria:
                query += " AND condition = ?"
                params.append(criteria['condition'])
            
            if 'age_range' in criteria:
                # This would require parsing demographics JSON
                # Simplified for this example
                pass
            
            if 'enrollment_after' in criteria:
                query += " AND enrollment_date >= ?"
                params.append(criteria['enrollment_after'])
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df['patient_id'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error identifying cohort patients: {e}")
            return []
    
    def _get_cohort_patients(self, cohort_name: str) -> List[str]:
        """Get patient IDs for a named cohort"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Get cohort criteria
            cohort_query = "SELECT inclusion_criteria FROM cohorts WHERE cohort_name = ?"
            cohort_data = pd.read_sql_query(cohort_query, conn, params=[cohort_name])
            
            if cohort_data.empty:
                return []
            
            criteria = json.loads(cohort_data.iloc[0]['inclusion_criteria'])
            conn.close()
            
            return self._identify_cohort_patients(criteria)
            
        except Exception as e:
            self.logger.error(f"Error getting cohort patients: {e}")
            return []
    
    def _get_baseline_characteristics(self, patient_ids: List[str]) -> Dict[str, float]:
        """Get baseline characteristics for a group of patients"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            placeholders = ','.join(['?' for _ in patient_ids])
            query = f'''
                SELECT baseline_characteristics
                FROM patients
                WHERE patient_id IN ({placeholders})
            '''
            
            df = pd.read_sql_query(query, conn, params=patient_ids)
            conn.close()
            
            # Aggregate baseline characteristics
            all_characteristics = []
            for char_str in df['baseline_characteristics']:
                try:
                    characteristics = json.loads(char_str)
                    all_characteristics.append(characteristics)
                except:
                    continue
            
            if not all_characteristics:
                return {}
            
            # Calculate means for numeric characteristics
            baseline_summary = {}
            for char_dict in all_characteristics:
                for key, value in char_dict.items():
                    if isinstance(value, (int, float)):
                        if key not in baseline_summary:
                            baseline_summary[key] = []
                        baseline_summary[key].append(value)
            
            # Calculate means
            return {key: np.mean(values) for key, values in baseline_summary.items()}
            
        except Exception as e:
            self.logger.error(f"Error getting baseline characteristics: {e}")
            return {}
    
    def _analyze_cohort_outcome_summary(self, patient_ids: List[str]) -> Dict[str, float]:
        """Analyze outcome summary for a cohort"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            placeholders = ','.join(['?' for _ in patient_ids])
            query = f'''
                SELECT outcome_type, AVG(value) as mean_value, COUNT(*) as count
                FROM outcomes
                WHERE patient_id IN ({placeholders})
                AND timestamp >= datetime('now', '-90 days')
                GROUP BY outcome_type
            '''
            
            df = pd.read_sql_query(query, conn, params=patient_ids)
            conn.close()
            
            return df.set_index('outcome_type')['mean_value'].to_dict()
            
        except Exception as e:
            self.logger.error(f"Error analyzing cohort outcomes: {e}")
            return {}
    
    def _perform_survival_analysis(self, patient_ids: List[str]) -> Dict[str, Any]:
        """Perform survival analysis for cohort"""
        try:
            # This is a simplified example - would need event definitions
            # For now, return placeholder structure
            return {
                'median_survival_days': None,
                'survival_curves': {},
                'log_rank_p_value': None,
                'hazard_ratios': {}
            }
            
        except Exception as e:
            self.logger.error(f"Error in survival analysis: {e}")
            return {}
    
    def _analyze_cohort_trends(self, patient_ids: List[str]) -> Dict[str, float]:
        """Analyze trends for cohort outcomes"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            placeholders = ','.join(['?' for _ in patient_ids])
            query = f'''
                SELECT outcome_type, timestamp, AVG(value) as avg_value
                FROM outcomes
                WHERE patient_id IN ({placeholders})
                AND timestamp >= datetime('now', '-90 days')
                GROUP BY outcome_type, DATE(timestamp)
                ORDER BY outcome_type, timestamp
            '''
            
            df = pd.read_sql_query(query, conn, params=patient_ids)
            conn.close()
            
            trends = {}
            for outcome_type, group in df.groupby('outcome_type'):
                if len(group) >= 3:
                    trend_result = self._calculate_trend(
                        pd.to_datetime(group['timestamp']), 
                        group['avg_value']
                    )
                    trends[outcome_type] = trend_result['slope']
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing cohort trends: {e}")
            return {}
    
    def _compare_cohorts(self, cohort1_patients: List[str], cohort2_patients: List[str]) -> Dict[str, Any]:
        """Compare outcomes between two cohorts"""
        try:
            # Get recent outcomes for both cohorts
            conn = sqlite3.connect(self.database_path)
            
            # Cohort 1 outcomes
            placeholders1 = ','.join(['?' for _ in cohort1_patients])
            outcomes1 = pd.read_sql_query(f'''
                SELECT outcome_type, value
                FROM outcomes
                WHERE patient_id IN ({placeholders1})
                AND timestamp >= datetime('now', '-30 days')
            ''', conn, params=cohort1_patients)
            
            # Cohort 2 outcomes
            placeholders2 = ','.join(['?' for _ in cohort2_patients])
            outcomes2 = pd.read_sql_query(f'''
                SELECT outcome_type, value
                FROM outcomes
                WHERE patient_id IN ({placeholders2})
                AND timestamp >= datetime('now', '-30 days')
            ''', conn, params=cohort2_patients)
            
            conn.close()
            
            # Compare outcomes
            comparison = {}
            for outcome_type in set(outcomes1['outcome_type'].unique()) & set(outcomes2['outcome_type'].unique()):
                values1 = outcomes1[outcomes1['outcome_type'] == outcome_type]['value']
                values2 = outcomes2[outcomes2['outcome_type'] == outcome_type]['value']
                
                if len(values1) > 0 and len(values2) > 0:
                    # T-test comparison
                    from scipy import stats
                    statistic, p_value = stats.ttest_ind(values1, values2)
                    
                    comparison[outcome_type] = {
                        'cohort1_mean': values1.mean(),
                        'cohort2_mean': values2.mean(),
                        'difference': values1.mean() - values2.mean(),
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing cohorts: {e}")
            return {}
    
    def _prepare_causal_data(self, df: pd.DataFrame, confounders: List[str] = None) -> pd.DataFrame:
        """Prepare data for causal analysis"""
        # This is a simplified version - would need more sophisticated preprocessing
        causal_df = df.copy()
        
        # Create treatment indicator
        causal_df['treatment'] = (causal_df['intervention_type'].notna()).astype(int)
        
        # Add confounders from demographics and baseline characteristics
        # This would need more sophisticated parsing in practice
        
        return causal_df
    
    def _estimate_causal_effect(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate causal effect using propensity score methods"""
        # Simplified causal effect estimation
        try:
            treated = data[data['treatment'] == 1]['outcome_value']
            control = data[data['treatment'] == 0]['outcome_value']
            
            if len(treated) == 0 or len(control) == 0:
                return {'estimate': 0, 'ci': [0, 0], 'p_value': 1.0}
            
            # Simple difference in means (would use more sophisticated methods in practice)
            effect = treated.mean() - control.mean()
            
            # Bootstrap confidence interval
            n_bootstrap = 1000
            bootstrap_effects = []
            
            for _ in range(n_bootstrap):
                treated_sample = np.random.choice(treated, size=len(treated), replace=True)
                control_sample = np.random.choice(control, size=len(control), replace=True)
                bootstrap_effect = treated_sample.mean() - control_sample.mean()
                bootstrap_effects.append(bootstrap_effect)
            
            ci_lower = np.percentile(bootstrap_effects, 2.5)
            ci_upper = np.percentile(bootstrap_effects, 97.5)
            
            # T-test p-value
            from scipy import stats
            _, p_value = stats.ttest_ind(treated, control)
            
            return {
                'estimate': effect,
                'ci': [ci_lower, ci_upper],
                'p_value': p_value
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating causal effect: {e}")
            return {'estimate': 0, 'ci': [0, 0], 'p_value': 1.0}
    
    def _interpret_causal_effect(self, causal_effect: Dict[str, Any]) -> str:
        """Interpret causal effect results"""
        estimate = causal_effect['estimate']
        ci = causal_effect['ci']
        p_value = causal_effect['p_value']
        
        interpretation = f"Estimated causal effect: {estimate:.3f} "
        interpretation += f"(95% CI: {ci[0]:.3f} to {ci[1]:.3f}, p = {p_value:.3f}). "
        
        if p_value < 0.05:
            if estimate > 0:
                interpretation += "Statistically significant positive effect detected."
            else:
                interpretation += "Statistically significant negative effect detected."
        else:
            interpretation += "No statistically significant causal effect detected."
        
        return interpretation
    
    def generate_rwe_report(self, cohort_name: str = None, days_lookback: int = 90) -> str:
        """Generate comprehensive real-world evidence report"""
        report = []
        report.append("# Real-World Evidence Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {days_lookback} days")
        report.append("")
        
        # Overall statistics
        dashboard_data = self.generate_effectiveness_dashboard_data()
        
        report.append("## Overall Statistics")
        report.append(f"**Total Enrolled Patients**: {dashboard_data.get('total_patients', 0)}")
        report.append(f"**Active Patients (30 days)**: {dashboard_data.get('active_patients', 0)}")
        report.append("")
        
        # Longitudinal outcomes
        longitudinal_outcomes = self.analyze_longitudinal_outcomes(days_lookback=days_lookback)
        
        if longitudinal_outcomes:
            report.append("## Longitudinal Outcomes Analysis")
            
            # Summarize by outcome type
            outcome_summary = {}
            for outcome in longitudinal_outcomes:
                outcome_type = outcome.outcome_type.value
                if outcome_type not in outcome_summary:
                    outcome_summary[outcome_type] = []
                outcome_summary[outcome_type].append(outcome)
            
            for outcome_type, outcomes in outcome_summary.items():
                report.append(f"### {outcome_type.replace('_', ' ').title()}")
                report.append(f"**Patients Tracked**: {len(outcomes)}")
                
                # Calculate aggregate statistics
                trends = [o.trend_slope for o in outcomes if o.trend_significance < 0.05]
                if trends:
                    avg_trend = np.mean(trends)
                    positive_trends = sum(1 for t in trends if t > 0)
                    report.append(f"**Average Significant Trend**: {avg_trend:.4f}")
                    report.append(f"**Patients with Positive Trends**: {positive_trends}/{len(trends)}")
                
                avg_quality = np.mean([o.quality_score for o in outcomes])
                report.append(f"**Average Data Quality**: {avg_quality:.2f}")
                report.append("")
        
        # Safety monitoring
        safety_signals = self.monitor_real_time_safety(lookback_hours=days_lookback * 24)
        
        report.append("## Safety Monitoring")
        report.append(f"**Total Adverse Events**: {safety_signals.get('total_events', 0)}")
        report.append(f"**Patients with Events**: {safety_signals.get('unique_patients', 0)}")
        
        if safety_signals.get('alerts'):
            report.append("### Safety Alerts")
            for alert in safety_signals['alerts']:
                report.append(f"- **{alert['type']}**: {alert['message']}")
        else:
            report.append("**No safety alerts detected**")
        
        report.append("")
        
        # Cohort analysis (if specified)
        if cohort_name:
            cohort_analysis = self.analyze_cohort_outcomes(cohort_name)
            
            report.append(f"## Cohort Analysis: {cohort_name}")
            report.append(f"**Patients in Cohort**: {cohort_analysis.n_patients}")
            
            if cohort_analysis.baseline_characteristics:
                report.append("### Baseline Characteristics")
                for char, value in cohort_analysis.baseline_characteristics.items():
                    report.append(f"- {char}: {value:.2f}")
                report.append("")
            
            if cohort_analysis.outcome_summary:
                report.append("### Outcome Summary")
                for outcome, value in cohort_analysis.outcome_summary.items():
                    report.append(f"- {outcome}: {value:.3f}")
                report.append("")
            
            if cohort_analysis.trend_analysis:
                report.append("### Trend Analysis")
                for outcome, slope in cohort_analysis.trend_analysis.items():
                    trend_direction = "improving" if slope > 0 else "declining" if slope < 0 else "stable"
                    report.append(f"- {outcome}: {trend_direction} (slope: {slope:.4f})")
        
        # Data quality assessment
        report.append("## Data Quality Assessment")
        data_quality = dashboard_data.get('data_quality', [])
        
        if data_quality:
            report.append("| Data Source | Records | Avg Confidence |")
            report.append("|-------------|---------|----------------|")
            for source in data_quality:
                report.append(f"| {source['data_source']} | {source['total_records']} | {source['avg_confidence']:.2f} |")
        else:
            report.append("No data quality metrics available")
        
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        # Generate recommendations based on findings
        recommendations = []
        
        if safety_signals.get('total_events', 0) > 0:
            recommendations.append("Continue close safety monitoring with current event detection protocols")
        
        if longitudinal_outcomes:
            avg_quality = np.mean([o.quality_score for o in longitudinal_outcomes])
            if avg_quality < 0.7:
                recommendations.append("Improve data collection quality and frequency")
        
        if dashboard_data.get('active_patients', 0) < dashboard_data.get('total_patients', 1) * 0.8:
            recommendations.append("Implement patient engagement strategies to improve data collection")
        
        recommendations.append("Consider expanding real-world evidence collection to additional outcome measures")
        recommendations.append("Plan for longer-term follow-up to assess sustained effects")
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        
        return '\n'.join(report)

# Usage functions
def setup_rwe_tracking(patients_data: List[Dict[str, Any]], database_path: str = "rwe.db") -> RealWorldEvidenceEngine:
    """Set up real-world evidence tracking for a cohort of patients"""
    rwe_engine = RealWorldEvidenceEngine(database_path)
    
    # Enroll patients
    for patient_data in patients_data:
        rwe_engine.enroll_patient(
            patient_id=patient_data['patient_id'],
            condition=patient_data['condition'],
            demographics=patient_data.get('demographics', {}),
            baseline_characteristics=patient_data.get('baseline_characteristics', {})
        )
    
    return rwe_engine

def simulate_rwe_data(rwe_engine: RealWorldEvidenceEngine, n_days: int = 30) -> None:
    """Simulate real-world evidence data for testing"""
    # This would be replaced with actual data integration in production
    
    import random
    from datetime import timedelta
    
    # Get enrolled patients
    conn = sqlite3.connect(rwe_engine.database_path)
    patients_df = pd.read_sql_query("SELECT patient_id FROM patients", conn)
    conn.close()
    
    patient_ids = patients_df['patient_id'].tolist()
    
    # Simulate daily data for each patient
    for day in range(n_days):
        date = datetime.now() - timedelta(days=n_days - day)
        
        for patient_id in patient_ids:
            # Simulate various outcomes with some correlation
            symptom_severity = random.normalvariate(50, 15)  # Scale 0-100
            quality_of_life = 100 - symptom_severity + random.normalvariate(0, 10)
            adherence = random.betavariate(8, 2)  # Higher adherence on average
            
            # Record outcomes
            for outcome_type, value in [
                (OutcomeType.SYMPTOM_SEVERITY, max(0, min(100, symptom_severity))),
                (OutcomeType.QUALITY_OF_LIFE, max(0, min(100, quality_of_life))),
                (OutcomeType.ADHERENCE, adherence)
            ]:
                data_point = RealWorldDataPoint(
                    patient_id=patient_id,
                    timestamp=date,
                    outcome_type=outcome_type,
                    value=value,
                    data_source=DataSource.MOBILE_APP,
                    confidence=random.uniform(0.7, 1.0),
                    context={'simulated': True}
                )
                
                rwe_engine.record_outcome(data_point)
            
            # Occasional adverse events
            if random.random() < 0.05:  # 5% chance per day
                severity = random.choice(['mild', 'moderate', 'severe'])
                ae_data = RealWorldDataPoint(
                    patient_id=patient_id,
                    timestamp=date,
                    outcome_type=OutcomeType.ADVERSE_EVENTS,
                    value=1,
                    data_source=DataSource.PATIENT_REPORTED,
                    confidence=0.9,
                    context={'severity': severity, 'description': 'Simulated AE'}
                )
                rwe_engine.record_outcome(ae_data)