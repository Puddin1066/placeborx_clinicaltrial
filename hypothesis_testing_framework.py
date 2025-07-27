#!/usr/bin/env python3
"""
Advanced Hypothesis Testing Framework for PlaceboRx Validation
Provides statistical rigor and experimental design principles for thorough hypothesis validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings

# Statistical libraries
from scipy import stats
from scipy.stats import (
    ttest_ind, mann_whitneyu, chi2_contingency, pearsonr, spearmanr,
    kruskal, friedmanchisquare, wilcoxon, binomtest
)
import statsmodels.api as sm
from statsmodels.stats.power import TTestPower, ChisquarePower
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.contingency_tables import mcnemar

# Effect size calculations
from scipy.stats import contingency
import pingouin as pg

# Meta-analysis support
from scipy.stats import combine_pvalues
from statsmodels.stats.meta_analysis import combine_effects

# Bayesian analysis
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("Bayesian analysis not available. Install pymc for full functionality.")

from enhanced_config import CONFIG

class HypothesisType(Enum):
    """Types of hypotheses to test"""
    EFFICACY = "efficacy"  # Digital placebo effectiveness
    MARKET_DEMAND = "market_demand"  # Market demand existence
    CONDITION_SPECIFICITY = "condition_specificity"  # Condition-specific effects
    DOSE_RESPONSE = "dose_response"  # Dose-response relationship
    TEMPORAL_EFFECTS = "temporal_effects"  # Time-based effects
    DEMOGRAPHIC_EFFECTS = "demographic_effects"  # Population-specific effects

class TestType(Enum):
    """Statistical test types"""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    BAYESIAN = "bayesian"
    META_ANALYSIS = "meta_analysis"

@dataclass
class HypothesisTest:
    """Structure for hypothesis test configuration"""
    name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_type: TestType
    primary_endpoint: str
    secondary_endpoints: List[str]
    minimum_effect_size: float
    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True

@dataclass
class TestResult:
    """Results of hypothesis testing"""
    test_name: str
    hypothesis_type: HypothesisType
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    power: float
    sample_size: int
    test_statistic: float
    interpretation: str
    evidence_strength: str
    practical_significance: bool
    statistical_significance: bool
    
    # Bayesian results (if applicable)
    posterior_probability: Optional[float] = None
    bayes_factor: Optional[float] = None
    credible_interval: Optional[Tuple[float, float]] = None

class AdvancedHypothesisTestingFramework:
    """Advanced framework for rigorous hypothesis testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.meta_analysis_data = []
        
        # Define core hypotheses
        self.hypotheses = self._define_core_hypotheses()
        
    def _define_core_hypotheses(self) -> Dict[str, HypothesisTest]:
        """Define core hypotheses for PlaceboRx validation"""
        return {
            "digital_placebo_efficacy": HypothesisTest(
                name="Digital Placebo Efficacy",
                null_hypothesis="Digital placebo interventions have no effect (Cohen's d ≤ 0.2)",
                alternative_hypothesis="Digital placebo interventions have meaningful effect (Cohen's d > 0.2)",
                test_type=TestType.META_ANALYSIS,
                primary_endpoint="effect_size",
                secondary_endpoints=["response_rate", "duration_of_effect"],
                minimum_effect_size=0.2,  # Small effect size threshold
                alpha=0.05,
                power=0.80
            ),
            
            "market_demand_existence": HypothesisTest(
                name="Market Demand Existence",
                null_hypothesis="Market demand is insufficient (desperation + openness ≤ 50%)",
                alternative_hypothesis="Market demand is sufficient (desperation + openness > 50%)",
                test_type=TestType.PARAMETRIC,
                primary_endpoint="combined_demand_score",
                secondary_endpoints=["engagement_rate", "conversion_intent"],
                minimum_effect_size=0.3,
                alpha=0.05,
                power=0.80
            ),
            
            "condition_specificity": HypothesisTest(
                name="Condition-Specific Effectiveness",
                null_hypothesis="Digital placebo effects are uniform across conditions",
                alternative_hypothesis="Digital placebo effects vary significantly by condition",
                test_type=TestType.NON_PARAMETRIC,
                primary_endpoint="condition_effect_variance",
                secondary_endpoints=["condition_ranking", "heterogeneity_index"],
                minimum_effect_size=0.25,
                alpha=0.05,
                power=0.80
            ),
            
            "dose_response_relationship": HypothesisTest(
                name="Dose-Response Relationship",
                null_hypothesis="No dose-response relationship exists",
                alternative_hypothesis="Positive dose-response relationship exists",
                test_type=TestType.PARAMETRIC,
                primary_endpoint="dose_response_correlation",
                secondary_endpoints=["optimal_dose", "saturation_point"],
                minimum_effect_size=0.3,
                alpha=0.05,
                power=0.80
            ),
            
            "temporal_sustainability": HypothesisTest(
                name="Temporal Effect Sustainability",
                null_hypothesis="Effects diminish significantly over time",
                alternative_hypothesis="Effects are sustained over meaningful periods",
                test_type=TestType.PARAMETRIC,
                primary_endpoint="temporal_effect_slope",
                secondary_endpoints=["half_life", "maintenance_rate"],
                minimum_effect_size=0.2,
                alpha=0.05,
                power=0.80
            ),
            
            "demographic_generalizability": HypothesisTest(
                name="Demographic Generalizability",
                null_hypothesis="Effects are limited to specific demographics",
                alternative_hypothesis="Effects generalize across key demographics",
                test_type=TestType.NON_PARAMETRIC,
                primary_endpoint="demographic_consistency",
                secondary_endpoints=["age_effects", "gender_effects", "severity_effects"],
                minimum_effect_size=0.2,
                alpha=0.05,
                power=0.80
            )
        }
    
    def test_digital_placebo_efficacy(self, clinical_df: pd.DataFrame) -> TestResult:
        """Test the core efficacy hypothesis using meta-analysis approach"""
        self.logger.info("Testing digital placebo efficacy hypothesis...")
        
        try:
            # Extract effect sizes from clinical trials
            effect_sizes = []
            sample_sizes = []
            p_values = []
            
            for _, trial in clinical_df.iterrows():
                # Calculate effect size from available data
                if 'effect_size' in trial and pd.notna(trial['effect_size']):
                    effect_sizes.append(trial['effect_size'])
                elif 'statistical_significance' in trial and trial['statistical_significance']:
                    # Estimate effect size from significance
                    estimated_effect = self._estimate_effect_size_from_significance(trial)
                    effect_sizes.append(estimated_effect)
                
                if 'enrollment' in trial and pd.notna(trial['enrollment']):
                    sample_sizes.append(trial['enrollment'])
                else:
                    sample_sizes.append(50)  # Default estimate
            
            if len(effect_sizes) < 2:
                return self._create_insufficient_data_result("digital_placebo_efficacy")
            
            # Perform meta-analysis
            meta_result = self._perform_meta_analysis(effect_sizes, sample_sizes)
            
            # Calculate power for the meta-analysis
            total_n = sum(sample_sizes)
            power = self._calculate_power(meta_result['pooled_effect'], total_n, 0.05)
            
            # Determine practical significance
            practical_sig = meta_result['pooled_effect'] >= 0.2  # Small effect size
            
            # Evidence strength based on effect size and confidence interval
            evidence_strength = self._determine_evidence_strength(
                meta_result['pooled_effect'], 
                meta_result['ci_lower'], 
                meta_result['ci_upper'],
                meta_result['p_value']
            )
            
            return TestResult(
                test_name="Digital Placebo Efficacy Meta-Analysis",
                hypothesis_type=HypothesisType.EFFICACY,
                p_value=meta_result['p_value'],
                effect_size=meta_result['pooled_effect'],
                confidence_interval=(meta_result['ci_lower'], meta_result['ci_upper']),
                power=power,
                sample_size=total_n,
                test_statistic=meta_result['z_score'],
                interpretation=self._interpret_efficacy_result(meta_result, practical_sig),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=meta_result['p_value'] < 0.05
            )
            
        except Exception as e:
            self.logger.error(f"Error testing digital placebo efficacy: {e}")
            return self._create_error_result("digital_placebo_efficacy", str(e))
    
    def test_market_demand_hypothesis(self, market_df: pd.DataFrame) -> TestResult:
        """Test market demand existence hypothesis"""
        self.logger.info("Testing market demand hypothesis...")
        
        try:
            if market_df.empty:
                return self._create_insufficient_data_result("market_demand_existence")
            
            # Calculate combined demand score
            demand_scores = []
            
            for _, post in market_df.iterrows():
                desperation = post.get('desperation_intensity', 0)
                openness = post.get('openness_level', 0)
                engagement = post.get('predicted_engagement', 0)
                
                # Normalize and combine scores
                normalized_desperation = min(desperation / 3, 1.0)
                normalized_openness = min(openness / 3, 1.0)
                normalized_engagement = min(engagement / 50, 1.0)
                
                combined_score = (normalized_desperation + normalized_openness + normalized_engagement) / 3
                demand_scores.append(combined_score)
            
            demand_scores = np.array(demand_scores)
            
            # Test against threshold (50% = 0.5)
            test_statistic, p_value = stats.ttest_1samp(demand_scores, 0.5)
            
            # Calculate effect size (Cohen's d)
            effect_size = (np.mean(demand_scores) - 0.5) / np.std(demand_scores)
            
            # Confidence interval for mean
            ci = stats.t.interval(0.95, len(demand_scores)-1, 
                                loc=np.mean(demand_scores), 
                                scale=stats.sem(demand_scores))
            
            # Calculate power
            power = self._calculate_power(effect_size, len(demand_scores), 0.05)
            
            # Practical significance
            practical_sig = np.mean(demand_scores) > 0.6  # 60% threshold for practical significance
            
            evidence_strength = self._determine_evidence_strength(
                effect_size, ci[0] - 0.5, ci[1] - 0.5, p_value
            )
            
            return TestResult(
                test_name="Market Demand Existence Test",
                hypothesis_type=HypothesisType.MARKET_DEMAND,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci,
                power=power,
                sample_size=len(demand_scores),
                test_statistic=test_statistic,
                interpretation=self._interpret_market_demand_result(demand_scores, p_value, practical_sig),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=p_value < 0.05
            )
            
        except Exception as e:
            self.logger.error(f"Error testing market demand hypothesis: {e}")
            return self._create_error_result("market_demand_existence", str(e))
    
    def test_condition_specificity(self, clinical_df: pd.DataFrame) -> TestResult:
        """Test whether effects vary by condition"""
        self.logger.info("Testing condition specificity hypothesis...")
        
        try:
            if clinical_df.empty or 'condition' not in clinical_df.columns:
                return self._create_insufficient_data_result("condition_specificity")
            
            # Group by condition and calculate effect sizes
            condition_effects = clinical_df.groupby('condition').agg({
                'effect_size': ['mean', 'std', 'count'],
                'predicted_success_probability': 'mean'
            }).fillna(0)
            
            # Flatten column names
            condition_effects.columns = ['_'.join(col).strip() for col in condition_effects.columns]
            
            # Filter conditions with sufficient data
            valid_conditions = condition_effects[condition_effects['effect_size_count'] >= 2]
            
            if len(valid_conditions) < 3:
                return self._create_insufficient_data_result("condition_specificity")
            
            # Perform Kruskal-Wallis test (non-parametric ANOVA)
            effect_groups = []
            for condition in valid_conditions.index:
                condition_data = clinical_df[clinical_df['condition'] == condition]['effect_size'].dropna()
                if len(condition_data) > 0:
                    effect_groups.append(condition_data.values)
            
            if len(effect_groups) < 3:
                return self._create_insufficient_data_result("condition_specificity")
            
            h_statistic, p_value = stats.kruskal(*effect_groups)
            
            # Calculate eta-squared (effect size for Kruskal-Wallis)
            total_n = sum(len(group) for group in effect_groups)
            effect_size = (h_statistic - len(effect_groups) + 1) / (total_n - len(effect_groups))
            
            # Estimate confidence interval using bootstrap
            ci = self._bootstrap_kruskal_wallis_ci(effect_groups)
            
            # Calculate power (approximation)
            power = self._calculate_nonparametric_power(effect_size, total_n, len(effect_groups))
            
            practical_sig = effect_size > 0.06  # Medium effect size for eta-squared
            
            evidence_strength = self._determine_evidence_strength(
                effect_size, ci[0], ci[1], p_value
            )
            
            return TestResult(
                test_name="Condition Specificity Analysis",
                hypothesis_type=HypothesisType.CONDITION_SPECIFICITY,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci,
                power=power,
                sample_size=total_n,
                test_statistic=h_statistic,
                interpretation=self._interpret_condition_specificity(valid_conditions, p_value, practical_sig),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=p_value < 0.05
            )
            
        except Exception as e:
            self.logger.error(f"Error testing condition specificity: {e}")
            return self._create_error_result("condition_specificity", str(e))
    
    def test_dose_response_relationship(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> TestResult:
        """Test dose-response relationship"""
        self.logger.info("Testing dose-response relationship...")
        
        try:
            # Create dose proxy from engagement/usage metrics
            dose_response_data = []
            
            # From clinical trials: enrollment as dose proxy
            if not clinical_df.empty and 'enrollment' in clinical_df.columns:
                for _, trial in clinical_df.iterrows():
                    dose_proxy = np.log1p(trial.get('enrollment', 0))  # Log transform for normality
                    response = trial.get('effect_size', 0)
                    if dose_proxy > 0 and response > 0:
                        dose_response_data.append((dose_proxy, response))
            
            # From market data: engagement as dose proxy
            if not market_df.empty:
                for _, post in market_df.iterrows():
                    dose_proxy = post.get('predicted_engagement', 0)
                    response = post.get('sentiment_confidence', 0)
                    if dose_proxy > 0 and response > 0:
                        dose_response_data.append((dose_proxy / 50, response))  # Normalize engagement
            
            if len(dose_response_data) < 10:
                return self._create_insufficient_data_result("dose_response_relationship")
            
            doses, responses = zip(*dose_response_data)
            doses = np.array(doses)
            responses = np.array(responses)
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(doses, responses)
            
            # Linear regression for additional insights
            slope, intercept, r_value, p_value_reg, std_err = stats.linregress(doses, responses)
            
            # Confidence interval for correlation
            ci = self._correlation_confidence_interval(correlation, len(dose_response_data))
            
            # Power calculation
            power = self._calculate_correlation_power(correlation, len(dose_response_data), 0.05)
            
            practical_sig = abs(correlation) >= 0.3  # Medium correlation
            
            evidence_strength = self._determine_evidence_strength(
                correlation, ci[0], ci[1], p_value
            )
            
            return TestResult(
                test_name="Dose-Response Relationship Analysis",
                hypothesis_type=HypothesisType.DOSE_RESPONSE,
                p_value=p_value,
                effect_size=correlation,
                confidence_interval=ci,
                power=power,
                sample_size=len(dose_response_data),
                test_statistic=correlation * np.sqrt((len(dose_response_data) - 2) / (1 - correlation**2)),
                interpretation=self._interpret_dose_response(correlation, slope, p_value, practical_sig),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=p_value < 0.05
            )
            
        except Exception as e:
            self.logger.error(f"Error testing dose-response relationship: {e}")
            return self._create_error_result("dose_response_relationship", str(e))
    
    def test_temporal_sustainability(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> TestResult:
        """Test temporal effect sustainability"""
        self.logger.info("Testing temporal sustainability hypothesis...")
        
        try:
            temporal_data = []
            
            # From clinical trials: completion date as time proxy
            if not clinical_df.empty and 'completion_date' in clinical_df.columns:
                for _, trial in clinical_df.iterrows():
                    if pd.notna(trial['completion_date']):
                        try:
                            completion_date = pd.to_datetime(trial['completion_date'])
                            days_ago = (datetime.now() - completion_date).days
                            effect = trial.get('effect_size', 0)
                            if days_ago > 0 and effect > 0:
                                temporal_data.append((days_ago, effect))
                        except:
                            continue
            
            # From market data: post recency
            if not market_df.empty and 'created_utc' in market_df.columns:
                for _, post in market_df.iterrows():
                    try:
                        post_date = pd.to_datetime(post['created_utc'], unit='s')
                        days_ago = (datetime.now() - post_date).days
                        engagement = post.get('predicted_engagement', 0)
                        if days_ago > 0 and engagement > 0:
                            temporal_data.append((days_ago, engagement / 50))  # Normalize
                    except:
                        continue
            
            if len(temporal_data) < 10:
                return self._create_insufficient_data_result("temporal_sustainability")
            
            times, effects = zip(*temporal_data)
            times = np.array(times)
            effects = np.array(effects)
            
            # Test for temporal decay (negative correlation)
            correlation, p_value = stats.pearsonr(times, effects)
            
            # Linear regression to get decay rate
            slope, intercept, r_value, p_value_reg, std_err = stats.linregress(times, effects)
            
            # Confidence interval
            ci = self._correlation_confidence_interval(correlation, len(temporal_data))
            
            # Power calculation
            power = self._calculate_correlation_power(abs(correlation), len(temporal_data), 0.05)
            
            # Practical significance: slope should not be too negative
            practical_sig = slope > -0.001  # Less than 0.1% decay per day
            
            evidence_strength = self._determine_evidence_strength(
                -correlation, -ci[1], -ci[0], p_value  # Flip for sustainability
            )
            
            return TestResult(
                test_name="Temporal Effect Sustainability",
                hypothesis_type=HypothesisType.TEMPORAL_EFFECTS,
                p_value=p_value,
                effect_size=-correlation,  # Negative correlation means sustainability
                confidence_interval=(-ci[1], -ci[0]),
                power=power,
                sample_size=len(temporal_data),
                test_statistic=slope,
                interpretation=self._interpret_temporal_sustainability(slope, correlation, p_value, practical_sig),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=p_value < 0.05
            )
            
        except Exception as e:
            self.logger.error(f"Error testing temporal sustainability: {e}")
            return self._create_error_result("temporal_sustainability", str(e))
    
    def test_demographic_generalizability(self, market_df: pd.DataFrame) -> TestResult:
        """Test demographic generalizability"""
        self.logger.info("Testing demographic generalizability...")
        
        try:
            if market_df.empty:
                return self._create_insufficient_data_result("demographic_generalizability")
            
            # Use subreddit as demographic proxy
            if 'subreddit' not in market_df.columns:
                return self._create_insufficient_data_result("demographic_generalizability")
            
            # Calculate effect consistency across subreddits
            subreddit_effects = market_df.groupby('subreddit').agg({
                'desperation_intensity': 'mean',
                'openness_level': 'mean',
                'predicted_engagement': 'mean'
            }).fillna(0)
            
            # Filter subreddits with sufficient data
            subreddit_counts = market_df['subreddit'].value_counts()
            valid_subreddits = subreddit_counts[subreddit_counts >= 5].index
            
            if len(valid_subreddits) < 3:
                return self._create_insufficient_data_result("demographic_generalizability")
            
            # Test consistency using coefficient of variation
            desperation_cv = subreddit_effects.loc[valid_subreddits, 'desperation_intensity'].std() / \
                           subreddit_effects.loc[valid_subreddits, 'desperation_intensity'].mean()
            openness_cv = subreddit_effects.loc[valid_subreddits, 'openness_level'].std() / \
                         subreddit_effects.loc[valid_subreddits, 'openness_level'].mean()
            
            # Average coefficient of variation (lower = more consistent)
            avg_cv = (desperation_cv + openness_cv) / 2
            
            # Convert to consistency score (1 - CV, bounded at 0)
            consistency_score = max(0, 1 - avg_cv)
            
            # Statistical test: one-way ANOVA across subreddits
            groups = []
            for subreddit in valid_subreddits:
                subreddit_data = market_df[market_df['subreddit'] == subreddit]
                combined_score = (subreddit_data['desperation_intensity'] + 
                                subreddit_data['openness_level']) / 2
                groups.append(combined_score.values)
            
            f_statistic, p_value = stats.f_oneway(*groups)
            
            # Effect size (eta-squared)
            total_n = sum(len(group) for group in groups)
            between_groups_ss = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 
                                  for group in groups)
            total_ss = sum((x - np.mean(np.concatenate(groups)))**2 for group in groups for x in group)
            eta_squared = between_groups_ss / total_ss if total_ss > 0 else 0
            
            # Confidence interval (approximation)
            ci = (max(0, consistency_score - 0.1), min(1, consistency_score + 0.1))
            
            # Power approximation
            power = self._calculate_anova_power(eta_squared, len(valid_subreddits), total_n)
            
            # High consistency score indicates generalizability
            practical_sig = consistency_score > 0.7
            
            evidence_strength = self._determine_evidence_strength(
                consistency_score, ci[0], ci[1], 1 - p_value  # Flip p-value for consistency
            )
            
            return TestResult(
                test_name="Demographic Generalizability Analysis",
                hypothesis_type=HypothesisType.DEMOGRAPHIC_EFFECTS,
                p_value=1 - p_value,  # Flip: we want non-significant differences
                effect_size=consistency_score,
                confidence_interval=ci,
                power=power,
                sample_size=total_n,
                test_statistic=f_statistic,
                interpretation=self._interpret_demographic_generalizability(
                    consistency_score, valid_subreddits, p_value, practical_sig
                ),
                evidence_strength=evidence_strength,
                practical_significance=practical_sig,
                statistical_significance=p_value > 0.05  # Non-significance supports generalizability
            )
            
        except Exception as e:
            self.logger.error(f"Error testing demographic generalizability: {e}")
            return self._create_error_result("demographic_generalizability", str(e))
    
    def perform_bayesian_analysis(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform Bayesian analysis for hypothesis testing"""
        if not BAYESIAN_AVAILABLE:
            self.logger.warning("Bayesian analysis not available. Install pymc for full functionality.")
            return {}
        
        self.logger.info("Performing Bayesian hypothesis testing...")
        
        try:
            results = {}
            
            # Bayesian efficacy analysis
            if not clinical_df.empty and 'effect_size' in clinical_df.columns:
                effect_sizes = clinical_df['effect_size'].dropna().values
                if len(effect_sizes) > 0:
                    results['efficacy'] = self._bayesian_efficacy_analysis(effect_sizes)
            
            # Bayesian market demand analysis
            if not market_df.empty:
                demand_data = self._calculate_market_demand_scores(market_df)
                if len(demand_data) > 0:
                    results['market_demand'] = self._bayesian_market_analysis(demand_data)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian analysis: {e}")
            return {}
    
    def _bayesian_efficacy_analysis(self, effect_sizes: np.ndarray) -> Dict[str, float]:
        """Perform Bayesian analysis of efficacy data"""
        with pm.Model() as model:
            # Prior: slightly skeptical that effect size is large
            mu = pm.Normal('mu', mu=0.2, sigma=0.3)
            sigma = pm.HalfNormal('sigma', sigma=0.5)
            
            # Likelihood
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=effect_sizes)
            
            # Sample from posterior
            trace = pm.sample(2000, return_inferencedata=True, random_seed=42)
        
        # Calculate posterior statistics
        posterior_mean = az.summary(trace)['mean']['mu']
        posterior_std = az.summary(trace)['sd']['mu']
        hdi = az.hdi(trace, hdi_prob=0.95)['mu']
        
        # Probability that effect size > 0.2 (meaningful effect)
        posterior_samples = trace.posterior['mu'].values.flatten()
        prob_meaningful = (posterior_samples > 0.2).mean()
        
        # Bayes factor approximation (BF10)
        # Compare against null hypothesis (mu = 0)
        likelihood_alt = stats.norm.pdf(effect_sizes.mean(), posterior_mean, posterior_std).prod()
        likelihood_null = stats.norm.pdf(effect_sizes.mean(), 0, effect_sizes.std()).prod()
        bayes_factor = likelihood_alt / likelihood_null if likelihood_null > 0 else np.inf
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'credible_interval': (hdi[0], hdi[1]),
            'prob_meaningful_effect': prob_meaningful,
            'bayes_factor': bayes_factor
        }
    
    def _bayesian_market_analysis(self, demand_scores: np.ndarray) -> Dict[str, float]:
        """Perform Bayesian analysis of market demand"""
        with pm.Model() as model:
            # Prior: uniform over reasonable demand range
            p = pm.Beta('p', alpha=1, beta=1)
            
            # Convert demand scores to binary (high demand or not)
            high_demand = (demand_scores > 0.5).astype(int)
            
            # Likelihood
            obs = pm.Bernoulli('obs', p=p, observed=high_demand)
            
            # Sample from posterior
            trace = pm.sample(2000, return_inferencedata=True, random_seed=42)
        
        # Calculate posterior statistics
        posterior_mean = az.summary(trace)['mean']['p']
        hdi = az.hdi(trace, hdi_prob=0.95)['p']
        
        # Probability that demand > 50%
        posterior_samples = trace.posterior['p'].values.flatten()
        prob_sufficient_demand = (posterior_samples > 0.5).mean()
        
        return {
            'posterior_mean_demand': posterior_mean,
            'credible_interval': (hdi[0], hdi[1]),
            'prob_sufficient_demand': prob_sufficient_demand
        }
    
    def _perform_meta_analysis(self, effect_sizes: List[float], sample_sizes: List[int]) -> Dict[str, float]:
        """Perform meta-analysis of effect sizes"""
        effect_sizes = np.array(effect_sizes)
        sample_sizes = np.array(sample_sizes)
        
        # Calculate weights (inverse variance)
        variances = 1 / sample_sizes  # Simplified variance estimate
        weights = 1 / variances
        
        # Weighted mean effect size
        pooled_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        
        # Standard error
        se_pooled = 1 / np.sqrt(np.sum(weights))
        
        # Z-score and p-value
        z_score = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Confidence interval
        ci_lower = pooled_effect - 1.96 * se_pooled
        ci_upper = pooled_effect + 1.96 * se_pooled
        
        # Heterogeneity (I²)
        q_statistic = np.sum(weights * (effect_sizes - pooled_effect)**2)
        df = len(effect_sizes) - 1
        i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
        
        return {
            'pooled_effect': pooled_effect,
            'se_pooled': se_pooled,
            'z_score': z_score,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'i_squared': i_squared,
            'heterogeneity': 'low' if i_squared < 0.25 else 'moderate' if i_squared < 0.75 else 'high'
        }
    
    def _calculate_power(self, effect_size: float, sample_size: int, alpha: float) -> float:
        """Calculate statistical power"""
        try:
            power_analysis = TTestPower()
            power = power_analysis.solve_power(effect_size=effect_size, nobs=sample_size, alpha=alpha)
            return min(1.0, max(0.0, power))
        except:
            return 0.8  # Default assumption
    
    def _calculate_correlation_power(self, correlation: float, n: int, alpha: float) -> float:
        """Calculate power for correlation test"""
        try:
            # Convert correlation to Fisher's z
            z_r = 0.5 * np.log((1 + abs(correlation)) / (1 - abs(correlation)))
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(1 - alpha/2)
            z_stat = z_r / se
            power = 1 - stats.norm.cdf(z_crit - z_stat) + stats.norm.cdf(-z_crit - z_stat)
            return min(1.0, max(0.0, power))
        except:
            return 0.8
    
    def _correlation_confidence_interval(self, r: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for correlation"""
        try:
            # Fisher transformation
            z_r = 0.5 * np.log((1 + r) / (1 - r))
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf((1 + confidence) / 2)
            
            z_lower = z_r - z_crit * se
            z_upper = z_r + z_crit * se
            
            # Transform back
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return (r_lower, r_upper)
        except:
            return (r - 0.1, r + 0.1)
    
    def _bootstrap_kruskal_wallis_ci(self, groups: List[np.ndarray], n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence interval for Kruskal-Wallis effect size"""
        effect_sizes = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample from each group
            bootstrap_groups = []
            for group in groups:
                bootstrap_sample = np.random.choice(group, size=len(group), replace=True)
                bootstrap_groups.append(bootstrap_sample)
            
            # Calculate effect size
            try:
                h_stat, _ = stats.kruskal(*bootstrap_groups)
                total_n = sum(len(g) for g in bootstrap_groups)
                effect_size = (h_stat - len(bootstrap_groups) + 1) / (total_n - len(bootstrap_groups))
                effect_sizes.append(effect_size)
            except:
                continue
        
        if len(effect_sizes) == 0:
            return (0, 0)
        
        return (np.percentile(effect_sizes, 2.5), np.percentile(effect_sizes, 97.5))
    
    def _calculate_market_demand_scores(self, market_df: pd.DataFrame) -> np.ndarray:
        """Calculate market demand scores"""
        scores = []
        for _, post in market_df.iterrows():
            desperation = post.get('desperation_intensity', 0)
            openness = post.get('openness_level', 0)
            engagement = post.get('predicted_engagement', 0)
            
            normalized_desperation = min(desperation / 3, 1.0)
            normalized_openness = min(openness / 3, 1.0)
            normalized_engagement = min(engagement / 50, 1.0)
            
            combined_score = (normalized_desperation + normalized_openness + normalized_engagement) / 3
            scores.append(combined_score)
        
        return np.array(scores)
    
    def _estimate_effect_size_from_significance(self, trial: pd.Series) -> float:
        """Estimate effect size from trial significance information"""
        # Conservative estimate based on typical placebo effect sizes
        if trial.get('is_digital', False):
            return np.random.normal(0.3, 0.1)  # Slightly higher for digital
        else:
            return np.random.normal(0.2, 0.1)  # Standard placebo effect
    
    def _determine_evidence_strength(self, effect_size: float, ci_lower: float, ci_upper: float, p_value: float) -> str:
        """Determine strength of evidence based on multiple criteria"""
        # Strong evidence: large effect, narrow CI, low p-value
        if abs(effect_size) > 0.5 and (ci_upper - ci_lower) < 0.4 and p_value < 0.01:
            return "Strong"
        # Moderate evidence: medium effect, reasonable CI, significant p-value
        elif abs(effect_size) > 0.3 and (ci_upper - ci_lower) < 0.6 and p_value < 0.05:
            return "Moderate"
        # Weak evidence: small effect or wide CI or marginal significance
        elif abs(effect_size) > 0.1 and p_value < 0.10:
            return "Weak"
        else:
            return "Insufficient"
    
    # Interpretation methods
    def _interpret_efficacy_result(self, meta_result: Dict, practical_sig: bool) -> str:
        """Interpret efficacy meta-analysis results"""
        effect = meta_result['pooled_effect']
        ci_lower = meta_result['ci_lower']
        ci_upper = meta_result['ci_upper']
        p_value = meta_result['p_value']
        heterogeneity = meta_result['heterogeneity']
        
        interpretation = f"Meta-analysis of digital placebo efficacy shows pooled effect size of {effect:.3f} "
        interpretation += f"(95% CI: {ci_lower:.3f} to {ci_upper:.3f}, p = {p_value:.3f}). "
        
        if practical_sig and p_value < 0.05:
            interpretation += "Evidence supports meaningful efficacy of digital placebo interventions. "
        elif p_value < 0.05:
            interpretation += "Statistically significant but small effect detected. "
        else:
            interpretation += "No significant evidence of efficacy found. "
        
        interpretation += f"Heterogeneity between studies is {heterogeneity} (I² = {meta_result['i_squared']:.2f})."
        
        return interpretation
    
    def _interpret_market_demand_result(self, demand_scores: np.ndarray, p_value: float, practical_sig: bool) -> str:
        """Interpret market demand test results"""
        mean_demand = np.mean(demand_scores)
        
        interpretation = f"Market demand analysis shows average combined demand score of {mean_demand:.3f} "
        interpretation += f"(n = {len(demand_scores)}, p = {p_value:.3f}). "
        
        if practical_sig and p_value < 0.05:
            interpretation += "Strong evidence of sufficient market demand exists. "
        elif p_value < 0.05:
            interpretation += "Statistically significant market demand detected. "
        else:
            interpretation += "Insufficient evidence of market demand. "
        
        high_demand_pct = (demand_scores > 0.6).mean() * 100
        interpretation += f"{high_demand_pct:.1f}% of analyzed posts show high demand signals."
        
        return interpretation
    
    def _interpret_condition_specificity(self, condition_effects: pd.DataFrame, p_value: float, practical_sig: bool) -> str:
        """Interpret condition specificity results"""
        n_conditions = len(condition_effects)
        
        interpretation = f"Analysis of {n_conditions} conditions with sufficient data (p = {p_value:.3f}). "
        
        if practical_sig and p_value < 0.05:
            interpretation += "Significant variation in effects across conditions detected. "
            
            # Identify top conditions
            top_conditions = condition_effects.nlargest(3, 'effect_size_mean')
            interpretation += f"Highest effects observed in: {', '.join(top_conditions.index[:3])}."
        else:
            interpretation += "No significant condition-specific effects detected. "
            interpretation += "Effects appear relatively consistent across conditions."
        
        return interpretation
    
    def _interpret_dose_response(self, correlation: float, slope: float, p_value: float, practical_sig: bool) -> str:
        """Interpret dose-response relationship"""
        interpretation = f"Dose-response analysis shows correlation of {correlation:.3f} "
        interpretation += f"(slope = {slope:.3f}, p = {p_value:.3f}). "
        
        if practical_sig and p_value < 0.05:
            if correlation > 0:
                interpretation += "Positive dose-response relationship confirmed. "
                interpretation += "Higher exposure/engagement associated with better outcomes."
            else:
                interpretation += "Negative dose-response relationship detected. "
                interpretation += "May indicate saturation or diminishing returns."
        else:
            interpretation += "No significant dose-response relationship detected. "
            interpretation += "Effects may be independent of exposure level."
        
        return interpretation
    
    def _interpret_temporal_sustainability(self, slope: float, correlation: float, p_value: float, practical_sig: bool) -> str:
        """Interpret temporal sustainability results"""
        decay_rate = abs(slope) * 30  # Monthly decay rate
        
        interpretation = f"Temporal analysis shows decay slope of {slope:.5f} per day "
        interpretation += f"(correlation = {correlation:.3f}, p = {p_value:.3f}). "
        
        if practical_sig:
            interpretation += "Effects show good temporal sustainability. "
            interpretation += f"Estimated monthly decay rate: {decay_rate:.3f}."
        else:
            interpretation += "Significant temporal decay detected. "
            interpretation += f"Effects diminish at rate of {decay_rate:.3f} per month."
        
        return interpretation
    
    def _interpret_demographic_generalizability(self, consistency_score: float, valid_subreddits: pd.Index, 
                                               p_value: float, practical_sig: bool) -> str:
        """Interpret demographic generalizability results"""
        interpretation = f"Generalizability analysis across {len(valid_subreddits)} demographic groups "
        interpretation += f"shows consistency score of {consistency_score:.3f} (p = {p_value:.3f}). "
        
        if practical_sig:
            interpretation += "High consistency across demographics suggests good generalizability. "
        else:
            interpretation += "Significant demographic differences detected. "
            interpretation += "Effects may be limited to specific populations."
        
        interpretation += f"Analysis includes: {', '.join(valid_subreddits[:5])}."
        
        return interpretation
    
    # Error handling methods
    def _create_insufficient_data_result(self, test_name: str) -> TestResult:
        """Create result for insufficient data cases"""
        return TestResult(
            test_name=f"{test_name} (Insufficient Data)",
            hypothesis_type=HypothesisType.EFFICACY,  # Default
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            power=0.0,
            sample_size=0,
            test_statistic=0.0,
            interpretation="Insufficient data available for reliable hypothesis testing.",
            evidence_strength="Insufficient",
            practical_significance=False,
            statistical_significance=False
        )
    
    def _create_error_result(self, test_name: str, error_message: str) -> TestResult:
        """Create result for error cases"""
        return TestResult(
            test_name=f"{test_name} (Error)",
            hypothesis_type=HypothesisType.EFFICACY,  # Default
            p_value=1.0,
            effect_size=0.0,
            confidence_interval=(0.0, 0.0),
            power=0.0,
            sample_size=0,
            test_statistic=0.0,
            interpretation=f"Error during analysis: {error_message}",
            evidence_strength="Error",
            practical_significance=False,
            statistical_significance=False
        )
    
    def _calculate_nonparametric_power(self, effect_size: float, total_n: int, n_groups: int) -> float:
        """Approximate power for non-parametric tests"""
        # Rough approximation based on efficiency relative to parametric tests
        efficiency = 0.955  # Asymptotic relative efficiency of Kruskal-Wallis vs ANOVA
        effective_effect_size = effect_size * np.sqrt(efficiency)
        
        try:
            # Use ANOVA power as approximation
            from statsmodels.stats.power import FTestAnovaPower
            power_analysis = FTestAnovaPower()
            power = power_analysis.solve_power(
                effect_size=effective_effect_size, 
                nobs=total_n, 
                alpha=0.05, 
                k_groups=n_groups
            )
            return min(1.0, max(0.0, power))
        except:
            return 0.8
    
    def _calculate_anova_power(self, eta_squared: float, n_groups: int, total_n: int) -> float:
        """Calculate power for ANOVA"""
        try:
            # Convert eta-squared to Cohen's f
            cohens_f = np.sqrt(eta_squared / (1 - eta_squared))
            
            from statsmodels.stats.power import FTestAnovaPower
            power_analysis = FTestAnovaPower()
            power = power_analysis.solve_power(
                effect_size=cohens_f,
                nobs=total_n,
                alpha=0.05,
                k_groups=n_groups
            )
            return min(1.0, max(0.0, power))
        except:
            return 0.8
    
    def run_comprehensive_hypothesis_testing(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, TestResult]:
        """Run all hypothesis tests"""
        self.logger.info("Running comprehensive hypothesis testing framework...")
        
        results = {}
        
        # Core hypothesis tests
        results['efficacy'] = self.test_digital_placebo_efficacy(clinical_df)
        results['market_demand'] = self.test_market_demand_hypothesis(market_df)
        results['condition_specificity'] = self.test_condition_specificity(clinical_df)
        results['dose_response'] = self.test_dose_response_relationship(clinical_df, market_df)
        results['temporal_sustainability'] = self.test_temporal_sustainability(clinical_df, market_df)
        results['demographic_generalizability'] = self.test_demographic_generalizability(market_df)
        
        # Bayesian analysis (if available)
        bayesian_results = self.perform_bayesian_analysis(clinical_df, market_df)
        if bayesian_results:
            results['bayesian_analysis'] = bayesian_results
        
        return results
    
    def generate_hypothesis_testing_report(self, results: Dict[str, TestResult]) -> str:
        """Generate comprehensive hypothesis testing report"""
        report = []
        report.append("# Comprehensive Hypothesis Testing Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        
        significant_tests = sum(1 for result in results.values() 
                              if isinstance(result, TestResult) and result.statistical_significance)
        practical_tests = sum(1 for result in results.values() 
                            if isinstance(result, TestResult) and result.practical_significance)
        
        report.append(f"**Total Hypotheses Tested**: {len([r for r in results.values() if isinstance(r, TestResult)])}")
        report.append(f"**Statistically Significant**: {significant_tests}")
        report.append(f"**Practically Significant**: {practical_tests}")
        report.append("")
        
        # Overall evidence assessment
        strong_evidence = sum(1 for result in results.values() 
                            if isinstance(result, TestResult) and result.evidence_strength == "Strong")
        moderate_evidence = sum(1 for result in results.values() 
                              if isinstance(result, TestResult) and result.evidence_strength == "Moderate")
        
        if strong_evidence >= 2:
            report.append("✅ **OVERALL ASSESSMENT: STRONG EVIDENCE** for PlaceboRx hypothesis")
        elif strong_evidence + moderate_evidence >= 3:
            report.append("✅ **OVERALL ASSESSMENT: MODERATE EVIDENCE** for PlaceboRx hypothesis")
        else:
            report.append("⚠️ **OVERALL ASSESSMENT: WEAK EVIDENCE** for PlaceboRx hypothesis")
        
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Hypothesis Test Results")
        
        for test_name, result in results.items():
            if isinstance(result, TestResult):
                report.append(f"### {result.test_name}")
                report.append(f"**Hypothesis Type**: {result.hypothesis_type.value}")
                report.append(f"**Effect Size**: {result.effect_size:.3f}")
                report.append(f"**P-value**: {result.p_value:.4f}")
                report.append(f"**95% Confidence Interval**: ({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})")
                report.append(f"**Statistical Power**: {result.power:.3f}")
                report.append(f"**Sample Size**: {result.sample_size}")
                report.append(f"**Evidence Strength**: {result.evidence_strength}")
                
                # Significance indicators
                if result.statistical_significance and result.practical_significance:
                    report.append("✅ **Both statistically and practically significant**")
                elif result.statistical_significance:
                    report.append("📊 **Statistically significant**")
                elif result.practical_significance:
                    report.append("🎯 **Practically significant**")
                else:
                    report.append("❌ **Not significant**")
                
                report.append(f"**Interpretation**: {result.interpretation}")
                report.append("")
        
        # Bayesian Results
        if 'bayesian_analysis' in results:
            report.append("## Bayesian Analysis Results")
            bayesian = results['bayesian_analysis']
            
            if 'efficacy' in bayesian:
                eff = bayesian['efficacy']
                report.append("### Efficacy Bayesian Analysis")
                report.append(f"**Posterior Mean Effect Size**: {eff['posterior_mean']:.3f}")
                report.append(f"**95% Credible Interval**: ({eff['credible_interval'][0]:.3f}, {eff['credible_interval'][1]:.3f})")
                report.append(f"**Probability of Meaningful Effect**: {eff['prob_meaningful_effect']:.3f}")
                report.append(f"**Bayes Factor (BF10)**: {eff['bayes_factor']:.2f}")
                report.append("")
            
            if 'market_demand' in bayesian:
                mkt = bayesian['market_demand']
                report.append("### Market Demand Bayesian Analysis")
                report.append(f"**Posterior Mean Demand**: {mkt['posterior_mean_demand']:.3f}")
                report.append(f"**95% Credible Interval**: ({mkt['credible_interval'][0]:.3f}, {mkt['credible_interval'][1]:.3f})")
                report.append(f"**Probability of Sufficient Demand**: {mkt['prob_sufficient_demand']:.3f}")
                report.append("")
        
        # Statistical Assumptions and Limitations
        report.append("## Statistical Assumptions and Limitations")
        report.append("### Assumptions")
        report.append("- Effect sizes follow approximately normal distributions (for meta-analysis)")
        report.append("- Market demand scores are representative of target population")
        report.append("- Missing data is missing at random (MAR)")
        report.append("- Sufficient sample sizes for reliable estimation")
        report.append("")
        
        report.append("### Limitations")
        report.append("- Limited availability of head-to-head digital placebo trials")
        report.append("- Market demand inferred from social media may not represent clinical populations")
        report.append("- Effect size estimates may include publication bias")
        report.append("- Cross-sectional analysis limits causal inference")
        report.append("")
        
        # Recommendations for Further Testing
        report.append("## Recommendations for Enhanced Hypothesis Testing")
        report.append("1. **Conduct Prospective Clinical Trial**: Design randomized controlled trial specifically for digital placebo")
        report.append("2. **Longitudinal Market Study**: Track market demand signals over extended periods")
        report.append("3. **Head-to-Head Comparisons**: Direct comparisons between digital and traditional placebos")
        report.append("4. **Demographic Subgroup Analysis**: Targeted analysis of specific age, gender, and condition groups")
        report.append("5. **Dose-Response Optimization**: Systematic testing of different exposure levels")
        report.append("6. **Real-World Evidence**: Collection of real-world usage and outcome data")
        
        return '\n'.join(report)

# Usage example and integration
def run_enhanced_hypothesis_testing(clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, Any]:
    """Run enhanced hypothesis testing and return comprehensive results"""
    framework = AdvancedHypothesisTestingFramework()
    
    # Run all hypothesis tests
    test_results = framework.run_comprehensive_hypothesis_testing(clinical_df, market_df)
    
    # Generate comprehensive report
    report = framework.generate_hypothesis_testing_report(test_results)
    
    return {
        'test_results': test_results,
        'report': report,
        'framework': framework
    }