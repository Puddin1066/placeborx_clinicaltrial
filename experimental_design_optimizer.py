#!/usr/bin/env python3
"""
Experimental Design Optimizer for PlaceboRx Hypothesis Testing
Provides optimal study designs, power analysis, and sample size calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import itertools

# Statistical design libraries
from scipy import stats
from statsmodels.stats.power import TTestPower, FTestAnovaPower
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.contingency_tables import mcnemar_effectsize
import statsmodels.api as sm

# Experimental design
try:
    from pyDOE2 import lhs, fullfact, fracfact
    DOE_AVAILABLE = True
except ImportError:
    DOE_AVAILABLE = False
    
# Bayesian experimental design
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

from enhanced_config import CONFIG

class StudyType(Enum):
    """Types of studies to design"""
    RCT = "randomized_controlled_trial"
    CROSSOVER = "crossover_trial"
    FACTORIAL = "factorial_design"
    ADAPTIVE = "adaptive_trial"
    OBSERVATIONAL = "observational_study"
    PILOT = "pilot_study"
    DOSE_FINDING = "dose_finding_study"

class Endpoint(Enum):
    """Primary endpoint types"""
    CONTINUOUS = "continuous"
    BINARY = "binary"
    TIME_TO_EVENT = "time_to_event"
    COUNT = "count"
    ORDINAL = "ordinal"

@dataclass
class StudyParameters:
    """Parameters for study design"""
    study_type: StudyType
    primary_endpoint: Endpoint
    effect_size: float
    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True
    allocation_ratio: float = 1.0  # Control:Treatment ratio
    dropout_rate: float = 0.10
    interim_analyses: int = 0
    
    # Condition-specific parameters
    baseline_rate: Optional[float] = None  # For binary outcomes
    baseline_mean: Optional[float] = None  # For continuous outcomes
    baseline_sd: Optional[float] = None    # For continuous outcomes

@dataclass
class StudyDesign:
    """Complete study design specification"""
    name: str
    study_type: StudyType
    sample_size_per_group: int
    total_sample_size: int
    duration_weeks: int
    primary_endpoint: str
    secondary_endpoints: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    randomization_scheme: str
    blinding: str
    interim_analyses: List[int]  # Week numbers
    power: float
    detectable_effect_size: float
    estimated_cost: float
    feasibility_score: float
    
    # Digital placebo specific
    intervention_components: List[str]
    dose_schedule: str
    follow_up_schedule: List[int]  # Follow-up weeks

class ExperimentalDesignOptimizer:
    """Optimizer for experimental design to test PlaceboRx hypothesis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def optimize_study_design(self, 
                            target_conditions: List[str],
                            available_budget: float,
                            time_constraints: int,  # months
                            priority_hypotheses: List[str]) -> List[StudyDesign]:
        """Optimize study design based on constraints and priorities"""
        
        self.logger.info("Optimizing experimental designs...")
        
        designs = []
        
        # Generate designs for each priority hypothesis
        for hypothesis in priority_hypotheses:
            if hypothesis == "efficacy":
                designs.extend(self._design_efficacy_studies(target_conditions, available_budget, time_constraints))
            elif hypothesis == "dose_response":
                designs.extend(self._design_dose_response_studies(target_conditions, available_budget, time_constraints))
            elif hypothesis == "condition_specificity":
                designs.extend(self._design_condition_specificity_studies(target_conditions, available_budget, time_constraints))
            elif hypothesis == "temporal_sustainability":
                designs.extend(self._design_longitudinal_studies(target_conditions, available_budget, time_constraints))
        
        # Rank designs by feasibility and power
        ranked_designs = self._rank_designs(designs, available_budget, time_constraints)
        
        return ranked_designs[:5]  # Return top 5 designs
    
    def _design_efficacy_studies(self, conditions: List[str], budget: float, time_months: int) -> List[StudyDesign]:
        """Design studies to test digital placebo efficacy"""
        designs = []
        
        for condition in conditions[:3]:  # Top 3 conditions
            # Standard RCT design
            rct_design = self._create_efficacy_rct(condition, budget, time_months)
            if rct_design:
                designs.append(rct_design)
            
            # Crossover design (if suitable)
            if condition in ['chronic pain', 'anxiety', 'insomnia']:
                crossover_design = self._create_efficacy_crossover(condition, budget, time_months)
                if crossover_design:
                    designs.append(crossover_design)
            
            # Adaptive design
            adaptive_design = self._create_adaptive_efficacy_trial(condition, budget, time_months)
            if adaptive_design:
                designs.append(adaptive_design)
        
        return designs
    
    def _create_efficacy_rct(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create standard RCT design for efficacy testing"""
        
        # Estimate parameters based on condition
        effect_size = self._estimate_condition_effect_size(condition)
        baseline_improvement = self._estimate_baseline_improvement(condition)
        
        # Power analysis for sample size
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=0.80,
            alpha=0.05
        )
        
        # Account for dropout
        adjusted_sample_size = int(sample_size / 0.9)  # 10% dropout
        total_sample_size = adjusted_sample_size * 2  # Two groups
        
        # Estimate cost and duration
        estimated_cost = self._estimate_study_cost(total_sample_size, 12, StudyType.RCT)
        
        if estimated_cost > budget * 0.8:  # Use max 80% of budget for single study
            return None
        
        if time_months < 15:  # Need at least 15 months for full RCT
            return None
        
        return StudyDesign(
            name=f"Digital Placebo RCT for {condition.title()}",
            study_type=StudyType.RCT,
            sample_size_per_group=adjusted_sample_size,
            total_sample_size=total_sample_size,
            duration_weeks=52,  # 1 year
            primary_endpoint=f"{condition}_severity_change",
            secondary_endpoints=[
                f"{condition}_response_rate",
                "quality_of_life_change",
                "treatment_satisfaction",
                "adherence_rate"
            ],
            inclusion_criteria=[
                f"Diagnosed {condition}",
                "Age 18-75",
                "Stable medication regimen",
                "Smartphone access",
                "English proficiency"
            ],
            exclusion_criteria=[
                "Severe psychiatric comorbidity",
                "Substance abuse",
                "Pregnancy",
                "Participation in other trials"
            ],
            randomization_scheme="Permuted block randomization (block size 4)",
            blinding="Double-blind (participants and assessors)",
            interim_analyses=[26],  # Mid-study analysis
            power=0.80,
            detectable_effect_size=effect_size,
            estimated_cost=estimated_cost,
            feasibility_score=0.8,
            intervention_components=[
                "Digital placebo app",
                "Educational content",
                "Symptom tracking",
                "Behavioral exercises"
            ],
            dose_schedule="Daily app engagement (20-30 minutes)",
            follow_up_schedule=[2, 4, 8, 12, 26, 39, 52]
        )
    
    def _create_efficacy_crossover(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create crossover design for efficacy testing"""
        
        effect_size = self._estimate_condition_effect_size(condition) * 1.2  # Crossover more sensitive
        
        # Crossover requires fewer participants
        power_analysis = TTestPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            power=0.80,
            alpha=0.05
        )
        
        # Crossover needs about 50% fewer participants
        crossover_sample_size = int(sample_size * 0.5 / 0.9)  # Account for dropout
        
        estimated_cost = self._estimate_study_cost(crossover_sample_size, 16, StudyType.CROSSOVER)
        
        if estimated_cost > budget * 0.6:
            return None
        
        if time_months < 12:
            return None
        
        return StudyDesign(
            name=f"Digital Placebo Crossover Trial for {condition.title()}",
            study_type=StudyType.CROSSOVER,
            sample_size_per_group=crossover_sample_size,
            total_sample_size=crossover_sample_size,
            duration_weeks=32,  # 8 weeks per period + washout
            primary_endpoint=f"{condition}_severity_change",
            secondary_endpoints=[
                f"{condition}_response_rate",
                "period_effect_assessment",
                "carryover_effect_assessment"
            ],
            inclusion_criteria=[
                f"Stable {condition}",
                "Age 18-65",
                "No major medication changes expected",
                "Smartphone access"
            ],
            exclusion_criteria=[
                "Rapidly changing condition",
                "Poor app compliance history",
                "Cognitive impairment"
            ],
            randomization_scheme="AB/BA crossover with balanced sequences",
            blinding="Double-blind with matching placebo app",
            interim_analyses=[16],
            power=0.80,
            detectable_effect_size=effect_size,
            estimated_cost=estimated_cost,
            feasibility_score=0.7,  # More complex logistics
            intervention_components=[
                "Digital placebo app",
                "Control app (minimal features)",
                "Washout period monitoring"
            ],
            dose_schedule="8 weeks active, 8 weeks control (with 2-week washouts)",
            follow_up_schedule=[2, 6, 8, 10, 14, 16, 18, 22, 24, 32]
        )
    
    def _create_adaptive_efficacy_trial(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create adaptive trial design"""
        
        effect_size = self._estimate_condition_effect_size(condition)
        
        # Start with smaller sample, allow for expansion
        initial_sample = 100
        max_sample = 300
        
        estimated_cost = self._estimate_study_cost(max_sample, 18, StudyType.ADAPTIVE)
        
        if estimated_cost > budget:
            return None
        
        return StudyDesign(
            name=f"Adaptive Digital Placebo Trial for {condition.title()}",
            study_type=StudyType.ADAPTIVE,
            sample_size_per_group=initial_sample // 2,
            total_sample_size=max_sample,  # Maximum possible
            duration_weeks=78,  # 18 months
            primary_endpoint=f"{condition}_severity_change",
            secondary_endpoints=[
                f"{condition}_response_rate",
                "optimal_dose_identification",
                "subgroup_effects"
            ],
            inclusion_criteria=[
                f"Diagnosed {condition}",
                "Age 18-70",
                "Smartphone access",
                "Willing to continue for extended period"
            ],
            exclusion_criteria=[
                "Severe comorbidities",
                "History of poor study compliance"
            ],
            randomization_scheme="Response-adaptive randomization",
            blinding="Single-blind (assessor-blinded)",
            interim_analyses=[12, 26, 39, 52],  # Quarterly reviews
            power=0.80,
            detectable_effect_size=effect_size,
            estimated_cost=estimated_cost,
            feasibility_score=0.6,  # Complex design
            intervention_components=[
                "Adaptive digital placebo",
                "Real-time dose adjustment",
                "Personalized content",
                "ML-driven optimization"
            ],
            dose_schedule="Variable based on response (daily to weekly)",
            follow_up_schedule=[2, 4, 8, 12, 16, 20, 26, 32, 39, 52, 65, 78]
        )
    
    def _design_dose_response_studies(self, conditions: List[str], budget: float, time_months: int) -> List[StudyDesign]:
        """Design dose-response studies"""
        designs = []
        
        primary_condition = conditions[0] if conditions else "chronic pain"
        
        # Factorial design for dose-response
        factorial_design = self._create_factorial_dose_study(primary_condition, budget, time_months)
        if factorial_design:
            designs.append(factorial_design)
        
        # Phase II dose-finding study
        dose_finding_design = self._create_dose_finding_study(primary_condition, budget, time_months)
        if dose_finding_design:
            designs.append(dose_finding_design)
        
        return designs
    
    def _create_factorial_dose_study(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create factorial design for dose-response"""
        
        # 3x2 factorial: 3 doses x 2 frequencies
        n_groups = 6
        effect_size = 0.4  # Medium effect for dose response
        
        # ANOVA power analysis
        power_analysis = FTestAnovaPower()
        sample_size_per_group = power_analysis.solve_power(
            effect_size=effect_size,
            power=0.80,
            alpha=0.05,
            k_groups=n_groups
        )
        
        total_sample = int(sample_size_per_group * n_groups / 0.9)  # Account for dropout
        
        estimated_cost = self._estimate_study_cost(total_sample, 24, StudyType.FACTORIAL)
        
        if estimated_cost > budget * 0.7:
            return None
        
        return StudyDesign(
            name=f"Factorial Dose-Response Study for {condition.title()}",
            study_type=StudyType.FACTORIAL,
            sample_size_per_group=int(sample_size_per_group),
            total_sample_size=total_sample,
            duration_weeks=24,
            primary_endpoint="dose_response_slope",
            secondary_endpoints=[
                "optimal_dose_identification",
                "dose_by_frequency_interaction",
                "safety_by_dose"
            ],
            inclusion_criteria=[
                f"Mild to moderate {condition}",
                "Age 18-65",
                "Medication stable",
                "High smartphone usage"
            ],
            exclusion_criteria=[
                "Severe symptoms",
                "Poor digital literacy",
                "Irregular schedule"
            ],
            randomization_scheme="Balanced factorial randomization",
            blinding="Single-blind (outcome assessor)",
            interim_analyses=[12],
            power=0.80,
            detectable_effect_size=effect_size,
            estimated_cost=estimated_cost,
            feasibility_score=0.7,
            intervention_components=[
                "Low dose (10 min/day)",
                "Medium dose (20 min/day)", 
                "High dose (40 min/day)",
                "Daily vs 3x/week schedule"
            ],
            dose_schedule="Factorial combination of dose and frequency",
            follow_up_schedule=[4, 8, 12, 16, 20, 24]
        )
    
    def _create_dose_finding_study(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create dose-finding study with continuous reassessment"""
        
        sample_size = 120  # Typical Phase II sample
        estimated_cost = self._estimate_study_cost(sample_size, 16, StudyType.DOSE_FINDING)
        
        if estimated_cost > budget * 0.5:
            return None
        
        return StudyDesign(
            name=f"Digital Placebo Dose-Finding Study for {condition.title()}",
            study_type=StudyType.DOSE_FINDING,
            sample_size_per_group=30,  # 4 dose groups
            total_sample_size=sample_size,
            duration_weeks=16,
            primary_endpoint="optimal_biological_dose",
            secondary_endpoints=[
                "dose_limiting_toxicity",
                "efficacy_by_dose",
                "pharmacodynamic_markers"
            ],
            inclusion_criteria=[
                f"Treatment-naive {condition}",
                "Age 18-60",
                "Good digital literacy",
                "Willing for intensive monitoring"
            ],
            exclusion_criteria=[
                "Prior digital therapy",
                "Severe depression",
                "Poor compliance history"
            ],
            randomization_scheme="Continual reassessment method (CRM)",
            blinding="Open-label with blinded assessment",
            interim_analyses=[4, 8, 12],  # Monthly reviews
            power=0.75,  # Lower for exploratory study
            detectable_effect_size=0.5,
            estimated_cost=estimated_cost,
            feasibility_score=0.8,
            intervention_components=[
                "Escalating exposure protocol",
                "Safety monitoring",
                "Biomarker collection",
                "Real-time dose adjustment"
            ],
            dose_schedule="Dose escalation: 5, 15, 30, 60 minutes daily",
            follow_up_schedule=[1, 2, 4, 6, 8, 10, 12, 14, 16, 20]
        )
    
    def _design_condition_specificity_studies(self, conditions: List[str], budget: float, time_months: int) -> List[StudyDesign]:
        """Design studies to test condition specificity"""
        designs = []
        
        if len(conditions) >= 3:
            # Multi-condition comparison study
            multi_condition_design = self._create_multi_condition_study(conditions[:4], budget, time_months)
            if multi_condition_design:
                designs.append(multi_condition_design)
        
        return designs
    
    def _create_multi_condition_study(self, conditions: List[str], budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create study comparing effects across multiple conditions"""
        
        n_conditions = len(conditions)
        effect_size = 0.35  # Medium effect for between-condition differences
        
        # Power analysis for multi-group comparison
        power_analysis = FTestAnovaPower()
        sample_size_per_condition = power_analysis.solve_power(
            effect_size=effect_size,
            power=0.80,
            alpha=0.05,
            k_groups=n_conditions
        )
        
        total_sample = int(sample_size_per_condition * n_conditions / 0.85)  # Higher dropout expected
        
        estimated_cost = self._estimate_study_cost(total_sample, 20, StudyType.RCT)
        
        if estimated_cost > budget:
            return None
        
        return StudyDesign(
            name="Multi-Condition Digital Placebo Specificity Study",
            study_type=StudyType.RCT,
            sample_size_per_group=int(sample_size_per_condition),
            total_sample_size=total_sample,
            duration_weeks=20,
            primary_endpoint="condition_specific_response_rate",
            secondary_endpoints=[
                "effect_size_by_condition",
                "response_pattern_analysis",
                "cross_condition_efficacy"
            ],
            inclusion_criteria=[
                "Primary diagnosis of target condition",
                "Age 18-70",
                "Stable treatment regimen",
                "Smartphone proficiency"
            ],
            exclusion_criteria=[
                "Multiple primary conditions",
                "Severe psychiatric comorbidity",
                "Poor treatment adherence history"
            ],
            randomization_scheme="Stratified by condition, then randomized",
            blinding="Double-blind with condition-specific placebos",
            interim_analyses=[10],
            power=0.80,
            detectable_effect_size=effect_size,
            estimated_cost=estimated_cost,
            feasibility_score=0.6,  # Complex multi-condition logistics
            intervention_components=[
                f"Condition-specific apps for: {', '.join(conditions)}",
                "Standardized core features",
                "Condition-tailored content",
                "Cross-condition analysis tools"
            ],
            dose_schedule="Daily engagement, condition-specific duration",
            follow_up_schedule=[2, 5, 10, 15, 20, 24]
        )
    
    def _design_longitudinal_studies(self, conditions: List[str], budget: float, time_months: int) -> List[StudyDesign]:
        """Design longitudinal studies for temporal effects"""
        designs = []
        
        primary_condition = conditions[0] if conditions else "chronic pain"
        
        # Long-term follow-up study
        long_term_design = self._create_long_term_followup_study(primary_condition, budget, time_months)
        if long_term_design:
            designs.append(long_term_design)
        
        return designs
    
    def _create_long_term_followup_study(self, condition: str, budget: float, time_months: int) -> Optional[StudyDesign]:
        """Create long-term follow-up study"""
        
        if time_months < 24:  # Need at least 2 years
            return None
        
        sample_size = 200  # Moderate size for long-term study
        estimated_cost = self._estimate_study_cost(sample_size, 104, StudyType.OBSERVATIONAL)  # 2 years
        
        if estimated_cost > budget * 0.8:
            return None
        
        return StudyDesign(
            name=f"Long-term Digital Placebo Sustainability Study for {condition.title()}",
            study_type=StudyType.OBSERVATIONAL,
            sample_size_per_group=100,
            total_sample_size=sample_size,
            duration_weeks=104,  # 2 years
            primary_endpoint="long_term_effect_sustainability",
            secondary_endpoints=[
                "time_to_effect_decay",
                "maintenance_dose_requirements",
                "long_term_safety_profile",
                "quality_of_life_trajectory"
            ],
            inclusion_criteria=[
                f"Previous responders to digital placebo for {condition}",
                "Age 18-75",
                "Committed to long-term participation",
                "Stable living situation"
            ],
            exclusion_criteria=[
                "Major life changes expected",
                "Severe medical comorbidities",
                "History of poor study retention"
            ],
            randomization_scheme="Maintenance vs. withdrawal design",
            blinding="Single-blind (outcome assessor)",
            interim_analyses=[26, 52, 78],  # Every 6 months
            power=0.80,
            detectable_effect_size=0.3,
            estimated_cost=estimated_cost,
            feasibility_score=0.5,  # Challenging retention
            intervention_components=[
                "Maintenance digital placebo",
                "Gradual withdrawal protocol",
                "Relapse prevention strategies",
                "Long-term safety monitoring"
            ],
            dose_schedule="Flexible maintenance dosing",
            follow_up_schedule=[4, 8, 13, 26, 39, 52, 65, 78, 91, 104]
        )
    
    def calculate_optimal_sample_size(self, 
                                    study_params: StudyParameters) -> Dict[str, Any]:
        """Calculate optimal sample size with detailed power analysis"""
        
        results = {}
        
        try:
            if study_params.primary_endpoint == Endpoint.CONTINUOUS:
                # T-test power analysis
                power_analysis = TTestPower()
                sample_size = power_analysis.solve_power(
                    effect_size=study_params.effect_size,
                    power=study_params.power,
                    alpha=study_params.alpha
                )
                
                # Adjust for allocation ratio
                if study_params.allocation_ratio != 1.0:
                    n1 = sample_size * (1 + study_params.allocation_ratio) / (2 * study_params.allocation_ratio)
                    n2 = n1 * study_params.allocation_ratio
                    sample_size = n1 + n2
                
            elif study_params.primary_endpoint == Endpoint.BINARY:
                # Proportion test power analysis
                if study_params.baseline_rate is None:
                    study_params.baseline_rate = 0.3  # Default assumption
                
                effect_size_prop = proportion_effectsize(
                    study_params.baseline_rate,
                    study_params.baseline_rate + study_params.effect_size
                )
                
                # Use normal approximation for proportions
                z_alpha = stats.norm.ppf(1 - study_params.alpha/2) if study_params.two_sided else stats.norm.ppf(1 - study_params.alpha)
                z_beta = stats.norm.ppf(study_params.power)
                
                p1 = study_params.baseline_rate
                p2 = study_params.baseline_rate + study_params.effect_size
                p_pooled = (p1 + p2) / 2
                
                sample_size = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
                             z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (p2 - p1)**2
                
            else:
                # Default to continuous outcome approach
                power_analysis = TTestPower()
                sample_size = power_analysis.solve_power(
                    effect_size=study_params.effect_size,
                    power=study_params.power,
                    alpha=study_params.alpha
                )
            
            # Adjust for dropout
            adjusted_sample_size = sample_size / (1 - study_params.dropout_rate)
            
            # Account for interim analyses (inflate alpha)
            if study_params.interim_analyses > 0:
                # O'Brien-Fleming adjustment (simplified)
                alpha_adjusted = study_params.alpha / (1 + 0.5 * study_params.interim_analyses)
                
                if study_params.primary_endpoint == Endpoint.CONTINUOUS:
                    power_analysis = TTestPower() # Re-initialize for interim analysis
                    sample_size_interim = power_analysis.solve_power(
                        effect_size=study_params.effect_size,
                        power=study_params.power,
                        alpha=alpha_adjusted
                    )
                    adjusted_sample_size = sample_size_interim / (1 - study_params.dropout_rate)
            
            results = {
                'sample_size_per_group': int(np.ceil(adjusted_sample_size / 2)),
                'total_sample_size': int(np.ceil(adjusted_sample_size)),
                'unadjusted_sample_size': int(np.ceil(sample_size)),
                'dropout_inflation_factor': 1 / (1 - study_params.dropout_rate),
                'interim_inflation_factor': adjusted_sample_size / sample_size if study_params.interim_analyses > 0 else 1.0,
                'power_achieved': study_params.power,
                'alpha_used': study_params.alpha,
                'effect_size': study_params.effect_size
            }
            
            # Power curve analysis
            sample_sizes = np.arange(10, int(adjusted_sample_size * 2), 10)
            powers = []
            
            for n in sample_sizes:
                if study_params.primary_endpoint == Endpoint.CONTINUOUS:
                    power = power_analysis.solve_power(
                        effect_size=study_params.effect_size,
                        nobs=n,
                        alpha=study_params.alpha
                    )
                else:
                    # Simplified power calculation for other endpoints
                    power = 0.8 * (n / adjusted_sample_size)
                powers.append(min(power, 1.0))
            
            results['power_curve'] = {
                'sample_sizes': sample_sizes.tolist(),
                'powers': powers
            }
            
        except Exception as e:
            self.logger.error(f"Error in sample size calculation: {e}")
            results = {
                'sample_size_per_group': 50,  # Default
                'total_sample_size': 100,
                'error': str(e)
            }
        
        return results
    
    def _estimate_condition_effect_size(self, condition: str) -> float:
        """Estimate expected effect size for condition"""
        effect_sizes = {
            'chronic pain': 0.4,
            'anxiety': 0.5,
            'depression': 0.3,
            'insomnia': 0.6,
            'fibromyalgia': 0.3,
            'IBS': 0.4,
            'migraine': 0.4,
            'PTSD': 0.4
        }
        return effect_sizes.get(condition.lower(), 0.3)  # Default medium effect
    
    def _estimate_baseline_improvement(self, condition: str) -> float:
        """Estimate baseline improvement rate"""
        baselines = {
            'chronic pain': 0.25,
            'anxiety': 0.30,
            'depression': 0.20,
            'insomnia': 0.35,
            'fibromyalgia': 0.20,
            'IBS': 0.30,
            'migraine': 0.25,
            'PTSD': 0.15
        }
        return baselines.get(condition.lower(), 0.25)
    
    def _estimate_study_cost(self, sample_size: int, duration_weeks: int, study_type: StudyType) -> float:
        """Estimate study cost based on parameters"""
        
        # Base costs per participant
        base_cost_per_participant = {
            StudyType.RCT: 2000,
            StudyType.CROSSOVER: 2500,
            StudyType.FACTORIAL: 3000,
            StudyType.ADAPTIVE: 4000,
            StudyType.OBSERVATIONAL: 1000,
            StudyType.PILOT: 1500,
            StudyType.DOSE_FINDING: 3500
        }
        
        base_cost = base_cost_per_participant.get(study_type, 2000)
        
        # Duration multiplier
        duration_multiplier = 1 + (duration_weeks - 12) * 0.02  # 2% per week beyond 12
        
        # Complexity multiplier
        complexity_multiplier = {
            StudyType.RCT: 1.0,
            StudyType.CROSSOVER: 1.3,
            StudyType.FACTORIAL: 1.5,
            StudyType.ADAPTIVE: 2.0,
            StudyType.OBSERVATIONAL: 0.7,
            StudyType.PILOT: 0.8,
            StudyType.DOSE_FINDING: 1.8
        }
        
        total_cost = (sample_size * base_cost * duration_multiplier * 
                     complexity_multiplier.get(study_type, 1.0))
        
        # Add fixed costs (infrastructure, regulatory, etc.)
        fixed_costs = 50000  # Base infrastructure
        
        return total_cost + fixed_costs
    
    def _rank_designs(self, designs: List[StudyDesign], budget: float, time_months: int) -> List[StudyDesign]:
        """Rank study designs by feasibility and scientific value"""
        
        scored_designs = []
        
        for design in designs:
            score = self._calculate_design_score(design, budget, time_months)
            scored_designs.append((score, design))
        
        # Sort by score (descending)
        scored_designs.sort(key=lambda x: x[0], reverse=True)
        
        return [design for score, design in scored_designs]
    
    def _calculate_design_score(self, design: StudyDesign, budget: float, time_months: int) -> float:
        """Calculate composite score for study design"""
        
        score = 0.0
        
        # Budget feasibility (30%)
        if design.estimated_cost <= budget:
            budget_score = 1.0 - (design.estimated_cost / budget * 0.5)  # Prefer lower cost
        else:
            budget_score = 0.0  # Infeasible
        score += budget_score * 0.3
        
        # Time feasibility (20%)
        required_months = design.duration_weeks / 4.33 + 6  # Add setup time
        if required_months <= time_months:
            time_score = 1.0 - (required_months / time_months * 0.3)
        else:
            time_score = 0.0
        score += time_score * 0.2
        
        # Statistical power (25%)
        power_score = design.power
        score += power_score * 0.25
        
        # Scientific impact (15%)
        impact_scores = {
            StudyType.RCT: 0.9,
            StudyType.ADAPTIVE: 1.0,
            StudyType.FACTORIAL: 0.8,
            StudyType.CROSSOVER: 0.7,
            StudyType.DOSE_FINDING: 0.8,
            StudyType.OBSERVATIONAL: 0.6,
            StudyType.PILOT: 0.5
        }
        impact_score = impact_scores.get(design.study_type, 0.6)
        score += impact_score * 0.15
        
        # Feasibility score (10%)
        score += design.feasibility_score * 0.1
        
        return min(score, 1.0)
    
    def generate_protocol_template(self, design: StudyDesign) -> str:
        """Generate protocol template for selected design"""
        
        protocol = []
        protocol.append(f"# Study Protocol: {design.name}")
        protocol.append(f"Protocol Version: 1.0")
        protocol.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        protocol.append("")
        
        # Study overview
        protocol.append("## 1. Study Overview")
        protocol.append(f"**Study Design**: {design.study_type.value.replace('_', ' ').title()}")
        protocol.append(f"**Primary Endpoint**: {design.primary_endpoint}")
        protocol.append(f"**Sample Size**: {design.total_sample_size} participants")
        protocol.append(f"**Study Duration**: {design.duration_weeks} weeks")
        protocol.append(f"**Estimated Cost**: ${design.estimated_cost:,.0f}")
        protocol.append("")
        
        # Objectives
        protocol.append("## 2. Objectives")
        protocol.append("### Primary Objective")
        protocol.append(f"To evaluate the efficacy of digital placebo intervention on {design.primary_endpoint}")
        protocol.append("")
        protocol.append("### Secondary Objectives")
        for endpoint in design.secondary_endpoints:
            protocol.append(f"- To assess {endpoint}")
        protocol.append("")
        
        # Study population
        protocol.append("## 3. Study Population")
        protocol.append(f"**Target Sample Size**: {design.total_sample_size}")
        protocol.append(f"**Sample Size per Group**: {design.sample_size_per_group}")
        protocol.append("")
        
        protocol.append("### Inclusion Criteria")
        for criterion in design.inclusion_criteria:
            protocol.append(f"- {criterion}")
        protocol.append("")
        
        protocol.append("### Exclusion Criteria")
        for criterion in design.exclusion_criteria:
            protocol.append(f"- {criterion}")
        protocol.append("")
        
        # Study procedures
        protocol.append("## 4. Study Procedures")
        protocol.append(f"**Randomization**: {design.randomization_scheme}")
        protocol.append(f"**Blinding**: {design.blinding}")
        protocol.append("")
        
        protocol.append("### Intervention Components")
        for component in design.intervention_components:
            protocol.append(f"- {component}")
        protocol.append("")
        
        protocol.append(f"**Dose Schedule**: {design.dose_schedule}")
        protocol.append("")
        
        # Follow-up schedule
        protocol.append("### Follow-up Schedule")
        protocol.append("| Visit | Week | Assessments |")
        protocol.append("|-------|------|-------------|")
        protocol.append("| Baseline | 0 | Eligibility, randomization, baseline measures |")
        
        for i, week in enumerate(design.follow_up_schedule):
            visit_name = f"Follow-up {i+1}"
            if week in design.interim_analyses:
                visit_name += " (Interim Analysis)"
            protocol.append(f"| {visit_name} | {week} | Primary/secondary endpoints, safety |")
        
        protocol.append("")
        
        # Statistical analysis
        protocol.append("## 5. Statistical Analysis")
        protocol.append(f"**Primary Analysis**: Intention-to-treat analysis comparing {design.primary_endpoint}")
        protocol.append(f"**Statistical Power**: {design.power}")
        protocol.append(f"**Detectable Effect Size**: {design.detectable_effect_size}")
        protocol.append(f"**Significance Level**: 0.05 (two-sided)")
        protocol.append("")
        
        if design.interim_analyses:
            protocol.append("### Interim Analyses")
            protocol.append(f"Planned interim analyses at weeks: {', '.join(map(str, design.interim_analyses))}")
            protocol.append("O'Brien-Fleming spending function will be used for alpha adjustment.")
            protocol.append("")
        
        # Timeline and milestones
        protocol.append("## 6. Study Timeline")
        protocol.append("| Milestone | Timeline |")
        protocol.append("|-----------|----------|")
        protocol.append("| Protocol finalization | Month 1 |")
        protocol.append("| Regulatory approval | Month 2-3 |")
        protocol.append("| Site initiation | Month 4 |")
        protocol.append("| First patient enrolled | Month 5 |")
        protocol.append("| Last patient enrolled | Month 12 |")
        protocol.append(f"| Study completion | Month {int(design.duration_weeks/4.33) + 5} |")
        protocol.append("| Database lock | Month {int(design.duration_weeks/4.33) + 6} |")
        protocol.append("| Final report | Month {int(design.duration_weeks/4.33) + 8} |")
        protocol.append("")
        
        # Budget breakdown
        protocol.append("## 7. Budget Considerations")
        cost_per_participant = design.estimated_cost / design.total_sample_size
        protocol.append(f"**Total Estimated Cost**: ${design.estimated_cost:,.0f}")
        protocol.append(f"**Cost per Participant**: ${cost_per_participant:,.0f}")
        protocol.append("")
        protocol.append("### Budget Breakdown")
        protocol.append("- Personnel (40%): ${:,.0f}".format(design.estimated_cost * 0.4))
        protocol.append("- Technology/App Development (25%): ${:,.0f}".format(design.estimated_cost * 0.25))
        protocol.append("- Data Collection/Management (20%): ${:,.0f}".format(design.estimated_cost * 0.2))
        protocol.append("- Regulatory/Administrative (10%): ${:,.0f}".format(design.estimated_cost * 0.1))
        protocol.append("- Contingency (5%): ${:,.0f}".format(design.estimated_cost * 0.05))
        
        return '\n'.join(protocol)
    
    def generate_optimization_report(self, 
                                   optimized_designs: List[StudyDesign],
                                   constraints: Dict[str, Any]) -> str:
        """Generate experimental design optimization report"""
        
        report = []
        report.append("# Experimental Design Optimization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive summary
        report.append("## Executive Summary")
        report.append(f"**Available Budget**: ${constraints.get('budget', 0):,.0f}")
        report.append(f"**Time Constraint**: {constraints.get('time_months', 0)} months")
        report.append(f"**Target Conditions**: {', '.join(constraints.get('conditions', []))}")
        report.append(f"**Designs Generated**: {len(optimized_designs)}")
        report.append("")
        
        if optimized_designs:
            top_design = optimized_designs[0]
            report.append(f"**Recommended Design**: {top_design.name}")
            report.append(f"**Estimated Cost**: ${top_design.estimated_cost:,.0f}")
            report.append(f"**Duration**: {top_design.duration_weeks} weeks")
            report.append(f"**Sample Size**: {top_design.total_sample_size}")
            report.append(f"**Statistical Power**: {top_design.power}")
        
        report.append("")
        
        # Detailed design comparisons
        report.append("## Design Comparisons")
        report.append("| Rank | Design | Type | Sample Size | Duration (weeks) | Cost | Power | Feasibility |")
        report.append("|------|--------|------|-------------|------------------|------|-------|-------------|")
        
        for i, design in enumerate(optimized_designs[:5], 1):
            report.append(f"| {i} | {design.name[:30]}... | {design.study_type.value} | "
                         f"{design.total_sample_size} | {design.duration_weeks} | "
                         f"${design.estimated_cost:,.0f} | {design.power:.2f} | {design.feasibility_score:.2f} |")
        
        report.append("")
        
        # Detailed recommendations
        report.append("## Detailed Recommendations")
        
        for i, design in enumerate(optimized_designs[:3], 1):
            report.append(f"### Option {i}: {design.name}")
            report.append(f"**Study Type**: {design.study_type.value.replace('_', ' ').title()}")
            report.append(f"**Rationale**: Optimized for {design.primary_endpoint} with {design.power} power")
            report.append("")
            
            report.append("**Strengths**:")
            if design.power >= 0.8:
                report.append("- Adequate statistical power")
            if design.feasibility_score >= 0.7:
                report.append("- High feasibility score")
            if design.estimated_cost <= constraints.get('budget', float('inf')):
                report.append("- Within budget constraints")
            
            report.append("")
            report.append("**Key Parameters**:")
            report.append(f"- Sample size: {design.total_sample_size}")
            report.append(f"- Primary endpoint: {design.primary_endpoint}")
            report.append(f"- Follow-up duration: {design.duration_weeks} weeks")
            report.append(f"- Estimated cost: ${design.estimated_cost:,.0f}")
            report.append("")
            
            report.append("**Implementation Considerations**:")
            if design.study_type == StudyType.ADAPTIVE:
                report.append("- Requires adaptive trial expertise")
                report.append("- Complex statistical monitoring")
            elif design.study_type == StudyType.CROSSOVER:
                report.append("- Requires washout period optimization")
                report.append("- Participant burden considerations")
            elif design.study_type == StudyType.FACTORIAL:
                report.append("- Multiple intervention components")
                report.append("- Complex randomization scheme")
            
            report.append("")
        
        # Implementation timeline
        report.append("## Implementation Timeline")
        report.append("### Phase 1: Protocol Development (Months 1-3)")
        report.append("- Finalize study protocol")
        report.append("- Develop digital intervention")
        report.append("- Prepare regulatory submissions")
        report.append("")
        
        report.append("### Phase 2: Regulatory and Setup (Months 4-6)")
        report.append("- Obtain regulatory approvals")
        report.append("- Site initiation and training")
        report.append("- System testing and validation")
        report.append("")
        
        report.append("### Phase 3: Enrollment and Conduct (Months 7-24)")
        report.append("- Participant recruitment and enrollment")
        report.append("- Study conduct and monitoring")
        report.append("- Interim analyses (if applicable)")
        report.append("")
        
        report.append("### Phase 4: Analysis and Reporting (Months 25-30)")
        report.append("- Data analysis and interpretation")
        report.append("- Report preparation")
        report.append("- Regulatory submissions")
        report.append("")
        
        # Risk assessment
        report.append("## Risk Assessment and Mitigation")
        report.append("### High Risk Factors")
        report.append("- **Recruitment challenges**: Competitive landscape for digital health studies")
        report.append("- **Technology issues**: App performance and user experience")
        report.append("- **Retention**: Long-term follow-up in digital studies")
        report.append("")
        
        report.append("### Mitigation Strategies")
        report.append("- **Recruitment**: Multi-site approach, social media recruitment")
        report.append("- **Technology**: Extensive pilot testing, user feedback integration")
        report.append("- **Retention**: Engagement features, regular contact, incentives")
        report.append("")
        
        # Next steps
        report.append("## Recommended Next Steps")
        report.append("1. **Select optimal design** based on strategic priorities")
        report.append("2. **Secure funding** for selected study design")
        report.append("3. **Assemble study team** with appropriate expertise")
        report.append("4. **Develop detailed protocol** with regulatory input")
        report.append("5. **Initiate regulatory discussions** early in planning")
        report.append("6. **Plan pilot/feasibility study** if needed")
        
        return '\n'.join(report)

# Usage function
def optimize_experimental_design(target_conditions: List[str],
                               available_budget: float,
                               time_constraints: int,
                               priority_hypotheses: List[str]) -> Dict[str, Any]:
    """Main function to optimize experimental design"""
    
    optimizer = ExperimentalDesignOptimizer()
    
    # Generate optimized designs
    optimized_designs = optimizer.optimize_study_design(
        target_conditions=target_conditions,
        available_budget=available_budget,
        time_constraints=time_constraints,
        priority_hypotheses=priority_hypotheses
    )
    
    # Generate detailed report
    constraints = {
        'budget': available_budget,
        'time_months': time_constraints,
        'conditions': target_conditions,
        'hypotheses': priority_hypotheses
    }
    
    optimization_report = optimizer.generate_optimization_report(optimized_designs, constraints)
    
    # Generate protocol for top design
    protocol_template = ""
    if optimized_designs:
        protocol_template = optimizer.generate_protocol_template(optimized_designs[0])
    
    return {
        'optimized_designs': optimized_designs,
        'optimization_report': optimization_report,
        'protocol_template': protocol_template,
        'optimizer': optimizer
    }