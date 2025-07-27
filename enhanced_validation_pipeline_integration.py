#!/usr/bin/env python3
"""
Enhanced Validation Pipeline Integration
Comprehensive framework integrating all hypothesis testing improvements for PlaceboRx validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import enhanced modules
from hypothesis_testing_framework import (
    AdvancedHypothesisTestingFramework,
    run_enhanced_hypothesis_testing,
    HypothesisType,
    TestResult
)
from experimental_design_optimizer import (
    ExperimentalDesignOptimizer,
    optimize_experimental_design,
    StudyDesign,
    StudyType
)
from real_world_evidence_engine import (
    RealWorldEvidenceEngine,
    setup_rwe_tracking,
    simulate_rwe_data,
    OutcomeType,
    DataSource
)

# Import existing modules
from clinical_trials_analyzer import ClinicalTrialsAnalyzer
from market_analyzer import MarketAnalyzer
from enhanced_config import CONFIG, AnalysisMode, ValidationLevel
from data_quality import DataQualityValidator, ValidationResult
from ml_enhancement import MLEnhancementEngine
from visualization_engine import VisualizationEngine

class ValidationStage(Enum):
    """Stages of enhanced validation pipeline"""
    PRELIMINARY_ANALYSIS = "preliminary_analysis"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    EXPERIMENTAL_DESIGN = "experimental_design"
    REAL_WORLD_EVIDENCE = "real_world_evidence"
    COMPREHENSIVE_REPORTING = "comprehensive_reporting"

@dataclass
class EnhancedValidationResults:
    """Complete results from enhanced validation pipeline"""
    # Existing analysis results
    clinical_data: pd.DataFrame
    market_data: pd.DataFrame
    clinical_validation: ValidationResult
    market_validation: ValidationResult
    ml_insights: Dict[str, Any]
    
    # Enhanced analysis results
    hypothesis_test_results: Dict[str, TestResult]
    experimental_designs: List[StudyDesign]
    rwe_analysis: Dict[str, Any]
    
    # Quality and execution metrics
    execution_time: float
    quality_scores: Dict[str, float]
    validation_confidence: float
    recommendation_strength: str
    
    # Reports
    hypothesis_testing_report: str
    experimental_design_report: str
    rwe_report: str
    integrated_executive_summary: str

class ComprehensiveValidationPipeline:
    """Enhanced validation pipeline with comprehensive hypothesis testing capabilities"""
    
    def __init__(self, output_dir: str = "enhanced_validation_output"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.config = CONFIG
        self.data_validator = DataQualityValidator(self.config.quality.validation_level)
        self.ml_engine = MLEnhancementEngine()
        self.viz_engine = VisualizationEngine()
        self.hypothesis_framework = AdvancedHypothesisTestingFramework()
        self.design_optimizer = ExperimentalDesignOptimizer()
        
        # Initialize RWE engine if configured
        self.rwe_engine = None
        if self.config.output.enable_rwe_tracking:
            rwe_db_path = self.output_dir / "real_world_evidence.db"
            self.rwe_engine = RealWorldEvidenceEngine(str(rwe_db_path))
        
        self.results = EnhancedValidationResults(
            clinical_data=pd.DataFrame(),
            market_data=pd.DataFrame(),
            clinical_validation=None,
            market_validation=None,
            ml_insights={},
            hypothesis_test_results={},
            experimental_designs=[],
            rwe_analysis={},
            execution_time=0,
            quality_scores={},
            validation_confidence=0.0,
            recommendation_strength="Unknown",
            hypothesis_testing_report="",
            experimental_design_report="",
            rwe_report="",
            integrated_executive_summary=""
        )
    
    def run_comprehensive_validation(self) -> EnhancedValidationResults:
        """Run complete enhanced validation pipeline"""
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting comprehensive PlaceboRx validation pipeline...")
            
            # Stage 1: Preliminary Analysis (existing functionality)
            self.logger.info("Stage 1: Running preliminary analysis...")
            self._run_preliminary_analysis()
            
            # Stage 2: Advanced Hypothesis Testing
            self.logger.info("Stage 2: Running advanced hypothesis testing...")
            self._run_advanced_hypothesis_testing()
            
            # Stage 3: Experimental Design Optimization
            self.logger.info("Stage 3: Optimizing experimental designs...")
            self._optimize_experimental_designs()
            
            # Stage 4: Real-World Evidence Analysis
            self.logger.info("Stage 4: Analyzing real-world evidence...")
            self._analyze_real_world_evidence()
            
            # Stage 5: Comprehensive Reporting
            self.logger.info("Stage 5: Generating comprehensive reports...")
            self._generate_comprehensive_reports()
            
            # Calculate overall validation confidence
            self._calculate_validation_confidence()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.results.execution_time = execution_time
            
            self.logger.info(f"Comprehensive validation completed in {execution_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            return self.results
    
    def _run_preliminary_analysis(self):
        """Run preliminary clinical and market analysis"""
        try:
            # Clinical trials analysis
            clinical_analyzer = ClinicalTrialsAnalyzer(
                search_terms=self.config.clinical.search_terms,
                target_conditions=self.config.clinical.target_conditions
            )
            
            clinical_results = clinical_analyzer.search_trials()
            
            if clinical_results:
                self.results.clinical_data = pd.DataFrame(clinical_results)
                
                # Apply ML enhancements
                self.results.clinical_data = self.ml_engine.enhance_clinical_analysis(
                    self.results.clinical_data
                )
                
                # Validate data quality
                self.results.clinical_validation = self.data_validator.validate_clinical_data(
                    self.results.clinical_data
                )
                
                self.logger.info(f"Clinical analysis: {len(self.results.clinical_data)} trials analyzed")
            
            # Market analysis
            market_analyzer = MarketAnalyzer(
                subreddits=self.config.market.subreddits,
                keywords=self.config.market.desperation_keywords + self.config.market.openness_keywords
            )
            
            market_results = market_analyzer.analyze_market_sentiment()
            
            if market_results:
                self.results.market_data = pd.DataFrame(market_results)
                
                # Apply ML enhancements
                self.results.market_data = self.ml_engine.enhance_market_analysis(
                    self.results.market_data
                )
                
                # Validate data quality
                self.results.market_validation = self.data_validator.validate_market_data(
                    self.results.market_data
                )
                
                self.logger.info(f"Market analysis: {len(self.results.market_data)} posts analyzed")
            
            # Generate ML insights
            self.results.ml_insights = self.ml_engine.generate_ml_insights_report(
                self.results.clinical_data,
                self.results.market_data
            )
            
        except Exception as e:
            self.logger.error(f"Error in preliminary analysis: {e}")
    
    def _run_advanced_hypothesis_testing(self):
        """Run comprehensive hypothesis testing framework"""
        try:
            # Run all hypothesis tests
            test_results = self.hypothesis_framework.run_comprehensive_hypothesis_testing(
                self.results.clinical_data,
                self.results.market_data
            )
            
            self.results.hypothesis_test_results = test_results
            
            # Generate hypothesis testing report
            self.results.hypothesis_testing_report = self.hypothesis_framework.generate_hypothesis_testing_report(
                test_results
            )
            
            # Calculate quality scores for hypothesis testing
            self.results.quality_scores['hypothesis_testing'] = self._calculate_hypothesis_testing_quality()
            
            self.logger.info(f"Hypothesis testing: {len(test_results)} tests completed")
            
        except Exception as e:
            self.logger.error(f"Error in hypothesis testing: {e}")
    
    def _optimize_experimental_designs(self):
        """Optimize experimental designs for future studies"""
        try:
            # Define constraints based on config and findings
            available_budget = getattr(self.config, 'available_budget', 1000000)  # Default $1M
            time_constraints = getattr(self.config, 'time_constraints', 24)  # Default 24 months
            
            # Determine priority hypotheses based on current findings
            priority_hypotheses = self._determine_priority_hypotheses()
            
            # Optimize experimental designs
            design_results = optimize_experimental_design(
                target_conditions=self.config.clinical.target_conditions[:3],  # Top 3 conditions
                available_budget=available_budget,
                time_constraints=time_constraints,
                priority_hypotheses=priority_hypotheses
            )
            
            self.results.experimental_designs = design_results['optimized_designs']
            self.results.experimental_design_report = design_results['optimization_report']
            
            # Calculate quality scores for experimental design
            self.results.quality_scores['experimental_design'] = self._calculate_design_quality()
            
            self.logger.info(f"Experimental design: {len(self.results.experimental_designs)} designs optimized")
            
        except Exception as e:
            self.logger.error(f"Error in experimental design optimization: {e}")
    
    def _analyze_real_world_evidence(self):
        """Analyze real-world evidence (if available)"""
        try:
            rwe_analysis = {}
            
            if self.rwe_engine:
                # Generate effectiveness dashboard data
                dashboard_data = self.rwe_engine.generate_effectiveness_dashboard_data()
                rwe_analysis['dashboard_data'] = dashboard_data
                
                # Analyze longitudinal outcomes (if patients enrolled)
                longitudinal_outcomes = self.rwe_engine.analyze_longitudinal_outcomes(days_lookback=90)
                rwe_analysis['longitudinal_outcomes'] = len(longitudinal_outcomes)
                
                # Monitor safety signals
                safety_signals = self.rwe_engine.monitor_real_time_safety(lookback_hours=168)  # 1 week
                rwe_analysis['safety_signals'] = safety_signals
                
                # Generate RWE report
                self.results.rwe_report = self.rwe_engine.generate_rwe_report(days_lookback=90)
                
                # Calculate quality scores for RWE
                self.results.quality_scores['real_world_evidence'] = self._calculate_rwe_quality(dashboard_data)
                
                self.logger.info("Real-world evidence analysis completed")
            else:
                # Simulate RWE analysis for demonstration
                rwe_analysis = self._simulate_rwe_analysis()
                self.results.rwe_report = "Real-world evidence tracking not configured. Consider implementing RWE collection for enhanced validation."
                self.results.quality_scores['real_world_evidence'] = 0.0
                
                self.logger.info("RWE analysis simulated (not configured)")
            
            self.results.rwe_analysis = rwe_analysis
            
        except Exception as e:
            self.logger.error(f"Error in real-world evidence analysis: {e}")
            self.results.rwe_analysis = {'error': str(e)}
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive integrated reports"""
        try:
            # Generate visualizations
            if not self.results.clinical_data.empty:
                clinical_dashboard = self.viz_engine.create_clinical_dashboard(
                    self.results.clinical_data,
                    str(self.output_dir / "clinical_dashboard.html")
                )
            
            if not self.results.market_data.empty:
                market_dashboard = self.viz_engine.create_market_dashboard(
                    self.results.market_data,
                    str(self.output_dir / "market_dashboard.html")
                )
            
            # Create comparative analysis
            if not self.results.clinical_data.empty and not self.results.market_data.empty:
                comparative_dashboard = self.viz_engine.create_comparative_analysis(
                    self.results.clinical_data,
                    self.results.market_data,
                    str(self.output_dir / "comparative_dashboard.html")
                )
            
            # Generate integrated executive summary
            self.results.integrated_executive_summary = self._generate_integrated_executive_summary()
            
            # Save all reports to files
            self._save_reports_to_files()
            
            self.logger.info("Comprehensive reports generated")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive reports: {e}")
    
    def _calculate_validation_confidence(self):
        """Calculate overall validation confidence score"""
        try:
            confidence_factors = []
            
            # Data quality factors (30%)
            if self.results.clinical_validation:
                confidence_factors.append(('clinical_data_quality', self.results.clinical_validation.quality_score, 0.15))
            
            if self.results.market_validation:
                confidence_factors.append(('market_data_quality', self.results.market_validation.quality_score, 0.15))
            
            # Hypothesis testing factors (40%)
            significant_tests = sum(1 for result in self.results.hypothesis_test_results.values() 
                                  if isinstance(result, TestResult) and result.statistical_significance)
            total_tests = len([r for r in self.results.hypothesis_test_results.values() if isinstance(r, TestResult)])
            
            if total_tests > 0:
                hypothesis_score = significant_tests / total_tests
                confidence_factors.append(('hypothesis_testing', hypothesis_score, 0.40))
            
            # Sample sizes and power (20%)
            avg_power = 0.8  # Default assumption
            if self.results.hypothesis_test_results:
                powers = [r.power for r in self.results.hypothesis_test_results.values() 
                         if isinstance(r, TestResult) and r.power is not None]
                if powers:
                    avg_power = np.mean(powers)
            
            confidence_factors.append(('statistical_power', avg_power, 0.20))
            
            # Real-world evidence (10%)
            rwe_score = self.results.quality_scores.get('real_world_evidence', 0.0)
            confidence_factors.append(('real_world_evidence', rwe_score, 0.10))
            
            # Calculate weighted confidence
            total_confidence = sum(score * weight for _, score, weight in confidence_factors)
            self.results.validation_confidence = min(1.0, max(0.0, total_confidence))
            
            # Determine recommendation strength
            if self.results.validation_confidence >= 0.8:
                self.results.recommendation_strength = "Strong"
            elif self.results.validation_confidence >= 0.6:
                self.results.recommendation_strength = "Moderate"
            elif self.results.validation_confidence >= 0.4:
                self.results.recommendation_strength = "Weak"
            else:
                self.results.recommendation_strength = "Insufficient"
            
            self.logger.info(f"Validation confidence: {self.results.validation_confidence:.3f} ({self.results.recommendation_strength})")
            
        except Exception as e:
            self.logger.error(f"Error calculating validation confidence: {e}")
            self.results.validation_confidence = 0.0
            self.results.recommendation_strength = "Error"
    
    def _determine_priority_hypotheses(self) -> List[str]:
        """Determine priority hypotheses based on current findings"""
        priorities = []
        
        # Always include core efficacy testing
        priorities.append("efficacy")
        
        # Add other hypotheses based on data availability and findings
        if not self.results.clinical_data.empty:
            # Check if we have multiple conditions for specificity testing
            if len(self.results.clinical_data.get('condition', pd.Series()).unique()) >= 3:
                priorities.append("condition_specificity")
            
            # Check for temporal data
            if 'completion_date' in self.results.clinical_data.columns:
                priorities.append("temporal_sustainability")
        
        if not self.results.market_data.empty:
            # Check for engagement data for dose-response
            if 'predicted_engagement' in self.results.market_data.columns:
                priorities.append("dose_response")
        
        return priorities[:4]  # Limit to top 4 priorities
    
    def _calculate_hypothesis_testing_quality(self) -> float:
        """Calculate quality score for hypothesis testing"""
        try:
            if not self.results.hypothesis_test_results:
                return 0.0
            
            quality_factors = []
            
            for result in self.results.hypothesis_test_results.values():
                if isinstance(result, TestResult):
                    # Statistical power
                    power_score = result.power if result.power else 0.8
                    quality_factors.append(power_score)
                    
                    # Sample size adequacy
                    sample_adequacy = min(1.0, result.sample_size / 100)  # 100 as minimum adequate
                    quality_factors.append(sample_adequacy)
                    
                    # Evidence strength
                    strength_scores = {'Strong': 1.0, 'Moderate': 0.7, 'Weak': 0.4, 'Insufficient': 0.1}
                    evidence_score = strength_scores.get(result.evidence_strength, 0.1)
                    quality_factors.append(evidence_score)
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating hypothesis testing quality: {e}")
            return 0.0
    
    def _calculate_design_quality(self) -> float:
        """Calculate quality score for experimental designs"""
        try:
            if not self.results.experimental_designs:
                return 0.0
            
            quality_scores = []
            for design in self.results.experimental_designs:
                # Statistical power
                power_score = design.power
                
                # Feasibility
                feasibility_score = design.feasibility_score
                
                # Design appropriateness (higher for RCT, adaptive)
                design_scores = {
                    StudyType.RCT: 1.0,
                    StudyType.ADAPTIVE: 0.9,
                    StudyType.CROSSOVER: 0.8,
                    StudyType.FACTORIAL: 0.8,
                    StudyType.DOSE_FINDING: 0.7,
                    StudyType.OBSERVATIONAL: 0.6,
                    StudyType.PILOT: 0.5
                }
                design_score = design_scores.get(design.study_type, 0.5)
                
                # Combined score
                combined_score = (power_score + feasibility_score + design_score) / 3
                quality_scores.append(combined_score)
            
            return np.mean(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating design quality: {e}")
            return 0.0
    
    def _calculate_rwe_quality(self, dashboard_data: Dict[str, Any]) -> float:
        """Calculate quality score for real-world evidence"""
        try:
            quality_factors = []
            
            # Patient enrollment and activity
            total_patients = dashboard_data.get('total_patients', 0)
            active_patients = dashboard_data.get('active_patients', 0)
            
            if total_patients > 0:
                activity_rate = active_patients / total_patients
                quality_factors.append(activity_rate)
            
            # Data quality metrics
            data_quality = dashboard_data.get('data_quality', [])
            if data_quality:
                avg_confidence = np.mean([dq['avg_confidence'] for dq in data_quality])
                quality_factors.append(avg_confidence)
            
            # Data completeness (mock score)
            quality_factors.append(0.8)  # Placeholder
            
            return np.mean(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating RWE quality: {e}")
            return 0.0
    
    def _simulate_rwe_analysis(self) -> Dict[str, Any]:
        """Simulate RWE analysis for demonstration purposes"""
        return {
            'simulated': True,
            'total_patients': 0,
            'active_patients': 0,
            'longitudinal_outcomes': 0,
            'safety_signals': {
                'total_events': 0,
                'unique_patients': 0,
                'alerts': []
            }
        }
    
    def _generate_integrated_executive_summary(self) -> str:
        """Generate comprehensive integrated executive summary"""
        summary = []
        summary.append("# Comprehensive PlaceboRx Validation Report")
        summary.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"**Analysis Mode**: {self.config.analysis_mode.value}")
        summary.append(f"**Validation Level**: {self.config.quality.validation_level.value}")
        summary.append("")
        
        # Executive Summary
        summary.append("## Executive Summary")
        summary.append(f"**Overall Validation Confidence**: {self.results.validation_confidence:.1%}")
        summary.append(f"**Recommendation Strength**: {self.results.recommendation_strength}")
        summary.append(f"**Execution Time**: {self.results.execution_time:.1f} seconds")
        summary.append("")
        
        # Key Findings
        summary.append("## Key Findings")
        
        # Clinical analysis findings
        if not self.results.clinical_data.empty:
            summary.append(f"### Clinical Evidence")
            summary.append(f"- **Trials Analyzed**: {len(self.results.clinical_data)}")
            
            if 'predicted_success_probability' in self.results.clinical_data.columns:
                avg_success = self.results.clinical_data['predicted_success_probability'].mean()
                summary.append(f"- **Average Success Probability**: {avg_success:.1%}")
            
            if self.results.clinical_validation:
                summary.append(f"- **Data Quality Score**: {self.results.clinical_validation.quality_score:.2f}")
        
        # Market analysis findings
        if not self.results.market_data.empty:
            summary.append(f"### Market Evidence")
            summary.append(f"- **Posts Analyzed**: {len(self.results.market_data)}")
            
            if 'desperation_intensity' in self.results.market_data.columns:
                avg_desperation = self.results.market_data['desperation_intensity'].mean()
                summary.append(f"- **Average Desperation Level**: {avg_desperation:.2f}/3")
            
            if 'openness_level' in self.results.market_data.columns:
                avg_openness = self.results.market_data['openness_level'].mean()
                summary.append(f"- **Average Openness Level**: {avg_openness:.2f}/3")
        
        summary.append("")
        
        # Hypothesis Testing Results
        summary.append("## Hypothesis Testing Results")
        
        if self.results.hypothesis_test_results:
            significant_tests = sum(1 for result in self.results.hypothesis_test_results.values() 
                                  if isinstance(result, TestResult) and result.statistical_significance)
            practical_tests = sum(1 for result in self.results.hypothesis_test_results.values() 
                                if isinstance(result, TestResult) and result.practical_significance)
            total_tests = len([r for r in self.results.hypothesis_test_results.values() if isinstance(r, TestResult)])
            
            summary.append(f"- **Total Hypotheses Tested**: {total_tests}")
            summary.append(f"- **Statistically Significant**: {significant_tests}/{total_tests}")
            summary.append(f"- **Practically Significant**: {practical_tests}/{total_tests}")
            
            # Highlight key test results
            for test_name, result in self.results.hypothesis_test_results.items():
                if isinstance(result, TestResult) and result.statistical_significance:
                    summary.append(f"- **{result.test_name}**: Effect size {result.effect_size:.3f}, p = {result.p_value:.3f}")
        
        summary.append("")
        
        # Experimental Design Recommendations
        summary.append("## Experimental Design Recommendations")
        
        if self.results.experimental_designs:
            top_design = self.results.experimental_designs[0]
            summary.append(f"### Recommended Study Design")
            summary.append(f"- **Study Type**: {top_design.study_type.value.replace('_', ' ').title()}")
            summary.append(f"- **Sample Size**: {top_design.total_sample_size}")
            summary.append(f"- **Duration**: {top_design.duration_weeks} weeks")
            summary.append(f"- **Estimated Cost**: ${top_design.estimated_cost:,.0f}")
            summary.append(f"- **Statistical Power**: {top_design.power:.1%}")
            summary.append(f"- **Feasibility Score**: {top_design.feasibility_score:.2f}")
            
            summary.append(f"### Alternative Designs")
            for i, design in enumerate(self.results.experimental_designs[1:4], 2):
                summary.append(f"- **Option {i}**: {design.study_type.value.replace('_', ' ').title()} "
                             f"(N={design.total_sample_size}, ${design.estimated_cost:,.0f})")
        
        summary.append("")
        
        # Real-World Evidence
        summary.append("## Real-World Evidence")
        
        if self.results.rwe_analysis and not self.results.rwe_analysis.get('simulated', False):
            rwe_data = self.results.rwe_analysis.get('dashboard_data', {})
            summary.append(f"- **Enrolled Patients**: {rwe_data.get('total_patients', 0)}")
            summary.append(f"- **Active Patients**: {rwe_data.get('active_patients', 0)}")
            
            safety_signals = self.results.rwe_analysis.get('safety_signals', {})
            summary.append(f"- **Adverse Events**: {safety_signals.get('total_events', 0)}")
            
            if safety_signals.get('alerts'):
                summary.append("- **Safety Alerts**: Yes")
            else:
                summary.append("- **Safety Alerts**: None")
        else:
            summary.append("- **Status**: Not yet configured (recommended for post-launch validation)")
            summary.append("- **Recommendation**: Implement RWE tracking for continuous validation")
        
        summary.append("")
        
        # Overall Assessment
        summary.append("## Overall Assessment")
        
        if self.results.validation_confidence >= 0.8:
            summary.append("✅ **STRONG EVIDENCE** supports the PlaceboRx hypothesis")
            summary.append("**Recommendation**: Proceed with confidence to next development phase")
        elif self.results.validation_confidence >= 0.6:
            summary.append("✅ **MODERATE EVIDENCE** supports the PlaceboRx hypothesis")
            summary.append("**Recommendation**: Proceed with additional validation studies")
        elif self.results.validation_confidence >= 0.4:
            summary.append("⚠️ **WEAK EVIDENCE** for the PlaceboRx hypothesis")
            summary.append("**Recommendation**: Conduct pilot studies before major investment")
        else:
            summary.append("❌ **INSUFFICIENT EVIDENCE** for the PlaceboRx hypothesis")
            summary.append("**Recommendation**: Reassess concept or gather more data")
        
        summary.append("")
        
        # Strategic Recommendations
        summary.append("## Strategic Recommendations")
        
        recommendations = []
        
        # Based on validation confidence
        if self.results.validation_confidence >= 0.6:
            recommendations.append("**Advance to clinical development** with recommended study design")
            recommendations.append("**Secure funding** for clinical trials")
        
        # Based on hypothesis testing
        if self.results.hypothesis_test_results:
            significant_count = sum(1 for r in self.results.hypothesis_test_results.values() 
                                  if isinstance(r, TestResult) and r.statistical_significance)
            if significant_count >= 2:
                recommendations.append("**Focus on validated hypotheses** in product development")
        
        # Based on experimental design
        if self.results.experimental_designs:
            recommendations.append("**Implement top-ranked study design** for definitive validation")
            recommendations.append("**Consider adaptive trial design** for efficiency")
        
        # RWE recommendations
        if not self.results.rwe_analysis.get('simulated', True):
            recommendations.append("**Continue real-world evidence collection** for ongoing validation")
        else:
            recommendations.append("**Implement real-world evidence framework** for post-launch monitoring")
        
        # General recommendations
        recommendations.extend([
            "**Establish regulatory pathway** early in development",
            "**Build multidisciplinary team** with digital health expertise",
            "**Plan for publication** of validation results",
            "**Consider partnership opportunities** with clinical institutions"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            summary.append(f"{i}. {rec}")
        
        summary.append("")
        
        # Next Steps
        summary.append("## Immediate Next Steps")
        summary.append("1. **Review detailed analysis reports** in generated output files")
        summary.append("2. **Present findings** to stakeholders and decision makers")
        summary.append("3. **Secure resources** for recommended next phase")
        summary.append("4. **Initiate regulatory discussions** if moving forward")
        summary.append("5. **Plan implementation timeline** for selected study design")
        
        return '\n'.join(summary)
    
    def _save_reports_to_files(self):
        """Save all reports to output files"""
        try:
            # Save hypothesis testing report
            if self.results.hypothesis_testing_report:
                with open(self.output_dir / "hypothesis_testing_report.md", 'w') as f:
                    f.write(self.results.hypothesis_testing_report)
            
            # Save experimental design report
            if self.results.experimental_design_report:
                with open(self.output_dir / "experimental_design_report.md", 'w') as f:
                    f.write(self.results.experimental_design_report)
            
            # Save RWE report
            if self.results.rwe_report:
                with open(self.output_dir / "real_world_evidence_report.md", 'w') as f:
                    f.write(self.results.rwe_report)
            
            # Save integrated executive summary
            if self.results.integrated_executive_summary:
                with open(self.output_dir / "integrated_executive_summary.md", 'w') as f:
                    f.write(self.results.integrated_executive_summary)
            
            # Save detailed results as JSON
            results_summary = {
                'validation_confidence': self.results.validation_confidence,
                'recommendation_strength': self.results.recommendation_strength,
                'execution_time': self.results.execution_time,
                'quality_scores': self.results.quality_scores,
                'data_summary': {
                    'clinical_trials': len(self.results.clinical_data),
                    'market_posts': len(self.results.market_data),
                    'hypothesis_tests': len(self.results.hypothesis_test_results),
                    'experimental_designs': len(self.results.experimental_designs)
                },
                'generated_timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / "validation_summary.json", 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            self.logger.info(f"All reports saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving reports: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get concise validation summary for programmatic access"""
        return {
            'overall_confidence': self.results.validation_confidence,
            'recommendation_strength': self.results.recommendation_strength,
            'key_metrics': {
                'clinical_trials_analyzed': len(self.results.clinical_data),
                'market_posts_analyzed': len(self.results.market_data),
                'hypothesis_tests_significant': sum(1 for r in self.results.hypothesis_test_results.values() 
                                                  if isinstance(r, TestResult) and r.statistical_significance),
                'top_study_design': self.results.experimental_designs[0].name if self.results.experimental_designs else None,
                'estimated_study_cost': self.results.experimental_designs[0].estimated_cost if self.results.experimental_designs else None
            },
            'quality_scores': self.results.quality_scores,
            'execution_time': self.results.execution_time,
            'output_directory': str(self.output_dir)
        }

# Main execution function
def run_comprehensive_validation(output_dir: str = "enhanced_validation_output") -> EnhancedValidationResults:
    """
    Run comprehensive PlaceboRx validation with all enhancements
    
    Args:
        output_dir: Directory to save validation outputs
        
    Returns:
        EnhancedValidationResults containing all analysis results
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/validation.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive PlaceboRx validation...")
    
    try:
        # Initialize and run validation pipeline
        pipeline = ComprehensiveValidationPipeline(output_dir)
        results = pipeline.run_comprehensive_validation()
        
        # Print summary
        summary = pipeline.get_validation_summary()
        logger.info("=== VALIDATION SUMMARY ===")
        logger.info(f"Overall Confidence: {summary['overall_confidence']:.1%}")
        logger.info(f"Recommendation: {summary['recommendation_strength']}")
        logger.info(f"Clinical Trials: {summary['key_metrics']['clinical_trials_analyzed']}")
        logger.info(f"Market Posts: {summary['key_metrics']['market_posts_analyzed']}")
        logger.info(f"Significant Tests: {summary['key_metrics']['hypothesis_tests_significant']}")
        logger.info(f"Output Directory: {summary['output_directory']}")
        logger.info("=== END SUMMARY ===")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive validation: {e}")
        raise

if __name__ == "__main__":
    # Run comprehensive validation
    results = run_comprehensive_validation()
    print("Comprehensive validation completed!")
    print(f"Check output directory for detailed reports: enhanced_validation_output/")