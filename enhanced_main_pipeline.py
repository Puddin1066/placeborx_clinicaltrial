#!/usr/bin/env python3
"""
Enhanced PlaceboRx Validation Pipeline
Solo entrepreneur execution with advanced analytics and controls
"""

import time
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Import existing components
from clinical_trials_analyzer import ClinicalTrialsAnalyzer
from market_analyzer import MarketAnalyzer
from pubmed_analyzer import PubMedAnalyzer
from openai_processor import OpenAIProcessor

# Import new enhanced components
from enhanced_config import CONFIG, AnalysisMode, ValidationLevel
from data_quality import DataQualityValidator, ValidationResult
from ml_enhancement import MLEnhancementEngine
from visualization_engine import VisualizationEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class EnhancedValidationPipeline:
    """Enhanced validation pipeline with advanced analytics and controls"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = CONFIG
        
        # Initialize components
        self.data_validator = DataQualityValidator(self.config.quality.validation_level)
        self.ml_engine = MLEnhancementEngine()
        self.viz_engine = VisualizationEngine()
        self.pubmed_analyzer = PubMedAnalyzer()
        self.openai_processor = OpenAIProcessor()
        
        # Results storage
        self.results = {
            'clinical_data': pd.DataFrame(),
            'market_data': pd.DataFrame(),
            'pubmed_data': {},
            'openai_insights': {},
            'clinical_validation': None,
            'market_validation': None,
            'pubmed_validation': None,
            'ml_insights': {},
            'execution_time': 0,
            'quality_scores': {}
        }
        
    def validate_environment(self) -> bool:
        """Validate environment and configuration"""
        self.logger.info("🔍 Validating environment and configuration...")
        
        # Validate configuration
        config_issues = self.config.validate()
        if config_issues:
            self.logger.error("Configuration validation failed:")
            for issue in config_issues:
                self.logger.error(f"  - {issue}")
            return False
        
        # Check required directories
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        self.logger.info("✅ Environment validation passed")
        return True
    
    def run_enhanced_clinical_analysis(self) -> ValidationResult:
        """Run enhanced clinical trials analysis with ML and quality validation"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🔬 ENHANCED CLINICAL TRIALS ANALYSIS")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Step 1: Basic clinical analysis
            analyzer = ClinicalTrialsAnalyzer()
            clinical_df = analyzer.analyze_trials()
            
            self.logger.info(f"Retrieved {len(clinical_df)} clinical trials")
            
            # Step 2: Data quality validation
            validation_result = self.data_validator.validate_clinical_data(clinical_df)
            
            if validation_result.is_valid:
                self.logger.info(f"✅ Clinical data validation passed (Quality Score: {validation_result.quality_score:.2f})")
                clinical_df = validation_result.cleaned_data
            else:
                self.logger.warning(f"⚠️ Clinical data validation issues found (Quality Score: {validation_result.quality_score:.2f})")
                for issue in validation_result.issues:
                    self.logger.warning(f"  - {issue}")
                
                # Use cleaned data even if validation failed
                if validation_result.cleaned_data is not None:
                    clinical_df = validation_result.cleaned_data
            
            # Step 3: ML Enhancement
            if self.config.enable_ml_enhancement and not clinical_df.empty:
                self.logger.info("🤖 Applying ML enhancements...")
                clinical_df = self.ml_engine.enhance_clinical_analysis(clinical_df)
                self.logger.info("✅ ML enhancement completed")
            
            # Step 4: Save enhanced data
            clinical_df.to_csv(f"{self.config.output.output_dir}/enhanced_clinical_trials.csv", index=False)
            
            # Store results
            self.results['clinical_data'] = clinical_df
            self.results['clinical_validation'] = validation_result
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"⏱️ Enhanced clinical analysis completed in {elapsed_time:.1f} seconds")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced clinical analysis failed: {e}")
            raise
    
    def run_enhanced_market_analysis(self) -> ValidationResult:
        """Run enhanced market analysis with ML and quality validation"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 ENHANCED MARKET ANALYSIS")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Step 1: Basic market analysis
            analyzer = MarketAnalyzer()
            market_df = analyzer.run_analysis()
            
            if isinstance(market_df, dict) and 'posts' in market_df:
                posts_df = pd.DataFrame(market_df['posts'])
                self.logger.info(f"Retrieved {len(posts_df)} market posts")
            else:
                posts_df = pd.DataFrame()
                self.logger.warning("No market posts retrieved")
            
            # Step 2: Data quality validation
            validation_result = self.data_validator.validate_market_data(posts_df)
            
            if validation_result.is_valid:
                self.logger.info(f"✅ Market data validation passed (Quality Score: {validation_result.quality_score:.2f})")
                market_df = validation_result.cleaned_data
            else:
                self.logger.warning(f"⚠️ Market data validation issues found (Quality Score: {validation_result.quality_score:.2f})")
                for issue in validation_result.issues:
                    self.logger.warning(f"  - {issue}")
                market_df = validation_result.cleaned_data
            
            # Step 3: ML enhancement
            if self.config.enable_ml_enhancement and not market_df.empty:
                self.logger.info("🤖 Applying ML enhancement to market data...")
                enhanced_market = self.ml_engine.enhance_market_analysis(market_df)
                market_df = enhanced_market
            
            # Store results
            self.results['market_data'] = market_df
            self.results['market_validation'] = validation_result
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ Market analysis completed in {execution_time:.2f} seconds")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in market analysis: {e}")
            return ValidationResult(is_valid=False, quality_score=0.0, issues=[str(e)])
    
    def run_enhanced_pubmed_analysis(self) -> Dict[str, Any]:
        """Run enhanced PubMed literature analysis for hypothesis testing"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📚 ENHANCED PUBMED LITERATURE ANALYSIS")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Run PubMed analysis
            pubmed_results = self.pubmed_analyzer.analyze_placebo_literature()
            
            self.logger.info(f"Retrieved {pubmed_results['total_articles']} PubMed articles")
            self.logger.info(f"Digital placebo articles: {pubmed_results['digital_placebo_articles']}")
            self.logger.info(f"Open-label placebo articles: {pubmed_results['placebo_articles']}")
            
            # Store results
            self.results['pubmed_data'] = pubmed_results
            
            # Generate validation result
            evidence = pubmed_results.get('hypothesis_evidence', {})
            literature_support = evidence.get('literature_support', {})
            
            digital_evidence = literature_support.get('digital_placebo_evidence', {})
            olp_evidence = literature_support.get('open_label_placebo_evidence', {})
            
            # Calculate quality score based on evidence strength
            quality_score = 0.0
            issues = []
            
            if digital_evidence.get('strength') == 'Strong':
                quality_score += 0.5
            elif digital_evidence.get('strength') == 'Moderate':
                quality_score += 0.3
            else:
                issues.append("Limited digital placebo literature")
            
            if olp_evidence.get('strength') == 'Strong':
                quality_score += 0.5
            elif olp_evidence.get('strength') == 'Moderate':
                quality_score += 0.3
            else:
                issues.append("Limited open-label placebo literature")
            
            validation_result = ValidationResult(
                is_valid=quality_score > 0.3,
                quality_score=quality_score,
                issues=issues
            )
            
            self.results['pubmed_validation'] = validation_result
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ PubMed analysis completed in {execution_time:.2f} seconds")
            
            return pubmed_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in PubMed analysis: {e}")
            return {}
    
    def run_enhanced_openai_analysis(self) -> Dict[str, Any]:
        """Run enhanced OpenAI analysis on all data sources"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🤖 ENHANCED OPENAI ANALYSIS")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Get all data sources
            clinical_data = self.results.get('clinical_data', pd.DataFrame())
            market_data = self.results.get('market_data', {})
            pubmed_data = self.results.get('pubmed_data', {})
            
            # Process all data through OpenAI
            openai_insights = self.openai_processor.process_all_data(
                clinical_data, market_data, pubmed_data
            )
            
            # Store results
            self.results['openai_insights'] = openai_insights
            
            # Log key insights
            hypothesis_validation = openai_insights.get('hypothesis_validation', {})
            validation_score = hypothesis_validation.get('validation_score', 0)
            
            self.logger.info(f"OpenAI analysis completed:")
            self.logger.info(f"  - Hypothesis validation score: {validation_score}/100")
            self.logger.info(f"  - Clinical insights: {len(openai_insights.get('clinical_insights', {}))} items")
            self.logger.info(f"  - Market insights: {len(openai_insights.get('market_insights', {}))} items")
            self.logger.info(f"  - PubMed insights: {len(openai_insights.get('pubmed_insights', {}))} items")
            self.logger.info(f"  - Cross-analysis: {len(openai_insights.get('cross_analysis', {}))} items")
            self.logger.info(f"  - Recommendations: {len(openai_insights.get('recommendations', {}))} items")
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ OpenAI analysis completed in {execution_time:.2f} seconds")
            
            return openai_insights
            
        except Exception as e:
            self.logger.error(f"❌ Error in OpenAI analysis: {e}")
            return {}
    
    def generate_advanced_visualizations(self):
        """Generate comprehensive visualizations and dashboards"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📈 GENERATING ADVANCED VISUALIZATIONS")
        self.logger.info("="*60)
        
        try:
            clinical_df = self.results['clinical_data']
            market_df = self.results['market_data']
            
            output_dir = self.config.output.output_dir
            
            # 1. Clinical Dashboard
            if not clinical_df.empty:
                self.logger.info("Creating clinical trials dashboard...")
                clinical_dashboard = self.viz_engine.create_clinical_dashboard(
                    clinical_df, 
                    f"{output_dir}/clinical_dashboard.html"
                )
                self.logger.info(f"✅ Clinical dashboard: {clinical_dashboard}")
            
            # 2. Market Dashboard
            if not market_df.empty:
                self.logger.info("Creating market analysis dashboard...")
                market_dashboard = self.viz_engine.create_market_dashboard(
                    market_df,
                    f"{output_dir}/market_dashboard.html"
                )
                self.logger.info(f"✅ Market dashboard: {market_dashboard}")
            
            # 3. Comparative Analysis
            if not clinical_df.empty and not market_df.empty:
                self.logger.info("Creating comparative analysis dashboard...")
                comparative_dashboard = self.viz_engine.create_comparative_analysis(
                    clinical_df, market_df,
                    f"{output_dir}/comparative_analysis.html"
                )
                self.logger.info(f"✅ Comparative dashboard: {comparative_dashboard}")
            
            # 4. Executive Summary
            self.logger.info("Creating executive summary visualization...")
            executive_summary = self.viz_engine.create_executive_summary_visual(
                clinical_df, market_df,
                f"{output_dir}/executive_summary.html"
            )
            self.logger.info(f"✅ Executive summary: {executive_summary}")
            
            self.logger.info("✅ All visualizations generated successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Visualization generation failed: {e}")
    
    def generate_comprehensive_reports(self):
        """Generate comprehensive analysis reports"""
        self.logger.info("\n" + "="*60)
        self.logger.info("📋 GENERATING COMPREHENSIVE REPORTS")
        self.logger.info("="*60)
        
        try:
            output_dir = self.config.output.output_dir
            
            # 1. Data Quality Report
            if self.results['clinical_validation'] and self.results['market_validation']:
                quality_report = self.data_validator.generate_quality_report(
                    self.results['clinical_validation'],
                    self.results['market_validation']
                )
                
                with open(f"{output_dir}/data_quality_report.md", 'w') as f:
                    f.write(quality_report)
                self.logger.info("✅ Data quality report generated")
            
            # 2. ML Insights Report
            if self.config.enable_ml_enhancement:
                ml_report = self.ml_engine.generate_ml_insights_report(
                    self.results['clinical_data'],
                    self.results['market_data']
                )
                
                with open(f"{output_dir}/ml_insights_report.md", 'w') as f:
                    f.write(ml_report)
                self.logger.info("✅ ML insights report generated")
            
            # 3. Visualization Report
            viz_report = self.viz_engine.generate_visualization_report()
            with open(f"{output_dir}/visualization_report.md", 'w') as f:
                f.write(viz_report)
            self.logger.info("✅ Visualization report generated")
            
            # 4. Enhanced Executive Summary
            exec_report = self._generate_enhanced_executive_summary()
            with open(f"{output_dir}/enhanced_executive_summary.md", 'w') as f:
                f.write(exec_report)
            self.logger.info("✅ Enhanced executive summary generated")
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
    
    def _generate_enhanced_executive_summary(self) -> str:
        """Generate enhanced executive summary with all insights"""
        report = []
        report.append("# Enhanced PlaceboRx Validation Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Mode: {self.config.mode.value}")
        report.append(f"Validation Level: {self.config.quality.validation_level.value}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        
        clinical_df = self.results['clinical_data']
        market_df = self.results['market_data']
        
        # Clinical validation summary
        if not clinical_df.empty:
            report.append("### Clinical Evidence Assessment")
            
            total_trials = len(clinical_df)
            digital_trials = clinical_df['is_digital'].sum() if 'is_digital' in clinical_df.columns else 0
            
            if 'predicted_success_probability' in clinical_df.columns:
                avg_success = clinical_df['predicted_success_probability'].mean()
                high_success_trials = (clinical_df['predicted_success_probability'] > 0.7).sum()
                
                report.append(f"- **Total relevant trials analyzed**: {total_trials}")
                report.append(f"- **Digital intervention trials**: {digital_trials}")
                report.append(f"- **Average predicted success probability**: {avg_success:.2f}")
                report.append(f"- **High-probability success trials**: {high_success_trials}")
                
                if avg_success > 0.7:
                    report.append("✅ **CLINICAL VALIDATION: STRONG**")
                elif avg_success > 0.5:
                    report.append("✅ **CLINICAL VALIDATION: MODERATE**")
                else:
                    report.append("⚠️ **CLINICAL VALIDATION: WEAK**")
            
            # Quality assessment
            if self.results['clinical_validation']:
                quality_score = self.results['clinical_validation'].quality_score
                report.append(f"- **Data quality score**: {quality_score:.2f}/1.00")
        
        report.append("")
        
        # Market validation summary
        if not market_df.empty:
            report.append("### Market Demand Assessment")
            
            total_posts = len(market_df)
            report.append(f"- **Total relevant posts analyzed**: {total_posts}")
            
            if 'desperation_intensity' in market_df.columns:
                avg_desperation = market_df['desperation_intensity'].mean()
                high_desperation = (market_df['desperation_intensity'] > 2).sum()
                report.append(f"- **Average desperation level**: {avg_desperation:.2f}")
                report.append(f"- **High desperation posts**: {high_desperation}")
            
            if 'openness_level' in market_df.columns:
                avg_openness = market_df['openness_level'].mean()
                report.append(f"- **Average openness to alternatives**: {avg_openness:.2f}")
            
            if 'advanced_sentiment' in market_df.columns:
                sentiment_dist = market_df['advanced_sentiment'].value_counts()
                report.append("- **Sentiment distribution**:")
                for sentiment, count in sentiment_dist.items():
                    pct = (count / len(market_df)) * 100
                    report.append(f"  - {sentiment}: {count} ({pct:.1f}%)")
            
            # Market validation score
            if 'desperation_intensity' in market_df.columns and 'openness_level' in market_df.columns:
                market_score = (
                    min(market_df['desperation_intensity'].mean() / 3, 1.0) +
                    min(market_df['openness_level'].mean() / 3, 1.0)
                ) / 2
                
                if market_score > 0.7:
                    report.append("✅ **MARKET VALIDATION: STRONG**")
                elif market_score > 0.4:
                    report.append("✅ **MARKET VALIDATION: MODERATE**")
                else:
                    report.append("⚠️ **MARKET VALIDATION: WEAK**")
            
            # Quality assessment
            if self.results['market_validation']:
                quality_score = self.results['market_validation'].quality_score
                report.append(f"- **Data quality score**: {quality_score:.2f}/1.00")
        
        # ML Insights Summary
        if self.config.enable_ml_enhancement:
            report.append("")
            report.append("### Machine Learning Insights")
            
            if 'user_persona' in market_df.columns:
                persona_dist = market_df['user_persona'].value_counts()
                report.append("**Identified User Personas**:")
                for persona, count in persona_dist.items():
                    report.append(f"- {persona.title()}: {count} users")
            
            if 'trial_cluster' in clinical_df.columns:
                n_clusters = clinical_df['trial_cluster'].nunique()
                report.append(f"**Clinical Trial Clusters**: {n_clusters} distinct groups identified")
            
            if 'primary_emotion' in market_df.columns:
                top_emotions = market_df['primary_emotion'].value_counts().head(3)
                report.append("**Top Emotional Patterns**:")
                for emotion, count in top_emotions.items():
                    report.append(f"- {emotion}: {count}")
        
        # Opportunity Analysis
        report.append("")
        report.append("### Key Opportunities Identified")
        
        # Digital placebo opportunity
        if not clinical_df.empty and 'is_digital' in clinical_df.columns:
            digital_ratio = clinical_df['is_digital'].mean()
            if digital_ratio < 0.3:
                report.append("🎯 **Digital Placebo Gap**: Low representation of digital interventions in clinical trials")
        
        # High desperation market segments
        if not market_df.empty and 'desperation_intensity' in market_df.columns:
            high_desperation_segments = market_df[market_df['desperation_intensity'] > 2]
            if len(high_desperation_segments) > 0:
                top_subreddits = high_desperation_segments['subreddit'].value_counts().head(3)
                report.append("🎯 **High-Desperation Market Segments**:")
                for subreddit, count in top_subreddits.items():
                    report.append(f"  - r/{subreddit}: {count} high-desperation posts")
        
        # Configuration and Performance
        report.append("")
        report.append("### Analysis Configuration")
        report.append(f"- **Analysis Mode**: {self.config.mode.value}")
        report.append(f"- **Validation Level**: {self.config.quality.validation_level.value}")
        report.append(f"- **ML Enhancement**: {'Enabled' if self.config.enable_ml_enhancement else 'Disabled'}")
        report.append(f"- **Total Execution Time**: {self.results['execution_time']:.1f} minutes")
        
        # Next Steps
        report.append("")
        report.append("### Recommended Next Steps")
        
        # Determine recommendations based on validation scores
        clinical_score = 0.5  # Default
        market_score = 0.5    # Default
        
        if not clinical_df.empty and 'predicted_success_probability' in clinical_df.columns:
            clinical_score = clinical_df['predicted_success_probability'].mean()
        
        if not market_df.empty and 'desperation_intensity' in market_df.columns and 'openness_level' in market_df.columns:
            market_score = (
                min(market_df['desperation_intensity'].mean() / 3, 1.0) +
                min(market_df['openness_level'].mean() / 3, 1.0)
            ) / 2
        
        if clinical_score > 0.6 and market_score > 0.6:
            report.append("1. **HIGH PRIORITY**: Proceed with MVP development immediately")
            report.append("2. **MEDIUM PRIORITY**: Conduct small-scale pilot studies")
            report.append("3. **MEDIUM PRIORITY**: Regulatory pathway consultation")
        elif clinical_score > 0.4 or market_score > 0.6:
            report.append("1. **HIGH PRIORITY**: Conduct additional clinical research")
            report.append("2. **HIGH PRIORITY**: Develop prototype for market testing")
            report.append("3. **MEDIUM PRIORITY**: Investor presentation preparation")
        else:
            report.append("1. **HIGH PRIORITY**: Reassess product-market fit")
            report.append("2. **MEDIUM PRIORITY**: Explore alternative approaches")
            report.append("3. **LOW PRIORITY**: Consider pivot opportunities")
        
        # File References
        report.append("")
        report.append("### Generated Artifacts")
        report.append("**Data Files**:")
        report.append("- `enhanced_clinical_trials.csv` - Processed clinical trials data")
        report.append("- `enhanced_market_analysis.csv` - Processed market analysis data")
        report.append("")
        report.append("**Interactive Dashboards**:")
        report.append("- `clinical_dashboard.html` - Clinical trials analysis")
        report.append("- `market_dashboard.html` - Market validation analysis")
        report.append("- `comparative_analysis.html` - Cross-analysis insights")
        report.append("- `executive_summary.html` - Executive summary dashboard")
        report.append("")
        report.append("**Detailed Reports**:")
        report.append("- `data_quality_report.md` - Data quality assessment")
        report.append("- `ml_insights_report.md` - Machine learning insights")
        report.append("- `visualization_report.md` - Visualization summary")
        
        return '\n'.join(report)
    
    def save_models_and_artifacts(self):
        """Save trained models and pipeline artifacts"""
        self.logger.info("💾 Saving models and artifacts...")
        
        try:
            # Save ML models
            if self.config.enable_ml_enhancement:
                self.ml_engine.save_models("models")
            
            # Save configuration
            config_path = f"{self.config.output.output_dir}/pipeline_config.json"
            self.config.save_to_file(config_path)
            
            # Save execution metadata
            metadata = {
                'execution_time': self.results['execution_time'],
                'timestamp': datetime.now().isoformat(),
                'config_mode': self.config.mode.value,
                'validation_level': self.config.quality.validation_level.value,
                'ml_enabled': self.config.enable_ml_enhancement,
                'clinical_records': len(self.results['clinical_data']),
                'market_records': len(self.results['market_data'])
            }
            
            import json
            with open(f"{self.config.output.output_dir}/execution_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info("✅ Models and artifacts saved successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models and artifacts: {e}")
    
    def run_complete_pipeline(self):
        """Run the complete enhanced validation pipeline"""
        pipeline_start_time = time.time()
        
        self.logger.info("🚀 Enhanced PlaceboRx Validation Pipeline")
        self.logger.info(f"Analysis Mode: {self.config.mode.value}")
        self.logger.info(f"Validation Level: {self.config.quality.validation_level.value}")
        self.logger.info("="*60)
        
        try:
            # 1. Environment validation
            if not self.validate_environment():
                self.logger.error("❌ Environment validation failed")
                return False
            
            # 2. Enhanced clinical analysis
            clinical_validation = self.run_enhanced_clinical_analysis()
            
            # 3. Enhanced market analysis
            market_validation = self.run_enhanced_market_analysis()
            
            # 4. Enhanced PubMed analysis
            pubmed_data = self.run_enhanced_pubmed_analysis()
            
            # 5. Enhanced OpenAI analysis
            openai_insights = self.run_enhanced_openai_analysis()
            
            # 6. Generate visualizations
            if self.config.output.include_visualizations:
                self.generate_advanced_visualizations()
            
            # 7. Generate comprehensive reports
            self.generate_comprehensive_reports()
            
            # 8. Save models and artifacts
            self.save_models_and_artifacts()
            
            # Calculate total execution time
            total_time = time.time() - pipeline_start_time
            self.results['execution_time'] = total_time / 60  # Convert to minutes
            
            # Final summary
            self.logger.info("\n" + "="*60)
            self.logger.info("🎉 ENHANCED PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            self.logger.info(f"⏱️ Total execution time: {total_time/60:.1f} minutes")
            self.logger.info(f"📊 Clinical trials analyzed: {len(self.results['clinical_data'])}")
            self.logger.info(f"📱 Market posts analyzed: {len(self.results['market_data'])}")
            
            if clinical_validation and market_validation:
                avg_quality = (clinical_validation.quality_score + market_validation.quality_score) / 2
                self.logger.info(f"🏆 Overall data quality score: {avg_quality:.2f}/1.00")
            
            self.logger.info(f"📁 All outputs saved to: {self.config.output.output_dir}/")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            return False

def main():
    """Main execution function"""
    try:
        # Create and run enhanced pipeline
        pipeline = EnhancedValidationPipeline()
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\n✅ Enhanced validation pipeline completed successfully!")
            print(f"📊 Check the '{CONFIG.output.output_dir}' directory for all outputs")
            print("🌐 Open the HTML dashboards in your browser for interactive analysis")
        else:
            print("\n❌ Pipeline execution failed. Check the logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())