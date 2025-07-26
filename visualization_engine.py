import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import logging

from enhanced_config import CONFIG

class VisualizationEngine:
    """Advanced visualization and dashboard creation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_style()
        
    def setup_style(self):
        """Setup visualization styling"""
        plt.style.use(CONFIG.output.plot_style)
        sns.set_palette(CONFIG.output.color_palette)
        
        # Set default figure parameters
        plt.rcParams['figure.dpi'] = CONFIG.output.figure_dpi
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def create_clinical_dashboard(self, df: pd.DataFrame, output_path: str = "clinical_dashboard.html") -> str:
        """Create comprehensive clinical trials dashboard"""
        if df.empty:
            self.logger.warning("No clinical data provided for dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Trial Phase Distribution', 'Enrollment Distribution',
                'Trial Success Predictions', 'Intervention Categories',
                'Timeline Analysis', 'Geographic Distribution'
            ],
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "choropleth"}]]
        )
        
        # 1. Trial Phase Distribution (Pie Chart)
        if 'phase' in df.columns:
            phase_counts = df['phase'].value_counts()
            fig.add_trace(
                go.Pie(labels=phase_counts.index, values=phase_counts.values, name="Phase"),
                row=1, col=1
            )
        
        # 2. Enrollment Distribution (Histogram)
        if 'enrollment' in df.columns:
            fig.add_trace(
                go.Histogram(x=df['enrollment'], name="Enrollment", nbinsx=20),
                row=1, col=2
            )
        
        # 3. Trial Success Predictions (Scatter Plot)
        if 'predicted_success_probability' in df.columns and 'enrollment' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['enrollment'],
                    y=df['predicted_success_probability'],
                    mode='markers',
                    name="Success Prediction",
                    text=df.get('title', ''),
                    hovertemplate='<b>%{text}</b><br>Enrollment: %{x}<br>Success Prob: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Intervention Categories (Bar Chart)
        if 'intervention_category' in df.columns:
            cat_counts = df['intervention_category'].value_counts()
            fig.add_trace(
                go.Bar(x=cat_counts.index, y=cat_counts.values, name="Categories"),
                row=2, col=2
            )
        
        # 5. Timeline Analysis
        if 'completion_date' in df.columns:
            df_dated = df.dropna(subset=['completion_date'])
            if not df_dated.empty:
                df_dated['year'] = pd.to_datetime(df_dated['completion_date']).dt.year
                timeline_counts = df_dated['year'].value_counts().sort_index()
                fig.add_trace(
                    go.Scatter(
                        x=timeline_counts.index,
                        y=timeline_counts.values,
                        mode='lines+markers',
                        name="Trials per Year"
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title_text="Clinical Trials Analysis Dashboard",
            showlegend=False,
            height=1200,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html(output_path)
        
        # Create additional detailed visualizations
        self._create_detailed_clinical_plots(df)
        
        return output_path
    
    def create_market_dashboard(self, df: pd.DataFrame, output_path: str = "market_dashboard.html") -> str:
        """Create comprehensive market analysis dashboard"""
        if df.empty:
            self.logger.warning("No market data provided for dashboard")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Sentiment Distribution', 'Engagement Over Time',
                'Subreddit Activity', 'Emotion Analysis',
                'User Personas', 'Desperation vs Openness'
            ],
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Sentiment Distribution
        if 'advanced_sentiment' in df.columns:
            sentiment_counts = df['advanced_sentiment'].value_counts()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    name="Sentiment",
                    marker_colors=colors[:len(sentiment_counts)]
                ),
                row=1, col=1
            )
        
        # 2. Engagement Over Time
        if 'created_utc' in df.columns and 'score' in df.columns:
            df_time = df.copy()
            df_time['date'] = pd.to_datetime(df_time['created_utc'], unit='s')
            daily_engagement = df_time.groupby(df_time['date'].dt.date)['score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_engagement.index,
                    y=daily_engagement.values,
                    mode='lines+markers',
                    name="Avg Daily Score"
                ),
                row=1, col=2
            )
        
        # 3. Subreddit Activity
        if 'subreddit' in df.columns:
            subreddit_counts = df['subreddit'].value_counts().head(10)
            fig.add_trace(
                go.Bar(
                    x=subreddit_counts.values,
                    y=subreddit_counts.index,
                    orientation='h',
                    name="Post Counts"
                ),
                row=2, col=1
            )
        
        # 4. Emotion Analysis
        if 'primary_emotion' in df.columns:
            emotion_counts = df['primary_emotion'].value_counts().head(8)
            fig.add_trace(
                go.Bar(x=emotion_counts.index, y=emotion_counts.values, name="Emotions"),
                row=2, col=2
            )
        
        # 5. User Personas
        if 'user_persona' in df.columns:
            persona_counts = df['user_persona'].value_counts()
            fig.add_trace(
                go.Pie(labels=persona_counts.index, values=persona_counts.values, name="Personas"),
                row=3, col=1
            )
        
        # 6. Desperation vs Openness
        if 'desperation_intensity' in df.columns and 'openness_level' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['desperation_intensity'],
                    y=df['openness_level'],
                    mode='markers',
                    name="Psychological Profile",
                    text=df.get('title', ''),
                    marker=dict(
                        size=df.get('score', 5),
                        color=df.get('sentiment_confidence', 0.5),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    hovertemplate='<b>%{text}</b><br>Desperation: %{x}<br>Openness: %{y}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Market Analysis Dashboard",
            showlegend=False,
            height=1200,
            template="plotly_white"
        )
        
        # Save dashboard
        fig.write_html(output_path)
        
        # Create additional detailed visualizations
        self._create_detailed_market_plots(df)
        
        return output_path
    
    def create_comparative_analysis(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame,
                                  output_path: str = "comparative_dashboard.html") -> str:
        """Create comparative analysis dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Clinical vs Market Timeline',
                'Success Correlation Analysis',
                'Market Validation Summary',
                'Opportunity Matrix'
            ]
        )
        
        # 1. Timeline Comparison
        if not clinical_df.empty and not market_df.empty:
            # Clinical timeline
            if 'completion_date' in clinical_df.columns:
                clinical_timeline = self._prepare_timeline_data(clinical_df, 'completion_date', 'Clinical Trials')
                
            # Market timeline
            if 'created_utc' in market_df.columns:
                market_timeline = self._prepare_market_timeline(market_df)
                
                fig.add_trace(
                    go.Scatter(
                        x=market_timeline.index,
                        y=market_timeline.values,
                        name="Market Activity",
                        line=dict(color='orange')
                    ),
                    row=1, col=1
                )
        
        # 2. Success Correlation
        if 'predicted_success_probability' in clinical_df.columns:
            success_stats = self._calculate_success_metrics(clinical_df, market_df)
            
            fig.add_trace(
                go.Bar(
                    x=list(success_stats.keys()),
                    y=list(success_stats.values()),
                    name="Success Metrics"
                ),
                row=1, col=2
            )
        
        # 3. Market Validation Summary
        validation_scores = self._calculate_validation_scores(clinical_df, market_df)
        
        fig.add_trace(
            go.Scatterpolar(
                r=list(validation_scores.values()),
                theta=list(validation_scores.keys()),
                fill='toself',
                name='Validation Profile'
            ),
            row=2, col=1
        )
        
        # 4. Opportunity Matrix
        opportunities = self._identify_opportunities(clinical_df, market_df)
        
        if opportunities:
            fig.add_trace(
                go.Scatter(
                    x=[opp['market_demand'] for opp in opportunities],
                    y=[opp['clinical_evidence'] for opp in opportunities],
                    mode='markers+text',
                    text=[opp['name'] for opp in opportunities],
                    textposition="top center",
                    marker=dict(size=15, color='red'),
                    name="Opportunities"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Comparative Analysis Dashboard",
            height=800,
            template="plotly_white"
        )
        
        fig.write_html(output_path)
        return output_path
    
    def _create_detailed_clinical_plots(self, df: pd.DataFrame):
        """Create detailed clinical analysis plots"""
        try:
            # Trial Success Analysis
            if 'predicted_success_probability' in df.columns:
                plt.figure(figsize=(15, 10))
                
                # Success probability distribution
                plt.subplot(2, 3, 1)
                plt.hist(df['predicted_success_probability'], bins=20, alpha=0.7, color='skyblue')
                plt.title('Success Probability Distribution')
                plt.xlabel('Predicted Success Probability')
                plt.ylabel('Number of Trials')
                
                # Success by phase
                if 'phase' in df.columns:
                    plt.subplot(2, 3, 2)
                    phase_success = df.groupby('phase')['predicted_success_probability'].mean()
                    plt.bar(phase_success.index, phase_success.values, color='lightcoral')
                    plt.title('Average Success by Phase')
                    plt.xticks(rotation=45)
                
                # Enrollment vs Success
                if 'enrollment' in df.columns:
                    plt.subplot(2, 3, 3)
                    plt.scatter(df['enrollment'], df['predicted_success_probability'], alpha=0.6)
                    plt.xlabel('Enrollment')
                    plt.ylabel('Predicted Success')
                    plt.title('Enrollment vs Success Prediction')
                
                # Digital vs Non-digital trials
                if 'is_digital' in df.columns:
                    plt.subplot(2, 3, 4)
                    digital_success = df.groupby('is_digital')['predicted_success_probability'].mean()
                    plt.bar(['Non-Digital', 'Digital'], digital_success.values, color=['lightblue', 'darkblue'])
                    plt.title('Success Rate: Digital vs Non-Digital')
                
                # Trial clusters
                if 'trial_cluster' in df.columns:
                    plt.subplot(2, 3, 5)
                    cluster_counts = df['trial_cluster'].value_counts()
                    plt.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index], autopct='%1.1f%%')
                    plt.title('Trial Clusters')
                
                plt.tight_layout()
                plt.savefig('clinical_detailed_analysis.png', dpi=CONFIG.output.figure_dpi, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"Could not create detailed clinical plots: {e}")
    
    def _create_detailed_market_plots(self, df: pd.DataFrame):
        """Create detailed market analysis plots"""
        try:
            plt.figure(figsize=(15, 12))
            
            # Sentiment analysis
            if 'advanced_sentiment' in df.columns:
                plt.subplot(3, 3, 1)
                sentiment_counts = df['advanced_sentiment'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                plt.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors, autopct='%1.1f%%')
                plt.title('Sentiment Distribution')
                
            # Engagement patterns
            if 'score' in df.columns and 'num_comments' in df.columns:
                plt.subplot(3, 3, 2)
                plt.scatter(df['score'], df['num_comments'], alpha=0.6)
                plt.xlabel('Score')
                plt.ylabel('Number of Comments')
                plt.title('Engagement Pattern')
                
            # Subreddit comparison
            if 'subreddit' in df.columns:
                plt.subplot(3, 3, 3)
                top_subreddits = df['subreddit'].value_counts().head(8)
                plt.bar(range(len(top_subreddits)), top_subreddits.values)
                plt.xticks(range(len(top_subreddits)), top_subreddits.index, rotation=45)
                plt.title('Top Subreddits by Activity')
                
            # Emotional intensity
            if 'primary_emotion' in df.columns and 'emotion_confidence' in df.columns:
                plt.subplot(3, 3, 4)
                emotion_intensity = df.groupby('primary_emotion')['emotion_confidence'].mean()
                plt.bar(emotion_intensity.index, emotion_intensity.values, color='orange')
                plt.xticks(rotation=45)
                plt.title('Average Emotion Intensity')
                
            # Desperation analysis
            if 'desperation_intensity' in df.columns:
                plt.subplot(3, 3, 5)
                plt.hist(df['desperation_intensity'], bins=15, alpha=0.7, color='red')
                plt.title('Desperation Level Distribution')
                plt.xlabel('Desperation Intensity')
                
            # Openness analysis
            if 'openness_level' in df.columns:
                plt.subplot(3, 3, 6)
                plt.hist(df['openness_level'], bins=15, alpha=0.7, color='green')
                plt.title('Openness Level Distribution')
                plt.xlabel('Openness Level')
                
            # User personas
            if 'user_persona' in df.columns:
                plt.subplot(3, 3, 7)
                persona_counts = df['user_persona'].value_counts()
                plt.pie(persona_counts.values, labels=persona_counts.index, autopct='%1.1f%%')
                plt.title('User Personas')
                
            # Readability analysis
            if 'readability_score' in df.columns:
                plt.subplot(3, 3, 8)
                plt.hist(df['readability_score'], bins=15, alpha=0.7, color='purple')
                plt.title('Text Readability Distribution')
                plt.xlabel('Readability Score')
                
            # Predicted engagement
            if 'predicted_engagement' in df.columns and 'score' in df.columns:
                plt.subplot(3, 3, 9)
                plt.scatter(df['predicted_engagement'], df['score'], alpha=0.6)
                plt.xlabel('Predicted Engagement')
                plt.ylabel('Actual Score')
                plt.title('Engagement Prediction vs Reality')
                
            plt.tight_layout()
            plt.savefig('market_detailed_analysis.png', dpi=CONFIG.output.figure_dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create detailed market plots: {e}")
    
    def _prepare_timeline_data(self, df: pd.DataFrame, date_col: str, name: str) -> pd.Series:
        """Prepare timeline data for visualization"""
        df_clean = df.dropna(subset=[date_col])
        if df_clean.empty:
            return pd.Series()
        
        df_clean['date'] = pd.to_datetime(df_clean[date_col])
        return df_clean.groupby(df_clean['date'].dt.to_period('M')).size()
    
    def _prepare_market_timeline(self, df: pd.DataFrame) -> pd.Series:
        """Prepare market timeline data"""
        df_clean = df.dropna(subset=['created_utc'])
        if df_clean.empty:
            return pd.Series()
        
        df_clean['date'] = pd.to_datetime(df_clean['created_utc'], unit='s')
        return df_clean.groupby(df_clean['date'].dt.to_period('D')).size()
    
    def _calculate_success_metrics(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate success metrics for comparison"""
        metrics = {}
        
        if not clinical_df.empty:
            if 'predicted_success_probability' in clinical_df.columns:
                metrics['Avg Clinical Success'] = clinical_df['predicted_success_probability'].mean()
                metrics['High Success Trials'] = (clinical_df['predicted_success_probability'] > 0.7).mean()
        
        if not market_df.empty:
            if 'desperation_intensity' in market_df.columns:
                metrics['Market Desperation'] = market_df['desperation_intensity'].mean()
            if 'openness_level' in market_df.columns:
                metrics['Market Openness'] = market_df['openness_level'].mean()
        
        return metrics
    
    def _calculate_validation_scores(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate validation scores for radar chart"""
        scores = {
            'Clinical Evidence': 0.5,
            'Market Demand': 0.5,
            'Digital Feasibility': 0.5,
            'Commercial Viability': 0.5,
            'User Engagement': 0.5
        }
        
        if not clinical_df.empty:
            # Clinical evidence score
            if 'predicted_success_probability' in clinical_df.columns:
                scores['Clinical Evidence'] = clinical_df['predicted_success_probability'].mean()
            
            # Digital feasibility
            if 'is_digital' in clinical_df.columns:
                scores['Digital Feasibility'] = clinical_df['is_digital'].mean()
        
        if not market_df.empty:
            # Market demand
            if 'desperation_intensity' in market_df.columns:
                scores['Market Demand'] = min(market_df['desperation_intensity'].mean() / 3, 1.0)
            
            # User engagement
            if 'predicted_engagement' in market_df.columns:
                scores['User Engagement'] = min(market_df['predicted_engagement'].mean() / 50, 1.0)
            
            # Commercial viability
            if 'openness_level' in market_df.columns:
                scores['Commercial Viability'] = min(market_df['openness_level'].mean() / 3, 1.0)
        
        return scores
    
    def _identify_opportunities(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> List[Dict]:
        """Identify key opportunities based on data analysis"""
        opportunities = []
        
        # High demand + low clinical evidence = research opportunity
        if not market_df.empty and not clinical_df.empty:
            # Digital placebo opportunity
            if 'is_digital' in clinical_df.columns:
                digital_trials = clinical_df['is_digital'].mean()
                if digital_trials < 0.3:  # Low digital presence
                    opportunities.append({
                        'name': 'Digital Placebo',
                        'market_demand': 0.8,
                        'clinical_evidence': digital_trials
                    })
            
            # High desperation conditions
            if 'desperation_intensity' in market_df.columns:
                avg_desperation = market_df['desperation_intensity'].mean()
                if avg_desperation > 1.5:
                    opportunities.append({
                        'name': 'High Desperation Market',
                        'market_demand': min(avg_desperation / 3, 1.0),
                        'clinical_evidence': 0.4
                    })
        
        return opportunities
    
    def create_executive_summary_visual(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame,
                                      output_path: str = "executive_summary.html") -> str:
        """Create executive summary visualization"""
        # Create a comprehensive summary dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Clinical Validation Score', 'Market Demand Score', 'Overall Opportunity',
                'Key Metrics', 'Risk Assessment', 'Recommended Actions'
            ],
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        # Calculate key scores
        clinical_score = self._calculate_clinical_validation_score(clinical_df)
        market_score = self._calculate_market_validation_score(market_df)
        overall_score = (clinical_score + market_score) / 2
        
        # 1. Clinical Validation Score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=clinical_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Clinical Validation (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. Market Demand Score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=market_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Market Demand (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # 3. Overall Opportunity Score
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=overall_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Opportunity (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "purple"},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        # 4. Key Metrics
        metrics = self._get_key_metrics(clinical_df, market_df)
        fig.add_trace(
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                name="Key Metrics"
            ),
            row=2, col=1
        )
        
        # 5. Risk Assessment
        risks = self._assess_risks(clinical_df, market_df)
        fig.add_trace(
            go.Bar(
                x=list(risks.keys()),
                y=list(risks.values()),
                name="Risk Levels",
                marker_color='red'
            ),
            row=2, col=2
        )
        
        # 6. Recommended Actions (Table)
        actions = self._get_recommended_actions(clinical_score, market_score)
        fig.add_trace(
            go.Table(
                header=dict(values=['Priority', 'Action', 'Timeline']),
                cells=dict(values=[
                    [action['priority'] for action in actions],
                    [action['action'] for action in actions],
                    [action['timeline'] for action in actions]
                ])
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title_text="PlaceboRx Validation Executive Summary",
            height=800,
            template="plotly_white"
        )
        
        fig.write_html(output_path)
        return output_path
    
    def _calculate_clinical_validation_score(self, df: pd.DataFrame) -> float:
        """Calculate overall clinical validation score"""
        if df.empty:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Success probability
        if 'predicted_success_probability' in df.columns:
            score += df['predicted_success_probability'].mean()
            factors += 1
        
        # Digital intervention presence
        if 'is_digital' in df.columns:
            score += df['is_digital'].mean()
            factors += 1
        
        # Quality indicators
        if 'enrollment' in df.columns:
            high_enrollment = (df['enrollment'] > 100).mean()
            score += high_enrollment
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _calculate_market_validation_score(self, df: pd.DataFrame) -> float:
        """Calculate overall market validation score"""
        if df.empty:
            return 0.0
        
        score = 0.0
        factors = 0
        
        # Desperation level
        if 'desperation_intensity' in df.columns:
            score += min(df['desperation_intensity'].mean() / 3, 1.0)
            factors += 1
        
        # Openness to alternatives
        if 'openness_level' in df.columns:
            score += min(df['openness_level'].mean() / 3, 1.0)
            factors += 1
        
        # Engagement quality
        if 'predicted_engagement' in df.columns:
            score += min(df['predicted_engagement'].mean() / 50, 1.0)
            factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _get_key_metrics(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, float]:
        """Get key metrics for visualization"""
        metrics = {}
        
        if not clinical_df.empty:
            metrics['Total Trials'] = len(clinical_df)
            if 'is_digital' in clinical_df.columns:
                metrics['Digital Trials'] = clinical_df['is_digital'].sum()
        
        if not market_df.empty:
            metrics['Total Posts'] = len(market_df)
            if 'desperation_intensity' in market_df.columns:
                metrics['High Desperation'] = (market_df['desperation_intensity'] > 2).sum()
        
        return metrics
    
    def _assess_risks(self, clinical_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, float]:
        """Assess various risk factors"""
        risks = {
            'Limited Evidence': 0.5,
            'Market Saturation': 0.3,
            'Regulatory': 0.7,
            'Competition': 0.4
        }
        
        # Adjust based on data
        if not clinical_df.empty:
            if len(clinical_df) < 10:
                risks['Limited Evidence'] = 0.8
        
        if not market_df.empty:
            if len(market_df) > 1000:
                risks['Market Saturation'] = 0.6
        
        return risks
    
    def _get_recommended_actions(self, clinical_score: float, market_score: float) -> List[Dict]:
        """Get recommended actions based on scores"""
        actions = []
        
        if clinical_score < 0.5:
            actions.append({
                'priority': 'High',
                'action': 'Conduct pilot clinical study',
                'timeline': '6-12 months'
            })
        
        if market_score > 0.7:
            actions.append({
                'priority': 'High',
                'action': 'Develop MVP',
                'timeline': '3-6 months'
            })
        
        actions.append({
            'priority': 'Medium',
            'action': 'Regulatory consultation',
            'timeline': '1-3 months'
        })
        
        return actions
    
    def generate_visualization_report(self) -> str:
        """Generate a report of all created visualizations"""
        report = []
        report.append("# Visualization Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # List generated files
        viz_files = [
            'clinical_dashboard.html',
            'market_dashboard.html',
            'comparative_dashboard.html',
            'executive_summary.html',
            'clinical_detailed_analysis.png',
            'market_detailed_analysis.png'
        ]
        
        report.append("## Generated Visualizations")
        for file in viz_files:
            if os.path.exists(file):
                report.append(f"✅ {file}")
            else:
                report.append(f"❌ {file}")
        
        report.append("")
        report.append("## Visualization Features")
        report.append("- Interactive dashboards with drill-down capabilities")
        report.append("- Comprehensive statistical analysis plots")
        report.append("- Executive summary with key performance indicators")
        report.append("- Comparative analysis across data sources")
        report.append("- Risk assessment and opportunity identification")
        
        return '\n'.join(report)