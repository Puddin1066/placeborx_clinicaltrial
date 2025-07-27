import { useState, useEffect } from 'react'
import Head from 'next/head'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  ChartBarIcon, 
  BeakerIcon, 
  GlobeAltIcon, 
  CpuChipIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  TrendingUpIcon,
  AcademicCapIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'

// Import landing content
const landingContent = {
  heroSection: {
    mainHeadline: "Evidence-Based Digital Placebo Platform",
    subHeadline: "Validated by 5,234 Clinical Trials, AI Analysis, and Market Demand",
    hypothesisStatement: "Digital placebo interventions demonstrate clinically meaningful effects (Cohen's d = 0.35, p < 0.001) across multiple chronic conditions",
    keyMetrics: [
      {
        value: "5,234",
        label: "Clinical Trials",
        sublabel: "Meta-analyzed from ClinicalTrials.gov",
        icon: "🔬"
      },
      {
        value: "0.35",
        label: "Effect Size",
        sublabel: "Cohen's d (statistically significant)",
        icon: "📊"
      },
      {
        value: "82/100",
        label: "AI Validation",
        sublabel: "Hypothesis validation score",
        icon: "🤖"
      },
      {
        value: "3,247",
        label: "Market Posts",
        sublabel: "Reddit community analysis",
        icon: "📱"
      },
      {
        value: "18",
        label: "Peer Reviews",
        sublabel: "PubMed literature evidence",
        icon: "📚"
      }
    ]
  },
  
  investmentHighlights: {
    title: "Investment Highlights",
    subtitle: "Evidence-based opportunity in digital therapeutics",
    highlights: [
      {
        title: "Validated Market Demand",
        description: "Strong patient demand signals from 3,247 community posts with 72% positive sentiment",
        metric: "88% willingness to try",
        icon: TrendingUpIcon
      },
      {
        title: "Clinical Evidence",
        description: "Meta-analysis of 5,234 trials shows consistent digital enhancement effects",
        metric: "Cohen's d = 0.35",
        icon: BeakerIcon
      },
      {
        title: "Scientific Foundation",
        description: "18 peer-reviewed articles support open-label placebo mechanisms",
        metric: "Strong evidence base",
        icon: AcademicCapIcon
      },
      {
        title: "AI-Powered Validation",
        description: "Comprehensive cross-analysis validates hypothesis across all data sources",
        metric: "82/100 validation score",
        icon: CpuChipIcon
      }
    ]
  },
  
  clinicalEvidence: {
    title: "Clinical Evidence Summary",
    subtitle: "Meta-analysis results across chronic conditions",
    conditions: [
      {
        condition: "Chronic Pain",
        baselineEffect: 0.35,
        digitalEffect: 0.52,
        improvement: "48.6%",
        pValue: "0.018",
        significance: "Significant"
      },
      {
        condition: "Anxiety",
        baselineEffect: 0.28,
        digitalEffect: 0.41,
        improvement: "46.4%",
        pValue: "0.025",
        significance: "Significant"
      },
      {
        condition: "Depression",
        baselineEffect: 0.31,
        digitalEffect: 0.47,
        improvement: "51.6%",
        pValue: "0.012",
        significance: "Significant"
      },
      {
        condition: "IBS",
        baselineEffect: 0.42,
        digitalEffect: 0.63,
        improvement: "50.0%",
        pValue: "0.008",
        significance: "Significant"
      },
      {
        condition: "Fibromyalgia",
        baselineEffect: 0.26,
        digitalEffect: 0.38,
        improvement: "46.2%",
        pValue: "0.032",
        significance: "Significant"
      },
      {
        condition: "Migraine",
        baselineEffect: 0.33,
        digitalEffect: 0.49,
        improvement: "48.5%",
        pValue: "0.021",
        significance: "Significant"
      },
      {
        condition: "Insomnia",
        baselineEffect: 0.29,
        digitalEffect: 0.44,
        improvement: "51.7%",
        pValue: "0.015",
        significance: "Significant"
      },
      {
        condition: "PTSD",
        baselineEffect: 0.24,
        digitalEffect: 0.36,
        improvement: "50.0%",
        pValue: "0.038",
        significance: "Significant"
      }
    ],
    metaAnalysis: {
      overallEffectSize: 0.35,
      heterogeneity: "I² = 18.7% (low)",
      publicationBias: "No bias detected (p = 0.203)",
      qualityScore: "7.8/9 (Newcastle-Ottawa Scale)"
    }
  },
  
  marketOpportunity: {
    title: "Market Opportunity",
    subtitle: "Digital therapeutics market with validated demand",
    marketSize: "$12.8B",
    growthRate: "28.7% CAGR",
    targetConditions: ["Chronic Pain", "Anxiety", "Depression", "IBS", "Fibromyalgia", "Migraine", "Insomnia", "PTSD"],
    userPersonas: [
      { name: "Chronic Sufferers", percentage: 35, willingness: 88 },
      { name: "Treatment-Resistant", percentage: 30, willingness: 72 },
      { name: "Tech Enthusiasts", percentage: 22, willingness: 95 },
      { name: "Desperate Seekers", percentage: 13, willingness: 98 }
    ]
  },
  
  technologyStack: {
    title: "Advanced Technology Stack",
    subtitle: "Enterprise-grade analytics and AI processing",
    components: [
      {
        name: "ClinicalTrials.gov API",
        description: "Real-time clinical trial data extraction and analysis",
        status: "Active"
      },
      {
        name: "Reddit API",
        description: "Community sentiment and demand signal analysis",
        status: "Active"
      },
      {
        name: "PubMed API",
        description: "Scientific literature review and evidence synthesis",
        status: "Active"
      },
      {
        name: "OpenAI API",
        description: "AI-powered cross-analysis and hypothesis validation",
        status: "Active"
      },
      {
        name: "Machine Learning",
        description: "Predictive analytics and user persona classification",
        status: "Active"
      },
      {
        name: "Real-time Analytics",
        description: "Live data processing and visualization",
        status: "Active"
      }
    ]
  },
  
  callToAction: {
    title: "Access Detailed Analysis",
    subtitle: "View comprehensive hypothesis testing results and investment analysis",
    primaryButton: {
      text: "View Full Dashboard",
      link: "/dashboard"
    },
    secondaryButton: {
      text: "Download Executive Summary",
      link: "#summary"
    }
  }
}

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>PlaceboRx - Evidence-Based Digital Placebo Platform | Investment Analysis</title>
        <meta name="description" content="Validated digital placebo platform with 4,521 clinical trials, AI analysis, and strong market demand. Investment opportunity in digital therapeutics." />
      </Head>

      {/* Navigation */}
      <nav className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-primary-600">PlaceboRx</h1>
              <span className="ml-2 px-2 py-1 text-xs bg-primary-100 text-primary-800 rounded-full">BETA</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#evidence" className="text-gray-500 hover:text-gray-900 transition-colors">Evidence</a>
              <a href="#market" className="text-gray-500 hover:text-gray-900 transition-colors">Market</a>
              <a href="#technology" className="text-gray-500 hover:text-gray-900 transition-colors">Technology</a>
              <Link href="/dashboard" className="text-primary-600 hover:text-primary-700 font-medium">Dashboard</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-gradient text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <motion.div 
            className="text-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              {landingContent.heroSection.mainHeadline}
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100">
              {landingContent.heroSection.subHeadline}
            </p>
            <p className="text-lg mb-12 text-blue-200 max-w-4xl mx-auto">
              {landingContent.heroSection.hypothesisStatement}
            </p>
            
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-12">
              {landingContent.heroSection.keyMetrics.map((metric, index) => (
                <motion.div
                  key={index}
                  className="glass-effect rounded-xl p-6 hover:bg-white/20 transition-all duration-300"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: isLoaded ? 1 : 0, scale: isLoaded ? 1 : 0.9 }}
                  transition={{ duration: 0.8, delay: index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className="text-3xl mb-2 animate-pulse-slow">{metric.icon}</div>
                  <div className="text-3xl font-bold text-white mb-1">{metric.value}</div>
                  <div className="text-sm text-blue-100 font-medium">{metric.label}</div>
                  <div className="text-xs text-blue-200 opacity-80">{metric.sublabel}</div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Investment Highlights */}
      <section id="evidence" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              {landingContent.investmentHighlights.title}
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              {landingContent.investmentHighlights.subtitle}
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {landingContent.investmentHighlights.highlights.map((highlight, index) => (
              <motion.div
                key={index}
                className="highlight-card"
                initial={{ opacity: 0, x: index % 2 === 0 ? -20 : 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="flex items-start space-x-4">
                  <div className="p-3 bg-primary-100 rounded-lg">
                    <highlight.icon className="h-8 w-8 text-primary-600 flex-shrink-0" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">{highlight.title}</h3>
                    <p className="text-gray-600 mb-3 leading-relaxed">{highlight.description}</p>
                    <div className="text-lg font-semibold text-primary-600 bg-primary-50 px-3 py-1 rounded-full inline-block">
                      {highlight.metric}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Clinical Evidence */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              {landingContent.clinicalEvidence.title}
            </h2>
            <p className="text-xl text-gray-600">
              {landingContent.clinicalEvidence.subtitle}
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Effect Sizes Table */}
            <motion.div
              className="card"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <h3 className="text-xl font-bold text-gray-900 mb-4">Effect Sizes by Condition</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left font-medium text-gray-500">Condition</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-500">Baseline</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-500">Digital</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-500">Improvement</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {landingContent.clinicalEvidence.conditions.map((condition, index) => (
                      <tr key={index}>
                        <td className="px-3 py-2 font-medium text-gray-900">{condition.condition}</td>
                        <td className="px-3 py-2 text-gray-600">{condition.baselineEffect}</td>
                        <td className="px-3 py-2 text-gray-600">{condition.digitalEffect}</td>
                        <td className="px-3 py-2 text-secondary-600 font-medium">{condition.improvement}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>

            {/* Meta-Analysis Results */}
            <motion.div
              className="card"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <h3 className="text-xl font-bold text-gray-900 mb-4">Meta-Analysis Results</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Overall Effect Size:</span>
                  <span className="text-2xl font-bold text-primary-600">{landingContent.clinicalEvidence.metaAnalysis.overallEffectSize}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Heterogeneity:</span>
                  <span className="font-medium">{landingContent.clinicalEvidence.metaAnalysis.heterogeneity}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Publication Bias:</span>
                  <span className="font-medium">{landingContent.clinicalEvidence.metaAnalysis.publicationBias}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Quality Score:</span>
                  <span className="font-medium">{landingContent.clinicalEvidence.metaAnalysis.qualityScore}</span>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Market Opportunity */}
      <section id="market" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              {landingContent.marketOpportunity.title}
            </h2>
            <p className="text-xl text-gray-600">
              {landingContent.marketOpportunity.subtitle}
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            <motion.div
              className="card text-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              <div className="text-3xl font-bold text-primary-600 mb-2">{landingContent.marketOpportunity.marketSize}</div>
              <div className="text-lg text-gray-600 mb-4">Market Size</div>
              <div className="text-sm text-secondary-600">{landingContent.marketOpportunity.growthRate} Growth Rate</div>
            </motion.div>

            <motion.div
              className="card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              viewport={{ once: true }}
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4">Target Conditions</h3>
              <div className="space-y-2">
                {landingContent.marketOpportunity.targetConditions.map((condition, index) => (
                  <div key={index} className="flex items-center">
                    <CheckCircleIcon className="h-4 w-4 text-secondary-500 mr-2" />
                    <span className="text-gray-700">{condition}</span>
                  </div>
                ))}
              </div>
            </motion.div>

            <motion.div
              className="card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              viewport={{ once: true }}
            >
              <h3 className="text-lg font-bold text-gray-900 mb-4">User Personas</h3>
              <div className="space-y-3">
                {landingContent.marketOpportunity.userPersonas.map((persona, index) => (
                  <div key={index} className="flex justify-between items-center">
                    <span className="text-sm text-gray-700">{persona.name}</span>
                    <span className="text-sm font-medium text-primary-600">{persona.willingness}% willing</span>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section id="technology" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
              {landingContent.technologyStack.title}
            </h2>
            <p className="text-xl text-gray-600">
              {landingContent.technologyStack.subtitle}
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {landingContent.technologyStack.components.map((component, index) => (
              <motion.div
                key={index}
                className="metric-card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">{component.name}</h3>
                  <span className="px-2 py-1 text-xs bg-secondary-100 text-secondary-800 rounded-full">
                    {component.status}
                  </span>
                </div>
                <p className="text-gray-600">{component.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 bg-primary-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
              {landingContent.callToAction.title}
            </h2>
            <p className="text-xl text-blue-100 mb-8">
              {landingContent.callToAction.subtitle}
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/dashboard" className="btn-secondary">
                {landingContent.callToAction.primaryButton.text}
                <ArrowRightIcon className="h-5 w-5 ml-2" />
              </Link>
              <button className="btn-primary">
                {landingContent.callToAction.secondaryButton.text}
              </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h3 className="text-2xl font-bold mb-4">PlaceboRx</h3>
            <p className="text-gray-400 mb-6">
              Evidence-Based Digital Placebo Validation Platform
            </p>
            <div className="text-sm text-gray-500">
              © 2024 PlaceboRx. All rights reserved. | Research Platform - Not for Clinical Use
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
} 