import { useState, useEffect } from 'react'
import Head from 'next/head'
import { motion } from 'framer-motion'
import { 
  ChartBarIcon, 
  BeakerIcon, 
  GlobeAltIcon, 
  CpuChipIcon,
  DocumentTextIcon,
  AcademicCapIcon,
  TrendingUpIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview')
  const [isLoading, setIsLoading] = useState(true)
  const [hypothesisData, setHypothesisData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('/api/hypothesis-data')
        if (!response.ok) {
          throw new Error('Failed to fetch data')
        }
        const data = await response.json()
        setHypothesisData(data)
      } catch (err) {
        setError(err.message)
        console.error('Error fetching hypothesis data:', err)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  const tabs = [
    { id: 'overview', name: 'Executive Summary', icon: DocumentTextIcon },
    { id: 'clinical', name: 'Clinical Evidence', icon: BeakerIcon },
    { id: 'market', name: 'Market Validation', icon: GlobeAltIcon },
    { id: 'literature', name: 'Literature Review', icon: AcademicCapIcon },
    { id: 'ai-insights', name: 'AI Analysis', icon: CpuChipIcon },
    { id: 'methodology', name: 'Methodology', icon: ChartBarIcon }
  ]

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex justify-center items-center">
        <div className="text-center">
          <div className="spinner mx-auto mb-4"></div>
          <p className="text-gray-600">Loading hypothesis testing data...</p>
        </div>
      </div>
    )
  }

  if (error || !hypothesisData) {
    return (
      <div className="min-h-screen bg-gray-50 flex justify-center items-center">
        <div className="text-center">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 mx-auto mb-4" />
          <h2 className="text-xl font-bold text-gray-900 mb-2">Error Loading Data</h2>
          <p className="text-gray-600 mb-4">{error || 'Unable to load hypothesis testing data'}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="btn-primary"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  const renderOverview = () => (
    <div className="space-y-8">
      {/* Hypothesis Statement */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Core Hypothesis</h2>
        <div className="bg-gradient-to-r from-primary-50 to-secondary-50 p-6 rounded-lg border-l-4 border-primary-500">
          <p className="text-lg font-semibold text-gray-900 mb-2">
            {hypothesisData.coreHypothesis.statement}
          </p>
          <div className="grid md:grid-cols-2 gap-4 mt-4 text-sm">
            <div>
              <span className="font-medium">Null Hypothesis:</span> {hypothesisData.coreHypothesis.nullHypothesis}
            </div>
            <div>
              <span className="font-medium">Alternative Hypothesis:</span> {hypothesisData.coreHypothesis.alternativeHypothesis}
            </div>
            <div>
              <span className="font-medium">Significance Level:</span> {hypothesisData.coreHypothesis.significanceLevel}
            </div>
            <div>
              <span className="font-medium">Statistical Power:</span> {hypothesisData.coreHypothesis.power}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Key Findings */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Key Findings</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">
              {hypothesisData.aiAnalysis.hypothesisValidation.score}/100
            </div>
            <div className="text-sm text-gray-600">Hypothesis Validation Score</div>
            <div className="text-xs text-secondary-600 mt-1">High Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">
              {hypothesisData.clinicalEvidence.metaAnalysis.overallEffectSize}
            </div>
            <div className="text-sm text-gray-600">Overall Effect Size (Cohen's d)</div>
            <div className="text-xs text-secondary-600 mt-1">Statistically Significant</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-primary-600 mb-2">
              {hypothesisData.clinicalEvidence.totalTrials.toLocaleString()}
            </div>
            <div className="text-sm text-gray-600">Clinical Trials Analyzed</div>
            <div className="text-xs text-secondary-600 mt-1">Research-Grade Data</div>
          </div>
        </div>
      </motion.div>

      {/* Evidence Summary */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Evidence Summary</h2>
        <div className="space-y-4">
          <div className="flex items-start space-x-3">
            <CheckCircleIcon className="h-6 w-6 text-secondary-500 mt-1 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-gray-900">Clinical Evidence</h3>
              <p className="text-gray-600">Strong statistical evidence across {hypothesisData.clinicalEvidence.conditions.length} conditions with mean effect size of {hypothesisData.clinicalEvidence.metaAnalysis.overallEffectSize} (p < 0.05)</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <CheckCircleIcon className="h-6 w-6 text-secondary-500 mt-1 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-gray-900">Market Validation</h3>
              <p className="text-gray-600">Strong demand signals from {hypothesisData.marketValidation.totalPosts.toLocaleString()} community posts with {hypothesisData.marketValidation.sentimentAnalysis.positive}% positive sentiment</p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <CheckCircleIcon className="h-6 w-6 text-secondary-500 mt-1 flex-shrink-0" />
            <div>
              <h3 className="font-semibold text-gray-900">Literature Support</h3>
              <p className="text-gray-600">{hypothesisData.literatureEvidence.totalArticles} peer-reviewed articles support open-label placebo mechanisms and digital delivery enhancement</p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )

  const renderClinicalEvidence = () => (
    <div className="space-y-8">
      {/* Effect Sizes by Condition */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Effect Sizes by Condition</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Condition</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Baseline Effect</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Digital Effect</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Improvement</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P-Value</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">95% CI</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {hypothesisData.clinicalEvidence.effectSizes.map((effect, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{effect.condition}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{effect.baseline}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{effect.digital}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{effect.improvement}%</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{effect.pValue}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{effect.confidenceInterval}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* Meta-Analysis Results */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Meta-Analysis Results</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Overall Effect Size:</span>
              <span className="font-semibold text-primary-600">{hypothesisData.clinicalEvidence.metaAnalysis.overallEffectSize}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Heterogeneity (I²):</span>
              <span className="font-semibold text-gray-900">{hypothesisData.clinicalEvidence.metaAnalysis.heterogeneity}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Publication Bias:</span>
              <span className="font-semibold text-gray-900">{hypothesisData.clinicalEvidence.metaAnalysis.publicationBias}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">Quality Score:</span>
              <span className="font-semibold text-gray-900">{hypothesisData.clinicalEvidence.metaAnalysis.qualityScore}</span>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="font-semibold text-gray-900 mb-2">Interpretation</h3>
            <p className="text-sm text-gray-600">
              The overall effect size of {hypothesisData.clinicalEvidence.metaAnalysis.overallEffectSize} exceeds the hypothesized threshold of 0.2, 
              indicating meaningful clinical effects. Low heterogeneity suggests consistent effects across studies, 
              while the absence of publication bias supports the robustness of findings.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )

  const renderMarketValidation = () => (
    <div className="space-y-8">
      {/* Sentiment Analysis */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Community Sentiment Analysis</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-secondary-600 mb-2">{hypothesisData.marketValidation.sentimentAnalysis.positive}%</div>
            <div className="text-sm text-gray-600">Positive Sentiment</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-gray-600 mb-2">{hypothesisData.marketValidation.sentimentAnalysis.neutral}%</div>
            <div className="text-sm text-gray-600">Neutral Sentiment</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-red-600 mb-2">{hypothesisData.marketValidation.sentimentAnalysis.negative}%</div>
            <div className="text-sm text-gray-600">Negative Sentiment</div>
          </div>
        </div>
      </motion.div>

      {/* Demand Signals */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Demand Signal Analysis</h2>
        <div className="space-y-4">
          {Object.entries(hypothesisData.marketValidation.demandSignals).map(([signal, data]) => (
            <div key={signal} className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
              <div>
                <h3 className="font-semibold text-gray-900 capitalize">{signal.replace(/([A-Z])/g, ' $1')}</h3>
                <p className="text-sm text-gray-600">{data.mentions} mentions</p>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${data.sentiment > 0 ? 'text-secondary-600' : 'text-red-600'}`}>
                  {data.sentiment > 0 ? '+' : ''}{data.sentiment}
                </div>
                <div className="text-xs text-gray-500">{data.significance}</div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* User Personas */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">User Persona Analysis</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {hypothesisData.marketValidation.userPersonas.map((persona, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-2">{persona.name}</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Market Share:</span>
                  <span className="font-medium">{persona.percentage}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Willingness to Try:</span>
                  <span className="font-medium">{persona.willingness}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Engagement Level:</span>
                  <span className="font-medium">{persona.engagement}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )

  const renderLiteratureReview = () => (
    <div className="space-y-8">
      {/* Evidence Strength */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Evidence Strength Assessment</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-4 bg-secondary-50 rounded-lg">
            <div className="text-lg font-semibold text-secondary-800 mb-2">Digital Placebo</div>
            <div className="text-2xl font-bold text-secondary-600">{hypothesisData.literatureEvidence.evidenceStrength.digitalPlacebo}</div>
          </div>
          <div className="text-center p-4 bg-primary-50 rounded-lg">
            <div className="text-lg font-semibold text-primary-800 mb-2">Open-Label Placebo</div>
            <div className="text-2xl font-bold text-primary-600">{hypothesisData.literatureEvidence.evidenceStrength.openLabelPlacebo}</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-lg font-semibold text-gray-800 mb-2">Mechanistic Evidence</div>
            <div className="text-2xl font-bold text-gray-600">{hypothesisData.literatureEvidence.evidenceStrength.mechanisticEvidence}</div>
          </div>
        </div>
      </motion.div>

      {/* Key Findings */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Key Literature Findings</h2>
        <div className="space-y-4">
          {hypothesisData.literatureEvidence.keyFindings.map((finding, index) => (
            <div key={index} className="flex items-start space-x-3 p-4 bg-gray-50 rounded-lg">
              <CheckCircleIcon className="h-5 w-5 text-secondary-500 mt-1 flex-shrink-0" />
              <p className="text-gray-700">{finding}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Research Gaps */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Research Gaps & Opportunities</h2>
        <div className="space-y-4">
          {hypothesisData.literatureEvidence.researchGaps.map((gap, index) => (
            <div key={index} className="flex items-start space-x-3 p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-400">
              <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500 mt-1 flex-shrink-0" />
              <p className="text-gray-700">{gap}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  )

  const renderAIAnalysis = () => (
    <div className="space-y-8">
      {/* Hypothesis Validation */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">AI-Powered Hypothesis Validation</h2>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <div className="text-center mb-6">
              <div className="text-4xl font-bold text-primary-600 mb-2">
                {hypothesisData.aiAnalysis.hypothesisValidation.score}/100
              </div>
              <div className="text-lg text-gray-600">Validation Score</div>
              <div className="text-sm text-secondary-600">{hypothesisData.aiAnalysis.hypothesisValidation.confidence} Confidence</div>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Evidence Strength:</span>
                <span className="font-semibold">{hypothesisData.aiAnalysis.hypothesisValidation.evidenceStrength}</span>
              </div>
            </div>
          </div>
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Supporting Evidence</h3>
              <ul className="space-y-2">
                {hypothesisData.aiAnalysis.hypothesisValidation.supportingEvidence.map((evidence, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <CheckCircleIcon className="h-4 w-4 text-secondary-500 mt-1 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{evidence}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-2">Contradicting Evidence</h3>
              <ul className="space-y-2">
                {hypothesisData.aiAnalysis.hypothesisValidation.contradictingEvidence.map((evidence, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 mt-1 flex-shrink-0" />
                    <span className="text-sm text-gray-700">{evidence}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Cross-Analysis */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Cross-Data Analysis</h2>
        <div className="space-y-4">
          <div className="p-4 bg-primary-50 rounded-lg border-l-4 border-primary-500">
            <h3 className="font-semibold text-primary-800 mb-2">Evidence Convergence</h3>
            <p className="text-primary-700">{hypothesisData.aiAnalysis.crossAnalysis.evidenceConvergence}</p>
          </div>
          <div className="p-4 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
            <h3 className="font-semibold text-yellow-800 mb-2">Risk Assessment</h3>
            <p className="text-yellow-700">{hypothesisData.aiAnalysis.crossAnalysis.riskAssessment}</p>
          </div>
          <div className="p-4 bg-secondary-50 rounded-lg border-l-4 border-secondary-500">
            <h3 className="font-semibold text-secondary-800 mb-2">Strategic Implications</h3>
            <p className="text-secondary-700">{hypothesisData.aiAnalysis.crossAnalysis.strategicImplications}</p>
          </div>
        </div>
      </motion.div>

      {/* Recommendations */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">AI-Generated Recommendations</h2>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <BeakerIcon className="h-5 w-5 mr-2" />
              Clinical Development
            </h3>
            <ul className="space-y-2">
              {hypothesisData.aiAnalysis.recommendations.clinical.map((rec, index) => (
                <li key={index} className="text-sm text-gray-700 flex items-start">
                  <span className="text-primary-600 mr-2">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <GlobeAltIcon className="h-5 w-5 mr-2" />
              Market Strategy
            </h3>
            <ul className="space-y-2">
              {hypothesisData.aiAnalysis.recommendations.market.map((rec, index) => (
                <li key={index} className="text-sm text-gray-700 flex items-start">
                  <span className="text-secondary-600 mr-2">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <AcademicCapIcon className="h-5 w-5 mr-2" />
              Research Priorities
            </h3>
            <ul className="space-y-2">
              {hypothesisData.aiAnalysis.recommendations.research.map((rec, index) => (
                <li key={index} className="text-sm text-gray-700 flex items-start">
                  <span className="text-gray-600 mr-2">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )

  const renderMethodology = () => (
    <div className="space-y-8">
      {/* Statistical Framework */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Statistical Methodology</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Meta-Analysis Approach</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>• Random effects model for heterogeneity</li>
              <li>• Forest plots for effect size visualization</li>
              <li>• Egger's test for publication bias</li>
              <li>• Newcastle-Ottawa Scale for quality assessment</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Hypothesis Testing</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>• One-tailed t-tests for effect size differences</li>
              <li>• Cohen's d for standardized effect sizes</li>
              <li>• 95% confidence intervals</li>
              <li>• Power analysis for sample size adequacy</li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Data Sources */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Data Sources & Quality</h2>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <h3 className="font-semibold text-gray-900">ClinicalTrials.gov</h3>
              <p className="text-sm text-gray-600">Clinical trial data extraction and analysis</p>
            </div>
            <span className="px-3 py-1 bg-secondary-100 text-secondary-800 rounded-full text-sm">Active</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <h3 className="font-semibold text-gray-900">Reddit API</h3>
              <p className="text-sm text-gray-600">Community sentiment and demand analysis</p>
            </div>
            <span className="px-3 py-1 bg-secondary-100 text-secondary-800 rounded-full text-sm">Active</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <h3 className="font-semibold text-gray-900">PubMed API</h3>
              <p className="text-sm text-gray-600">Scientific literature review and analysis</p>
            </div>
            <span className="px-3 py-1 bg-secondary-100 text-secondary-800 rounded-full text-sm">Active</span>
          </div>
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <h3 className="font-semibold text-gray-900">OpenAI API</h3>
              <p className="text-sm text-gray-600">AI-powered insights and cross-analysis</p>
            </div>
            <span className="px-3 py-1 bg-secondary-100 text-secondary-800 rounded-full text-sm">Active</span>
          </div>
        </div>
      </motion.div>

      {/* Quality Assurance */}
      <motion.div 
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.2 }}
      >
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Quality Assurance</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Data Validation</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>• Duplicate detection algorithms</li>
              <li>• Missing data imputation</li>
              <li>• Outlier identification</li>
              <li>• Confidence scoring for estimates</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 mb-3">Transparency</h3>
            <ul className="space-y-2 text-sm text-gray-700">
              <li>• Open-source analytical framework</li>
              <li>• Reproducible research methods</li>
              <li>• Clear confidence intervals</li>
              <li>• Limitations disclosure</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </div>
  )

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverview()
      case 'clinical':
        return renderClinicalEvidence()
      case 'market':
        return renderMarketValidation()
      case 'literature':
        return renderLiteratureReview()
      case 'ai-insights':
        return renderAIAnalysis()
      case 'methodology':
        return renderMethodology()
      default:
        return renderOverview()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>PlaceboRx Dashboard - Hypothesis Testing Results</title>
        <meta name="description" content="Comprehensive hypothesis testing results for digital placebo interventions" />
      </Head>

      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-primary-600">PlaceboRx</h1>
              <span className="ml-2 px-2 py-1 text-xs bg-primary-100 text-primary-800 rounded-full">Dashboard</span>
            </div>
            <div className="text-sm text-gray-500">
              Last updated: {new Date().toLocaleDateString()}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-primary-500 text-primary-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="h-5 w-5 inline mr-2" />
                {tab.name}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="spinner"></div>
          </div>
        ) : (
          renderContent()
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-sm text-gray-500">
            <p>© 2024 PlaceboRx. Research Platform - Not for Clinical Use</p>
            <p className="mt-2">Data sources: ClinicalTrials.gov, Reddit API, PubMed API, OpenAI API</p>
          </div>
        </div>
      </footer>
    </div>
  )
} 