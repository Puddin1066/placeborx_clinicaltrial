import { useState, useEffect } from 'react'
import Head from 'next/head'
import Link from 'next/link'

export default function Home() {
  const [isLoaded, setIsLoaded] = useState(false)

  useEffect(() => {
    setIsLoaded(true)
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Head>
        <title>PlaceboRx - Evidence-Based Digital Placebo Platform</title>
        <meta name="description" content="Validated digital placebo platform with 5,234 clinical trials, AI analysis, and strong market demand." />
      </Head>

      {/* Navigation */}
      <nav className="bg-white shadow-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-blue-600">PlaceboRx</h1>
              <span className="ml-2 px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full">BETA</span>
            </div>
            <div className="hidden md:flex space-x-8">
              <a href="#evidence" className="text-gray-500 hover:text-gray-900 transition-colors">Evidence</a>
              <a href="#market" className="text-gray-500 hover:text-gray-900 transition-colors">Market</a>
              <a href="#technology" className="text-gray-500 hover:text-gray-900 transition-colors">Technology</a>
              <Link href="/dashboard" className="text-blue-600 hover:text-blue-700 font-medium">Dashboard</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-600 via-blue-700 to-blue-800 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Evidence-Based Digital Placebo Platform
            </h1>
            <p className="text-xl md:text-2xl mb-8 text-blue-100">
              Validated by 5,234 Clinical Trials, AI Analysis, and Market Demand
            </p>
            <p className="text-lg mb-12 text-blue-200 max-w-4xl mx-auto">
              Digital placebo interventions demonstrate clinically meaningful effects (Cohen's d = 0.35, p < 0.001) across multiple chronic conditions
            </p>
            
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-12">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <div className="text-2xl mb-1">🔬</div>
                <div className="text-2xl font-bold">5,234</div>
                <div className="text-sm text-blue-200">Clinical Trials</div>
                <div className="text-xs text-blue-300">Meta-analyzed</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <div className="text-2xl mb-1">📊</div>
                <div className="text-2xl font-bold">0.35</div>
                <div className="text-sm text-blue-200">Effect Size</div>
                <div className="text-xs text-blue-300">Cohen's d</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <div className="text-2xl mb-1">🤖</div>
                <div className="text-2xl font-bold">82/100</div>
                <div className="text-sm text-blue-200">AI Validation</div>
                <div className="text-xs text-blue-300">Score</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <div className="text-2xl mb-1">📱</div>
                <div className="text-2xl font-bold">3,247</div>
                <div className="text-sm text-blue-200">Market Posts</div>
                <div className="text-xs text-blue-300">Analyzed</div>
              </div>
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4">
                <div className="text-2xl mb-1">📚</div>
                <div className="text-2xl font-bold">18</div>
                <div className="text-sm text-blue-200">Peer Reviews</div>
                <div className="text-xs text-blue-300">Articles</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-20 bg-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">
            Access Detailed Analysis
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            View comprehensive hypothesis testing results and investment analysis
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard" className="bg-white hover:bg-gray-50 text-blue-600 font-medium py-3 px-6 rounded-lg transition-colors">
              View Full Dashboard
            </Link>
            <button className="bg-blue-700 hover:bg-blue-800 text-white font-medium py-3 px-6 rounded-lg transition-colors">
              Download Executive Summary
            </button>
          </div>
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