import '../styles/globals.css'
import Head from 'next/head'

export default function App({ Component, pageProps }) {
  return (
    <>
      <Head>
        <title>PlaceboRx - AI-Powered Digital Placebo Validation Platform</title>
        <meta name="description" content="Advanced analytics and machine learning for evidence-based digital therapeutics validation" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        
        {/* Open Graph */}
        <meta property="og:title" content="PlaceboRx - Digital Placebo Research Platform" />
        <meta property="og:description" content="Professional-grade analytics for digital placebo hypothesis testing" />
        <meta property="og:type" content="website" />
        <meta property="og:url" content="https://placeborx.vercel.app" />
        
        {/* Twitter */}
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="PlaceboRx - Digital Placebo Validation Platform" />
        <meta name="twitter:description" content="AI-powered analytics for digital therapeutics validation" />
        
        {/* Fonts */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      </Head>
      <Component {...pageProps} />
    </>
  )
} 