import { useState, useEffect, useRef } from 'react'
import EmotionVisualizer from './components/EmotionVisualizer'
import { motion } from 'framer-motion'

function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  // State
  const [emotion, setEmotion] = useState('Neutral')
  const [confidence, setConfidence] = useState(0.0)
  const [isStreamActive, setIsStreamActive] = useState(false)
  const [allScores, setAllScores] = useState({})
  const [faceDetected, setFaceDetected] = useState(false)

  // Model Management
  const [models, setModels] = useState([])
  const [currentModel, setCurrentModel] = useState('')

  useEffect(() => {
    // Initialize Camera
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          setIsStreamActive(true)
        }
      } catch (err) {
        console.error("Error accessing webcam:", err)
      }
    }
    startVideo()

    // Fetch Models
    const fetchModels = async () => {
      try {
        const res = await fetch('http://localhost:5001/models')
        const data = await res.json()
        setModels(data.models || [])
        setCurrentModel(data.current || 'default')
      } catch (err) {
        console.error("Failed to fetch models", err)
      }
    }
    fetchModels()
  }, [])

  const handleModelChange = async (e) => {
    const path = e.target.value
    try {
      const res = await fetch('http://localhost:5001/load_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      })
      const data = await res.json()
      if (data.success) {
        setCurrentModel(data.model)
      }
    } catch (err) {
      console.error("Failed to switch model", err)
    }
  }

  useEffect(() => {
    let intervalId
    const captureAndPredict = async () => {
      if (!videoRef.current || !canvasRef.current || !isStreamActive) return

      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      const imageBase64 = canvas.toDataURL('image/jpeg', 0.8)

      try {
        const response = await fetch('http://localhost:5001/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: imageBase64 })
        })

        const data = await response.json()

        // Update Face Detection Status
        setFaceDetected(data.face_detected)

        if (data.emotion) {
          setEmotion(data.emotion)
          setConfidence(data.confidence)
          setAllScores(data.all_scores || {})
        }
      } catch (err) {
        console.error(err)
      }
    }

    if (isStreamActive) {
      intervalId = setInterval(captureAndPredict, 200)
    }
    return () => clearInterval(intervalId)
  }, [isStreamActive])

  return (
    <div className="relative w-full h-screen flex items-center justify-center p-12">
      <EmotionVisualizer emotion={emotion} />

      <main className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-16 items-center">

        {/* Left: The Frame */}
        <div className="relative group">
          <div className={`photo-frame bg-white p-4 rotate-1 rounded-sm transition-all duration-700 hover:rotate-0 ${!faceDetected ? 'opacity-80' : ''}`}>
            <div className="relative overflow-hidden w-full h-auto">
              <video
                ref={videoRef}
                autoPlay
                muted
                className="w-full h-auto object-cover grayscale-[20%] contrast-110"
                style={{ transform: 'scaleX(-1)' }}
              />

              {/* Face Not Detected Overlay */}
              {!faceDetected && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/10 backdrop-blur-[2px]">
                  <div className="px-4 py-2 bg-white/80 text-charcoal font-sans text-xs tracking-widest uppercase border border-charcoal/10">
                    Searching for Subject...
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Metadata Label & Selector */}
          <div className="absolute -bottom-16 left-0 w-full flex items-center justify-between">
            <div className="font-sans text-xs text-stone tracking-widest uppercase">
              Fig 1. Live Subject Analysis
            </div>

            {/* Model Selector */}
            <select
              value={models.find(m => m.name === currentModel)?.path || ''}
              onChange={handleModelChange}
              className="font-sans text-xs text-stone bg-transparent border-none outline-none cursor-pointer tracking-widest uppercase hover:text-accent transition-colors text-right appearance-none"
              style={{ textAlignLast: 'right' }}
            >
              <option value="" disabled>Select Model</option>
              {models.map((m, idx) => (
                <option key={idx} value={m.path}>
                  Model: {m.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Right: The Data */}
        <div className="flex flex-col space-y-8">
          <div>
            <h2 className="font-serif text-6xl text-charcoal mb-2 transition-all duration-500">
              {emotion}
            </h2>
            <div className="h-0.5 w-12 bg-charcoal/20 my-4"></div>
            <p className="font-sans text-stone text-sm tracking-[0.2em] uppercase">
              {faceDetected ? `Confidence: ${Math.round(confidence * 100)}%` : 'Waiting for input...'}
            </p>
          </div>

          <div className={`space-y-4 transition-opacity duration-500 ${faceDetected ? 'opacity-100' : 'opacity-30 blur-sm'}`}>
            {['Happy', 'Sad', 'Neutral'].map((e) => (
              <div key={e} className="flex items-center justify-between group">
                <span className={`font-serif text-lg ${e === emotion ? 'text-charcoal' : 'text-stone/40'}`}>
                  {e}
                </span>
                <div className="flex items-center gap-4">
                  <span className="font-sans text-xs text-stone">
                    {allScores[e] ? Math.round(allScores[e] * 100) : 0}%
                  </span>
                  <div className="w-24 h-px bg-stone/20 overflow-hidden">
                    <motion.div
                      className={`h-full ${e === emotion ? 'bg-charcoal' : 'bg-stone/20'}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${(allScores[e] || 0) * 100}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="pt-12">
            <p className="font-serif text-stone/60 italic text-sm">
              "The face is the mirror of the mind, and eyes without speaking confess the secrets of the heart."
            </p>
          </div>
        </div>

      </main>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default App
