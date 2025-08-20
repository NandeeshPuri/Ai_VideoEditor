'use client'

import { useState, useRef } from 'react'
import VideoUploader from './components/VideoUploader'
import FeatureSelector from './components/FeatureSelector'
import ProcessingStatus from './components/ProcessingStatus'
import VideoPreview from './components/VideoPreview'

export default function Home() {
    const [uploadId, setUploadId] = useState<string | null>(null)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
    const [selectedStyle, setSelectedStyle] = useState<string>('cartoon')
    const [isProcessing, setIsProcessing] = useState(false)
    const [processingStatus, setProcessingStatus] = useState<any>(null)

    const handleUploadSuccess = (id: string, file: File) => {
        setUploadId(id)
        setUploadedFile(file)
        setIsProcessing(false)
        setProcessingStatus(null)
    }

    const handleFeatureToggle = (feature: string) => {
        setSelectedFeatures(prev =>
            prev.includes(feature)
                ? prev.filter(f => f !== feature)
                : [...prev, feature]
        )
    }

    const handleProcess = async () => {
        if (!uploadId || selectedFeatures.length === 0) return

        setIsProcessing(true)
        setProcessingStatus({ status: 'starting', progress: 0 })

        // Process each selected feature individually
        for (const feature of selectedFeatures) {
            try {
                let url = `http://localhost:8000/process/${feature}?upload_id=${uploadId}`
                
                // Add style parameter for style filters
                if (feature === 'style') {
                    url += `&style=${selectedStyle}`
                }

                const response = await fetch(url, {
                    method: 'POST',
                })

                if (!response.ok) {
                    throw new Error(`Failed to process ${feature}`)
                }
            } catch (error) {
                console.error(`Error processing ${feature}:`, error)
                const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred'
                setProcessingStatus({ status: 'error', error: errorMessage })
                setIsProcessing(false)
                return
            }
        }

        // Start polling for status
        pollProcessingStatus(uploadId)
    }

    const pollProcessingStatus = async (id: string) => {
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`http://localhost:8000/status/${id}`)
                if (response.ok) {
                    const status = await response.json()
                    setProcessingStatus(status)

                    if (status.status === 'completed' || status.status === 'error') {
                        clearInterval(interval)
                        setIsProcessing(false)
                    }
                }
            } catch (error) {
                console.error('Error polling status:', error)
            }
        }, 2000)
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
            <div className="container mx-auto px-4 py-8">
                <div className="text-center mb-8">
                    <h1 className="text-4xl font-bold text-gray-900 mb-4">
                        AI Video Editor MVP
                    </h1>
                    <p className="text-lg text-gray-600">
                        Transform your videos with AI-powered editing features
                    </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column - Upload & Preview */}
                    <div className="lg:col-span-2 space-y-6">
                        <VideoUploader
                            onUploadSuccess={handleUploadSuccess}
                            disabled={isProcessing}
                        />

                        {uploadedFile && (
                            <VideoPreview
                                file={uploadedFile}
                                uploadId={uploadId}
                                processingStatus={processingStatus}
                            />
                        )}
                    </div>

                    {/* Right Column - Features & Controls */}
                    <div className="space-y-6">
                        <FeatureSelector
                            selectedFeatures={selectedFeatures}
                            onFeatureToggle={handleFeatureToggle}
                            selectedStyle={selectedStyle}
                            onStyleChange={setSelectedStyle}
                            disabled={isProcessing}
                        />

                        {uploadId && selectedFeatures.length > 0 && (
                            <div className="bg-white rounded-lg p-6 shadow-lg">
                                <button
                                    onClick={handleProcess}
                                    disabled={isProcessing}
                                    className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                                >
                                    {isProcessing ? 'Processing...' : 'Process Video'}
                                </button>
                            </div>
                        )}

                        {processingStatus && (
                            <ProcessingStatus
                                status={processingStatus}
                                uploadId={uploadId}
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
