'use client'

import { useState, useRef } from 'react'
import VideoUploader from './components/VideoUploader'
import FeatureSelector from './components/FeatureSelector'
import ProcessingStatus from './components/ProcessingStatus'
import VideoPreview from './components/VideoPreview'
import VoiceTranslationSelector from './components/VoiceTranslationSelector'
import VideoCompilationSelector from './components/VideoCompilationSelector'
import ObjectRemovalSelector from './components/ObjectRemovalSelector'

interface BoundingBox {
    id: string
    x: number
    y: number
    width: number
    height: number
    label?: string
}

export default function Home() {
    const [mode, setMode] = useState<'single' | 'compilation'>('single')
    const [uploadId, setUploadId] = useState<string | null>(null)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)
    const [uploadedVideos, setUploadedVideos] = useState<{ id: string, file: File, name: string }[]>([])
    const [selectedVideos, setSelectedVideos] = useState<string[]>([])
    const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])
    const [selectedStyle, setSelectedStyle] = useState<string>('cartoon')
    const [targetLanguage, setTargetLanguage] = useState<string>('es')
    const [voiceType, setVoiceType] = useState<string>('female')
    const [addSubtitles, setAddSubtitles] = useState<boolean>(true)
    const [maxDuration, setMaxDuration] = useState<number>(60)
    const [transitionStyle, setTransitionStyle] = useState<string>('fade')
    const [selectedPreset, setSelectedPreset] = useState<string>('youtube_shorts')
    const [isProcessing, setIsProcessing] = useState(false)
    const [processingStatus, setProcessingStatus] = useState<any>(null)
    const [objectRemovalBoxes, setObjectRemovalBoxes] = useState<BoundingBox[]>([])

    const handleUploadSuccess = (id: string, file: File) => {
        setUploadId(id)
        setUploadedFile(file)

        // Add to uploaded videos list for compilation
        setUploadedVideos(prev => [...prev, { id, file, name: file.name }])

        setIsProcessing(false)
        setProcessingStatus(null)
        setObjectRemovalBoxes([]) // Reset object removal boxes
    }

    const handleModeChange = (newMode: 'single' | 'compilation') => {
        setMode(newMode)
        // Clear selections when switching modes
        setSelectedFeatures([])
        setSelectedVideos([])
        setProcessingStatus(null)
        setObjectRemovalBoxes([])
    }

    const handleFeatureToggle = (feature: string) => {
        setSelectedFeatures(prev =>
            prev.includes(feature)
                ? prev.filter(f => f !== feature)
                : [...prev, feature]
        )

        // Clear object removal boxes when feature is deselected
        if (feature === 'object-remove' && selectedFeatures.includes(feature)) {
            setObjectRemovalBoxes([])
        }
    }

    const handleProcess = async () => {
        // For compilation mode, need multiple videos
        if (mode === 'compilation') {
            if (selectedVideos.length < 2) {
                alert('Please select at least 2 videos for compilation')
                return
            }
        } else {
            // For single mode, need single video and features
            if (!uploadId) return
            if (selectedFeatures.length === 0) return

            // Check if object removal is selected but no boxes are marked
            if (selectedFeatures.includes('object-remove') && objectRemovalBoxes.length === 0) {
                alert('Please mark objects to remove before processing')
                return
            }
        }

        setIsProcessing(true)
        setProcessingStatus({ status: 'starting', progress: 0 })

        try {
            if (mode === 'compilation') {
                // Single compilation request
                const url = `http://localhost:8000/process/video-compilation?upload_ids=${selectedVideos.join(',')}&max_duration=${maxDuration}&transition_style=${transitionStyle}&preset=${selectedPreset}`

                const response = await fetch(url, {
                    method: 'POST',
                })

                if (!response.ok) {
                    throw new Error('Failed to process video compilation')
                }
            } else {
                // Process each selected feature individually for single mode
                for (const feature of selectedFeatures) {
                    let url = `http://localhost:8000/process/${feature}?upload_id=${uploadId}`

                    // Add style parameter for style filters
                    if (feature === 'style') {
                        url += `&style=${selectedStyle}`
                    }

                    // Add voice translation parameters
                    if (feature === 'voice-translate') {
                        url += `&target_language=${targetLanguage}&voice_type=${voiceType}&add_subtitles=${addSubtitles}`
                    }

                    // Add object removal parameters
                    if (feature === 'object-remove') {
                        const boxesParam = objectRemovalBoxes.map(box =>
                            `${box.x},${box.y},${box.x + box.width},${box.y + box.height}`
                        ).join(';')
                        url += `&bounding_boxes=${encodeURIComponent(boxesParam)}`
                    }

                    const response = await fetch(url, {
                        method: 'POST',
                    })

                    if (!response.ok) {
                        throw new Error(`Failed to process ${feature}`)
                    }
                }
            }
        } catch (error) {
            console.error('Error processing:', error)
            const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred'
            setProcessingStatus({ status: 'error', error: errorMessage })
            setIsProcessing(false)
            return
        }

        // Start polling for status
        const statusId = mode === 'compilation' ? selectedVideos[0] : uploadId
        pollProcessingStatus(statusId!)
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
                    <p className="text-lg text-gray-600 mb-6">
                        Transform your videos with AI-powered editing features
                    </p>

                    {/* Mode Toggle */}
                    <div className="flex justify-center mb-6">
                        <div className="bg-white rounded-lg p-1 shadow-lg border border-gray-200">
                            <div className="flex">
                                <button
                                    onClick={() => handleModeChange('single')}
                                    className={`px-6 py-3 rounded-md font-medium transition-all duration-200 ${mode === 'single'
                                        ? 'bg-blue-600 text-white shadow-md'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    🎬 Single Video Editing
                                </button>
                                <button
                                    onClick={() => handleModeChange('compilation')}
                                    className={`px-6 py-3 rounded-md font-medium transition-all duration-200 ${mode === 'compilation'
                                        ? 'bg-purple-600 text-white shadow-md'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                                        }`}
                                >
                                    📱 Social Media Compilation
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Mode Description */}
                    <div className="text-center">
                        {mode === 'single' ? (
                            <p className="text-sm text-gray-600">
                                Upload a single video and apply AI editing features
                            </p>
                        ) : (
                            <p className="text-sm text-gray-600">
                                Upload up to 5 videos and create AI-powered social media compilations
                            </p>
                        )}
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                    {/* Left Column - Upload & Preview */}
                    <div className="lg:col-span-2 space-y-6">
                        <VideoUploader
                            onUploadSuccess={handleUploadSuccess}
                            disabled={isProcessing}
                        />

                        {mode === 'single' && uploadedFile && (
                            <VideoPreview
                                file={uploadedFile}
                                uploadId={uploadId}
                                processingStatus={processingStatus}
                            />
                        )}

                        {/* Object Removal Selector - Only show when feature is selected */}
                        {mode === 'single' && selectedFeatures.includes('object-remove') && uploadedFile && (
                            <ObjectRemovalSelector
                                videoFile={uploadedFile}
                                onBoundingBoxesChange={setObjectRemovalBoxes}
                                disabled={isProcessing}
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
                            mode={mode}
                        />

                        {/* Voice Translation Settings - Only in Single Mode */}
                        {mode === 'single' && selectedFeatures.includes('voice-translate') && (
                            <VoiceTranslationSelector
                                onLanguageChange={setTargetLanguage}
                                onVoiceTypeChange={setVoiceType}
                                onSubtitleChange={setAddSubtitles}
                                disabled={isProcessing}
                            />
                        )}

                        {/* Compilation Settings - Only in Compilation Mode */}
                        {mode === 'compilation' && (
                            <VideoCompilationSelector
                                onUploadIdsChange={setSelectedVideos}
                                onMaxDurationChange={setMaxDuration}
                                onTransitionStyleChange={setTransitionStyle}
                                onPresetChange={setSelectedPreset}
                                disabled={isProcessing}
                                availableUploadIds={uploadedVideos.map(v => v.id)}
                            />
                        )}

                        {((mode === 'single' && uploadId && selectedFeatures.length > 0) || (mode === 'compilation' && selectedVideos.length >= 2 && selectedVideos.length <= 5)) && (
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
