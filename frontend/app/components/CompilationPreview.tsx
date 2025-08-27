'use client'

import { useRef, useEffect, useState } from 'react'

interface CompilationPreviewProps {
    uploadId: string | null
    processingStatus: any
    selectedVideos: string[]
}

export default function CompilationPreview({ uploadId, processingStatus, selectedVideos }: CompilationPreviewProps) {
    const processedVideoRef = useRef<HTMLVideoElement>(null)
    const [processedVideoUrl, setProcessedVideoUrl] = useState<string>('')
    const [isDownloading, setIsDownloading] = useState(false)
    const [isLoadingVideo, setIsLoadingVideo] = useState(false)

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed':
                return 'text-green-600 bg-green-100'
            case 'processing':
                return 'text-blue-600 bg-blue-100'
            case 'error':
                return 'text-red-600 bg-red-100'
            default:
                return 'text-gray-600 bg-gray-100'
        }
    }

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'completed':
                return '✅'
            case 'processing':
                return '⏳'
            case 'error':
                return '❌'
            default:
                return '🎬'
        }
    }

    const handleDownload = async () => {
        if (!uploadId || !processingStatus?.output_path) return

        setIsDownloading(true)
        try {
            const response = await fetch(`http://localhost:8000/download/by-upload/${uploadId}`)
            if (response.ok) {
                const blob = await response.blob()
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `compilation_${uploadId}.mp4`
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
                URL.revokeObjectURL(url)
            } else {
                console.error('Download failed')
            }
        } catch (error) {
            console.error('Download error:', error)
        } finally {
            setIsDownloading(false)
        }
    }

    const loadProcessedVideo = async () => {
        if (!uploadId) return

        setIsLoadingVideo(true)
        try {
            // Try to load the video from the streaming endpoint first (better for preview)
            const response = await fetch(`http://localhost:8000/video/${uploadId}`)
            if (response.ok) {
                const blob = await response.blob()
                const url = URL.createObjectURL(blob)
                setProcessedVideoUrl(url)
            } else {
                // Fallback to download endpoint
                const downloadResponse = await fetch(`http://localhost:8000/download/by-upload/${uploadId}`)
                if (downloadResponse.ok) {
                    const blob = await downloadResponse.blob()
                    const url = URL.createObjectURL(blob)
                    setProcessedVideoUrl(url)
                } else {
                    console.error('Failed to load video:', response.status, response.statusText)
                }
            }
        } catch (error) {
            console.error('Failed to load processed video:', error)
        } finally {
            setIsLoadingVideo(false)
        }
    }

    useEffect(() => {
        if (processingStatus?.status === 'completed') {
            // Add a small delay to ensure the file is written
            setTimeout(() => {
                loadProcessedVideo()
            }, 1000)
        }
    }, [processingStatus?.status, uploadId])

    useEffect(() => {
        // Clear processed video URL when processing starts
        if (processingStatus?.status === 'processing') {
            setProcessedVideoUrl('')
        }
    }, [processingStatus?.status])

    return (
        <div className="bg-white rounded-lg p-6 shadow-lg">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <span className="mr-2">🎬</span>
                Compilation Preview
            </h2>

            <div className="space-y-6">
                {/* Compilation Info */}
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4">
                    <h3 className="font-medium text-gray-900 mb-2">Compilation Details</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <span className="font-medium text-gray-700">Videos Used:</span>
                            <p className="text-gray-900">{selectedVideos.length} videos</p>
                        </div>
                        <div>
                            <span className="font-medium text-gray-700">Upload ID:</span>
                            <p className="text-gray-900 font-mono text-xs">{uploadId}</p>
                        </div>
                        {processingStatus?.clips_used && (
                            <div>
                                <span className="font-medium text-gray-700">Clips Used:</span>
                                <p className="text-gray-900">{processingStatus.clips_used} clips</p>
                            </div>
                        )}
                        {processingStatus?.total_duration && (
                            <div>
                                <span className="font-medium text-gray-700">Duration:</span>
                                <p className="text-gray-900">{processingStatus.total_duration.toFixed(1)}s</p>
                            </div>
                        )}
                        {processingStatus?.platform && (
                            <div>
                                <span className="font-medium text-gray-700">Platform:</span>
                                <p className="text-gray-900">{processingStatus.platform}</p>
                            </div>
                        )}
                        {processingStatus?.aspect_ratio && (
                            <div>
                                <span className="font-medium text-gray-700">Aspect Ratio:</span>
                                <p className="text-gray-900">{processingStatus.aspect_ratio}</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Processed Video - Show when completed */}
                {processingStatus?.status === 'completed' && (
                    <div>
                        <h3 className="text-lg font-medium text-gray-900 mb-3 flex items-center">
                            <span className="mr-2">✨</span>
                            Final Compilation
                        </h3>
                        {processedVideoUrl ? (
                            <div className="space-y-3">
                                <div className="relative bg-black rounded-lg overflow-hidden">
                                    <video
                                        ref={processedVideoRef}
                                        src={processedVideoUrl}
                                        controls
                                        className="w-full h-auto max-h-96"
                                        preload="metadata"
                                    >
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                                <div className="flex justify-between items-center">
                                    <button
                                        onClick={handleDownload}
                                        disabled={isDownloading}
                                        className="bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center"
                                    >
                                        {isDownloading ? (
                                            <>
                                                <span className="animate-spin mr-2">⏳</span>
                                                Downloading...
                                            </>
                                        ) : (
                                            <>
                                                <span className="mr-2">⬇️</span>
                                                Download Compilation
                                            </>
                                        )}
                                    </button>
                                    {processingStatus.message && (
                                        <p className="text-sm text-gray-600 italic">
                                            {processingStatus.message}
                                        </p>
                                    )}
                                </div>
                            </div>
                        ) : (
                            <div className="bg-gray-50 rounded-lg p-4 text-center">
                                {isLoadingVideo ? (
                                    <div className="flex items-center justify-center space-x-2">
                                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                                        <p className="text-gray-600">Loading compilation video...</p>
                                    </div>
                                ) : (
                                    <>
                                        <p className="text-gray-600 mb-3">Video not loaded yet</p>
                                        <button
                                            onClick={loadProcessedVideo}
                                            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
                                        >
                                            🔄 Load Video
                                        </button>
                                    </>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* Processing Status */}
                {processingStatus && (
                    <div className="bg-gray-50 rounded-lg p-4">
                        <div className="flex items-center space-x-3 mb-3">
                            <span className="text-2xl">{getStatusIcon(processingStatus.status)}</span>
                            <div>
                                <h3 className="font-medium text-gray-900">Processing Status</h3>
                                <p className={`text-sm px-2 py-1 rounded-full inline-block ${getStatusColor(processingStatus.status)}`}>
                                    {processingStatus.status}
                                </p>
                            </div>
                        </div>

                        {processingStatus.progress !== undefined && (
                            <div className="space-y-2">
                                <div className="flex justify-between text-sm text-gray-600">
                                    <span>Progress</span>
                                    <span>{processingStatus.progress}%</span>
                                </div>
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div
                                        className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${processingStatus.progress}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {processingStatus.message && (
                            <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                                <p className="text-sm text-blue-800">
                                    {processingStatus.message}
                                </p>
                            </div>
                        )}

                        {processingStatus.error && (
                            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
                                <p className="text-sm text-red-800">
                                    <span className="font-medium">Error:</span> {processingStatus.error}
                                </p>
                            </div>
                        )}

                        {processingStatus.output_path && (
                            <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                                <p className="text-sm text-green-800">
                                    <span className="font-medium">Output ready:</span> {processingStatus.output_path}
                                </p>
                            </div>
                        )}
                    </div>
                )}

                {/* Processing Steps Info */}
                {processingStatus?.status === 'processing' && (
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h3 className="font-medium text-blue-900 mb-2">Processing Steps</h3>
                        <div className="space-y-2 text-sm text-blue-800">
                            <div className="flex items-center">
                                <span className="mr-2">1️⃣</span>
                                <span>AI analyzing videos for best moments</span>
                            </div>
                            <div className="flex items-center">
                                <span className="mr-2">2️⃣</span>
                                <span>Creating compilation with transitions</span>
                            </div>
                            <div className="flex items-center">
                                <span className="mr-2">3️⃣</span>
                                <span>Optimizing for selected platform</span>
                            </div>
                            {processingStatus.message && (
                                <div className="flex items-center">
                                    <span className="mr-2">4️⃣</span>
                                    <span>{processingStatus.message}</span>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}
