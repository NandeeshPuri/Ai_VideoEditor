'use client'

import { useRef, useEffect, useState } from 'react'

interface VideoPreviewProps {
    file: File
    uploadId: string | null
    processingStatus: any
}

export default function VideoPreview({ file, uploadId, processingStatus }: VideoPreviewProps) {
    const videoRef = useRef<HTMLVideoElement>(null)
    const [videoUrl, setVideoUrl] = useState<string>('')

    useEffect(() => {
        if (file) {
            const url = URL.createObjectURL(file)
            setVideoUrl(url)
            return () => URL.revokeObjectURL(url)
        }
    }, [file])

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
                return '📹'
        }
    }

    return (
        <div className="bg-white rounded-lg p-6 shadow-lg">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Video Preview</h2>

            <div className="space-y-4">
                {/* Video Player */}
                <div className="relative bg-black rounded-lg overflow-hidden">
                    <video
                        ref={videoRef}
                        src={videoUrl}
                        controls
                        className="w-full h-auto max-h-96"
                        preload="metadata"
                    >
                        Your browser does not support the video tag.
                    </video>
                </div>

                {/* File Info */}
                <div className="bg-gray-50 rounded-lg p-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                            <span className="font-medium text-gray-700">Filename:</span>
                            <p className="text-gray-900 truncate">{file.name}</p>
                        </div>
                        <div>
                            <span className="font-medium text-gray-700">Size:</span>
                            <p className="text-gray-900">
                                {(file.size / (1024 * 1024)).toFixed(2)} MB
                            </p>
                        </div>
                        <div>
                            <span className="font-medium text-gray-700">Type:</span>
                            <p className="text-gray-900">{file.type}</p>
                        </div>
                        <div>
                            <span className="font-medium text-gray-700">Upload ID:</span>
                            <p className="text-gray-900 font-mono text-xs">{uploadId}</p>
                        </div>
                    </div>
                </div>

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
                                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                        style={{ width: `${processingStatus.progress}%` }}
                                    />
                                </div>
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
            </div>
        </div>
    )
}
