'use client'

import { useState, useRef } from 'react'

interface VideoUploaderProps {
    onUploadSuccess: (uploadId: string, file: File) => void
    disabled?: boolean
}

export default function VideoUploader({ onUploadSuccess, disabled }: VideoUploaderProps) {
    const [isDragging, setIsDragging] = useState(false)
    const [isUploading, setIsUploading] = useState(false)
    const [uploadProgress, setUploadProgress] = useState(0)
    const fileInputRef = useRef<HTMLInputElement>(null)

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
        if (!disabled) setIsDragging(true)
    }

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        setIsDragging(false)

        if (disabled) return

        const files = Array.from(e.dataTransfer.files)
        const videoFile = files.find(file => file.type.startsWith('video/'))

        if (videoFile) {
            handleFileUpload(videoFile)
        }
    }

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file && file.type.startsWith('video/')) {
            handleFileUpload(file)
        }
    }

    const handleFileUpload = async (file: File) => {
        if (disabled) return

        setIsUploading(true)
        setUploadProgress(0)

        try {
            const formData = new FormData()
            formData.append('file', file)

            const response = await fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                throw new Error('Upload failed')
            }

            const result = await response.json()
            onUploadSuccess(result.upload_id, file)

        } catch (error) {
            console.error('Upload error:', error)
            alert('Upload failed. Please try again.')
        } finally {
            setIsUploading(false)
            setUploadProgress(0)
        }
    }

    const openFileDialog = () => {
        fileInputRef.current?.click()
    }

    return (
        <div className="bg-white rounded-lg p-6 shadow-lg">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Video</h2>

            <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${isDragging
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-300 hover:border-gray-400'
                    } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={!disabled ? openFileDialog : undefined}
            >
                <div className="space-y-4">
                    <div className="text-6xl text-gray-400">📹</div>

                    <div>
                        <p className="text-lg font-medium text-gray-900">
                            {isUploading ? 'Uploading...' : 'Drop your video here'}
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                            or click to browse files
                        </p>
                    </div>

                    {isUploading && (
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${uploadProgress}%` }}
                            />
                        </div>
                    )}

                    <p className="text-xs text-gray-400">
                        Supports MP4, WebM, AVI, MOV and other video formats
                    </p>
                </div>
            </div>

            <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
                disabled={disabled}
            />
        </div>
    )
}
