'use client'

import { useState, useEffect } from 'react'

interface VideoCompilationSelectorProps {
    onUploadIdsChange: (uploadIds: string[]) => void
    onMaxDurationChange: (duration: number) => void
    onTransitionStyleChange: (style: string) => void
    onPresetChange: (preset: string) => void
    onApplyEffectsChange: (apply: boolean) => void
    onEffectTypeChange: (effectType: string) => void
    disabled?: boolean
    availableUploadIds?: string[]
}

interface CompilationPreset {
    id: string
    name: string
    max_duration: number
    description: string
    aspect_ratio: string
    features: string[]
    platform: string
    icon: string
}

interface TransitionStyle {
    id: string
    name: string
    description: string
    icon: string
}

interface PostCompilationEffect {
    id: string
    name: string
    description: string
    icon: string
    category: string
}

const COMPILATION_PRESETS: CompilationPreset[] = [
    {
        id: "youtube_shorts",
        name: "YouTube Shorts",
        max_duration: 60,
        description: "Vertical 9:16 format, perfect for mobile viewing",
        aspect_ratio: "9:16",
        features: ["Vertical format", "Mobile-optimized", "Trending content"],
        platform: "YouTube",
        icon: "📱"
    },
    {
        id: "youtube_standard",
        name: "YouTube Standard",
        max_duration: 300,
        description: "Traditional 16:9 format for desktop viewing",
        aspect_ratio: "16:9",
        features: ["Desktop optimized", "Longer content", "Detailed editing"],
        platform: "YouTube",
        icon: "📺"
    },
    {
        id: "instagram_reels",
        name: "Instagram Reels",
        max_duration: 90,
        description: "Vertical format with music and effects support",
        aspect_ratio: "9:16",
        features: ["Music integration", "Effects ready", "Story-friendly"],
        platform: "Instagram",
        icon: "📸"
    },
    {
        id: "instagram_stories",
        name: "Instagram Stories",
        max_duration: 15,
        description: "Quick 15-second stories for daily content",
        aspect_ratio: "9:16",
        features: ["Quick content", "Daily updates", "Engagement focused"],
        platform: "Instagram",
        icon: "📱"
    },
    {
        id: "tiktok",
        name: "TikTok",
        max_duration: 60,
        description: "Trending vertical format with viral potential",
        aspect_ratio: "9:16",
        features: ["Viral potential", "Trending format", "Music sync"],
        platform: "TikTok",
        icon: "🎵"
    },
    {
        id: "facebook_reels",
        name: "Facebook Reels",
        max_duration: 60,
        description: "Facebook's short-form video format",
        aspect_ratio: "9:16",
        features: ["Facebook optimized", "Community focused", "Shareable"],
        platform: "Facebook",
        icon: "👥"
    },
    {
        id: "twitter_video",
        name: "Twitter Video",
        max_duration: 140,
        description: "Twitter's video format with character limit awareness",
        aspect_ratio: "16:9",
        features: ["Twitter optimized", "Quick consumption", "Thread-friendly"],
        platform: "Twitter",
        icon: "🐦"
    },
    {
        id: "linkedin_video",
        name: "LinkedIn Video",
        max_duration: 600,
        description: "Professional format for business content",
        aspect_ratio: "16:9",
        features: ["Professional", "Business focused", "Educational"],
        platform: "LinkedIn",
        icon: "💼"
    },
    {
        id: "snapchat_spotlight",
        name: "Snapchat Spotlight",
        max_duration: 60,
        description: "Snapchat's vertical video format",
        aspect_ratio: "9:16",
        features: ["Snapchat native", "Youth audience", "Creative effects"],
        platform: "Snapchat",
        icon: "👻"
    },
    {
        id: "pinterest_video",
        name: "Pinterest Video",
        max_duration: 60,
        description: "Pinterest's visual discovery format",
        aspect_ratio: "2:3",
        features: ["Visual discovery", "Inspiration focused", "Pin-optimized"],
        platform: "Pinterest",
        icon: "📌"
    },
    {
        id: "custom",
        name: "Custom Format",
        max_duration: 300,
        description: "Customize your own format and settings",
        aspect_ratio: "16:9",
        features: ["Fully customizable", "Flexible duration", "Any platform"],
        platform: "Custom",
        icon: "⚙️"
    }
]

const TRANSITION_STYLES: TransitionStyle[] = [
    {
        id: "fade",
        name: "Fade In/Out",
        description: "Smooth fade transitions between clips",
        icon: "🌅"
    },
    {
        id: "crossfade",
        name: "Crossfade",
        description: "Overlapping fade transitions for seamless flow",
        icon: "🔄"
    },
    {
        id: "slide",
        name: "Slide",
        description: "Slide transitions between clips",
        icon: "➡️"
    },
    {
        id: "zoom",
        name: "Zoom",
        description: "Zoom in/out transitions for dynamic effect",
        icon: "🔍"
    },
    {
        id: "wipe",
        name: "Wipe",
        description: "Wipe transitions for modern look",
        icon: "🧹"
    },
    {
        id: "dissolve",
        name: "Dissolve",
        description: "Dissolve transitions for artistic effect",
        icon: "✨"
    }
]

const POST_COMPILATION_EFFECTS: PostCompilationEffect[] = [
    {
        id: "none",
        name: "No Effect",
        description: "Keep original video without any effects",
        icon: "🎬",
        category: "none"
    },
    {
        id: "vintage",
        name: "Vintage",
        description: "Classic film look with warm tones and grain",
        icon: "📷",
        category: "retro"
    },
    {
        id: "cinematic",
        name: "Cinematic",
        description: "Movie-like appearance with enhanced contrast",
        icon: "🎭",
        category: "professional"
    },
    {
        id: "warm",
        name: "Warm",
        description: "Cozy, golden-hour lighting effect",
        icon: "🌅",
        category: "color"
    },
    {
        id: "cool",
        name: "Cool",
        description: "Blue-tinted, modern aesthetic",
        icon: "❄️",
        category: "color"
    },
    {
        id: "dramatic",
        name: "Dramatic",
        description: "High contrast, bold colors for impact",
        icon: "⚡",
        category: "professional"
    },
    {
        id: "bright",
        name: "Bright",
        description: "Enhanced brightness and vibrant colors",
        icon: "☀️",
        category: "color"
    },
    {
        id: "moody",
        name: "Moody",
        description: "Dark, atmospheric mood with reduced brightness",
        icon: "🌙",
        category: "atmospheric"
    },
    {
        id: "vibrant",
        name: "Vibrant",
        description: "Saturated, eye-catching colors",
        icon: "🌈",
        category: "color"
    },
    {
        id: "monochrome",
        name: "Monochrome",
        description: "Classic black and white effect",
        icon: "⚫",
        category: "retro"
    },
    {
        id: "sepia",
        name: "Sepia",
        description: "Antique brown-tinted effect",
        icon: "📜",
        category: "retro"
    }
]

export default function VideoCompilationSelector({
    onUploadIdsChange,
    onMaxDurationChange,
    onTransitionStyleChange,
    onPresetChange,
    onApplyEffectsChange,
    onEffectTypeChange,
    disabled,
    availableUploadIds = []
}: VideoCompilationSelectorProps) {
    const [selectedUploadIds, setSelectedUploadIds] = useState<string[]>([])
    const [selectedPreset, setSelectedPreset] = useState<string>("youtube_shorts")
    const [selectedTransitionStyle, setSelectedTransitionStyle] = useState<string>("fade")
    const [customDuration, setCustomDuration] = useState<number>(60)
    const [applyEffects, setApplyEffects] = useState<boolean>(false)
    const [selectedEffect, setSelectedEffect] = useState<string>("none")

    useEffect(() => {
        onUploadIdsChange(selectedUploadIds)
    }, [selectedUploadIds, onUploadIdsChange])

    useEffect(() => {
        const preset = COMPILATION_PRESETS.find(p => p.id === selectedPreset)
        if (preset) {
            setCustomDuration(preset.max_duration)
            onMaxDurationChange(preset.max_duration)
            onPresetChange(selectedPreset)
        }
    }, [selectedPreset, onMaxDurationChange, onPresetChange])

    useEffect(() => {
        onTransitionStyleChange(selectedTransitionStyle)
    }, [selectedTransitionStyle, onTransitionStyleChange])

    useEffect(() => {
        onApplyEffectsChange(applyEffects)
    }, [applyEffects, onApplyEffectsChange])

    useEffect(() => {
        onEffectTypeChange(selectedEffect)
    }, [selectedEffect, onEffectTypeChange])

    const handleUploadIdToggle = (uploadId: string) => {
        if (selectedUploadIds.includes(uploadId)) {
            setSelectedUploadIds(selectedUploadIds.filter(id => id !== uploadId))
        } else {
            if (selectedUploadIds.length < 5) {
                setSelectedUploadIds([...selectedUploadIds, uploadId])
            }
        }
    }

    const handleCustomDurationChange = (duration: number) => {
        setCustomDuration(duration)
        onMaxDurationChange(duration)
    }

    const selectedPresetData = COMPILATION_PRESETS.find(p => p.id === selectedPreset)

    return (
        <div className="bg-white rounded-lg p-4 shadow-md border border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <span className="mr-2">🎬</span>
                Video Compilation Settings
            </h3>

            <div className="space-y-4">
                {/* Video Selection */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Select Videos (Max 5)
                    </label>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                        {availableUploadIds.length === 0 ? (
                            <p className="text-sm text-gray-500 italic">
                                Upload videos first to create a compilation
                            </p>
                        ) : (
                            availableUploadIds.map((uploadId) => (
                                <label key={uploadId} className="flex items-center">
                                    <input
                                        type="checkbox"
                                        checked={selectedUploadIds.includes(uploadId)}
                                        onChange={() => handleUploadIdToggle(uploadId)}
                                        disabled={disabled}
                                        className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 disabled:opacity-50"
                                    />
                                    <span className="ml-2 text-sm text-gray-700">
                                        Video {uploadId.slice(0, 8)}...
                                    </span>
                                </label>
                            ))
                        )}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                        Selected: {selectedUploadIds.length}/5 videos
                    </p>
                </div>

                {/* Social Media Platform Preset */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Choose Social Media Platform
                    </label>
                    <div className="grid grid-cols-1 gap-2 max-h-60 overflow-y-auto">
                        {COMPILATION_PRESETS.map((preset) => (
                            <label
                                key={preset.id}
                                className={`flex items-start p-3 border-2 rounded-lg cursor-pointer transition-all duration-200 hover:bg-gray-50 ${selectedPreset === preset.id
                                    ? 'border-purple-500 bg-purple-50'
                                    : 'border-gray-200 bg-white'
                                    }`}
                            >
                                <input
                                    type="radio"
                                    name="preset"
                                    value={preset.id}
                                    checked={selectedPreset === preset.id}
                                    onChange={(e) => setSelectedPreset(e.target.value)}
                                    disabled={disabled}
                                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 disabled:opacity-50 mt-0.5"
                                />
                                <div className="ml-3 flex-1">
                                    <div className="flex items-center justify-between mb-1">
                                        <div className="flex items-center">
                                            <span className="text-lg mr-2">{preset.icon}</span>
                                            <h4 className="font-semibold text-gray-900 text-sm">{preset.name}</h4>
                                        </div>
                                        <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                                            {preset.platform}
                                        </span>
                                    </div>
                                    <p className="text-xs text-gray-600 mb-2">{preset.description}</p>
                                    <div className="flex items-center space-x-3 text-xs text-gray-500">
                                        <span>📐 {preset.aspect_ratio}</span>
                                        <span>⏱️ {preset.max_duration}s</span>
                                    </div>
                                </div>
                            </label>
                        ))}
                    </div>
                </div>



                {/* Custom Duration */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Max Duration (seconds)
                    </label>
                    <input
                        type="number"
                        min="15"
                        max="600"
                        value={customDuration}
                        onChange={(e) => handleCustomDurationChange(parseInt(e.target.value) || 60)}
                        disabled={disabled}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                    />
                </div>

                {/* Transition Style */}
                <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                        Transition Style
                    </label>
                    <div className="grid grid-cols-2 gap-2">
                        {TRANSITION_STYLES.map((style) => (
                            <label key={style.id} className="flex items-center p-2 border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer">
                                <input
                                    type="radio"
                                    name="transitionStyle"
                                    value={style.id}
                                    checked={selectedTransitionStyle === style.id}
                                    onChange={(e) => setSelectedTransitionStyle(e.target.value)}
                                    disabled={disabled}
                                    className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 disabled:opacity-50"
                                />
                                <span className="ml-2 text-sm text-gray-700 flex items-center">
                                    <span className="mr-1">{style.icon}</span>
                                    {style.name}
                                </span>
                            </label>
                        ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                        {TRANSITION_STYLES.find(s => s.id === selectedTransitionStyle)?.description}
                    </p>
                </div>

                {/* Post-Compilation Effects */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <label className="block text-sm font-medium text-gray-700">
                            Apply Effects to Final Video
                        </label>
                        <label className="flex items-center">
                            <input
                                type="checkbox"
                                checked={applyEffects}
                                onChange={(e) => setApplyEffects(e.target.checked)}
                                disabled={disabled}
                                className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded disabled:opacity-50"
                            />
                            <span className="ml-2 text-sm text-gray-700">Enable Effects</span>
                        </label>
                    </div>

                    {applyEffects && (
                        <div className="space-y-3">
                            <label className="block text-sm font-medium text-gray-700">
                                Select Effect
                            </label>
                            <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                                {POST_COMPILATION_EFFECTS.map((effect) => (
                                    <label key={effect.id} className="flex items-center p-2 border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer">
                                        <input
                                            type="radio"
                                            name="effectType"
                                            value={effect.id}
                                            checked={selectedEffect === effect.id}
                                            onChange={(e) => setSelectedEffect(e.target.value)}
                                            disabled={disabled}
                                            className="h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 disabled:opacity-50"
                                        />
                                        <span className="ml-2 text-sm text-gray-700 flex items-center">
                                            <span className="mr-1">{effect.icon}</span>
                                            {effect.name}
                                        </span>
                                    </label>
                                ))}
                            </div>
                            <p className="text-xs text-gray-500">
                                {POST_COMPILATION_EFFECTS.find(e => e.id === selectedEffect)?.description}
                            </p>
                        </div>
                    )}
                </div>

                {/* Preview Info */}
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-md p-3">
                    <p className="text-sm text-blue-800">
                        <strong>🎯 AI Compilation Preview:</strong> Your {selectedUploadIds.length} videos will be analyzed for the best moments and compiled into a {customDuration}-second video optimized for{' '}
                        <span className="font-semibold text-purple-800">
                            {selectedPresetData?.name}
                        </span>
                        .
                    </p>
                    <p className="text-sm text-blue-700 mt-2">
                        <strong>✨ Features:</strong> AI best parts detection, {TRANSITION_STYLES.find(s => s.id === selectedTransitionStyle)?.name.toLowerCase()} transitions, and {selectedPresetData?.aspect_ratio} aspect ratio for {selectedPresetData?.platform.toLowerCase()} optimization.
                    </p>
                </div>
            </div>
        </div>
    )
}
