'use client'

import StyleFilterSelector from './StyleFilterSelector'

interface FeatureSelectorProps {
  selectedFeatures: string[]
  onFeatureToggle: (feature: string) => void
  selectedStyle?: string
  onStyleChange?: (style: string) => void
  disabled?: boolean
}

const AI_FEATURES = [
  {
    id: 'auto-cut-silence',
    name: 'AI Auto-Cut & Transitions',
    description: 'Remove silence with AI scene analysis and smooth transitions',
    icon: '✂️',
    category: 'Editing'
  },
  {
    id: 'background-removal',
    name: 'AI Background Removal',
    description: 'Remove or replace video backgrounds',
    icon: '🎭',
    category: 'Visual'
  },
  {
    id: 'subtitles',
    name: 'AI Subtitle Generation',
    description: 'Generate and burn-in subtitles',
    icon: '💬',
    category: 'Audio'
  },
  {
    id: 'scene-detection',
    name: 'AI Scene Detection',
    description: 'Auto-split video into scenes',
    icon: '🎬',
    category: 'Analysis'
  },
  {
    id: 'voice-translate',
    name: 'Voice Translation & Dubbing',
    description: 'Translate speech and generate new audio',
    icon: '🌍',
    category: 'Audio'
  },
  {
    id: 'style',
    name: 'AI Style Filters',
    description: 'Apply artistic styles to video',
    icon: '🎨',
    category: 'Visual'
  },
  {
    id: 'object-remove',
    name: 'AI Object Removal',
    description: 'Remove unwanted objects from video',
    icon: '🧹',
    category: 'Visual'
  }
]

export default function FeatureSelector({ 
  selectedFeatures, 
  onFeatureToggle, 
  selectedStyle = 'cartoon',
  onStyleChange,
  disabled 
}: FeatureSelectorProps) {
  const groupedFeatures = AI_FEATURES.reduce((acc, feature) => {
    if (!acc[feature.category]) {
      acc[feature.category] = []
    }
    acc[feature.category].push(feature)
    return acc
  }, {} as Record<string, typeof AI_FEATURES>)

  return (
    <div className="bg-white rounded-lg p-6 shadow-lg">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">AI Features</h2>

      <div className="space-y-6">
        {Object.entries(groupedFeatures).map(([category, features]) => (
          <div key={category}>
            <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide mb-3">
              {category}
            </h3>

            <div className="space-y-3">
              {features.map((feature) => (
                <label
                  key={feature.id}
                  className={`flex items-start space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${selectedFeatures.includes(feature.id)
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                    } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedFeatures.includes(feature.id)}
                    onChange={() => onFeatureToggle(feature.id)}
                    disabled={disabled}
                    className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-xl">{feature.icon}</span>
                      <span className="text-sm font-medium text-gray-900">
                        {feature.name}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {feature.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {selectedFeatures.length > 0 && (
        <div className="mt-6 p-3 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <span className="font-medium">{selectedFeatures.length}</span> feature{selectedFeatures.length !== 1 ? 's' : ''} selected
          </p>
        </div>
      )}

      {/* Style Filter Selector */}
      <StyleFilterSelector
        isVisible={selectedFeatures.includes('style')}
        selectedStyle={selectedStyle}
        onStyleChange={onStyleChange || (() => {})}
        disabled={disabled}
      />
    </div>
  )
}
