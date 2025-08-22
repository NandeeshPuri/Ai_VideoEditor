'use client'

import { useState, useEffect } from 'react'

interface VoiceTranslationSelectorProps {
  onLanguageChange: (language: string) => void
  onVoiceTypeChange: (voiceType: string) => void
  onSubtitleChange: (addSubtitles: boolean) => void
  disabled?: boolean
}

interface Language {
  code: string
  name: string
}

const SUPPORTED_LANGUAGES: Language[] = [
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "it", name: "Italian" },
  { code: "pt", name: "Portuguese" },
  { code: "ru", name: "Russian" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "zh", name: "Chinese (Mandarin)" },
  { code: "ar", name: "Arabic" },
  { code: "hi", name: "Hindi" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "tr", name: "Turkish" }
]

const VOICE_TYPES = [
  { id: "female", name: "Female Voice" },
  { id: "male", name: "Male Voice" }
]

export default function VoiceTranslationSelector({
  onLanguageChange,
  onVoiceTypeChange,
  onSubtitleChange,
  disabled
}: VoiceTranslationSelectorProps) {
  const [selectedLanguage, setSelectedLanguage] = useState<string>("es")
  const [selectedVoiceType, setSelectedVoiceType] = useState<string>("female")
  const [addSubtitles, setAddSubtitles] = useState<boolean>(true)

  useEffect(() => {
    onLanguageChange(selectedLanguage)
  }, [selectedLanguage, onLanguageChange])

  useEffect(() => {
    onVoiceTypeChange(selectedVoiceType)
  }, [selectedVoiceType, onVoiceTypeChange])

  useEffect(() => {
    onSubtitleChange(addSubtitles)
  }, [addSubtitles, onSubtitleChange])

  return (
    <div className="bg-white rounded-lg p-4 shadow-md border border-gray-200">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Voice Translation Settings
      </h3>

      <div className="space-y-4">
        {/* Language Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Target Language
          </label>
          <select
            value={selectedLanguage}
            onChange={(e) => setSelectedLanguage(e.target.value)}
            disabled={disabled}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          >
            {SUPPORTED_LANGUAGES.map((language) => (
              <option key={language.code} value={language.code}>
                {language.name}
              </option>
            ))}
          </select>
        </div>

        {/* Voice Type Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Voice Type
          </label>
          <div className="space-y-2">
            {VOICE_TYPES.map((voiceType) => (
              <label key={voiceType.id} className="flex items-center">
                <input
                  type="radio"
                  name="voiceType"
                  value={voiceType.id}
                  checked={selectedVoiceType === voiceType.id}
                  onChange={(e) => setSelectedVoiceType(e.target.value)}
                  disabled={disabled}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 disabled:opacity-50"
                />
                <span className="ml-2 text-sm text-gray-700">
                  {voiceType.name}
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Subtitle Options */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Subtitle Options
          </label>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={addSubtitles}
                onChange={(e) => setAddSubtitles(e.target.checked)}
                disabled={disabled}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 disabled:opacity-50"
              />
              <span className="ml-2 text-sm text-gray-700">
                Add translated subtitles to video
              </span>
            </label>
          </div>
        </div>

        {/* Preview Info */}
        <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
          <p className="text-sm text-blue-800">
            <strong>Preview:</strong> Your video will be translated to{' '}
            <span className="font-semibold">
              {SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.name}
            </span>{' '}
            with a{' '}
            <span className="font-semibold">
              {VOICE_TYPES.find(v => v.id === selectedVoiceType)?.name.toLowerCase()}
            </span>
            .
          </p>
          <p className="text-sm text-blue-700 mt-2">
            <strong>Features:</strong> Optimized speech translation, AI voice generation, and {addSubtitles ? 'burned-in' : 'optional'} subtitle translation.
          </p>
        </div>
      </div>
    </div>
  )
}
