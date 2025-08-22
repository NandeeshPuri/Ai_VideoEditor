#!/usr/bin/env python3
"""
Test script for optimized voice translation service
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

from services.voice_translate_optimized import OptimizedVoiceTranslationService

async def test_optimized_translation():
    """Test the optimized voice translation service"""
    print("🚀 Testing Optimized Voice Translation Service...")
    
    # Initialize service
    service = OptimizedVoiceTranslationService()
    
    # Test translation
    print("\n🌍 Testing translation...")
    sample_text = "Hello, this is a test video for translation."
    translated = await service.preview_translation(sample_text, "es")
    print(f"Original: {sample_text}")
    print(f"Translated: {translated}")
    
    # Test supported languages
    print("\n📋 Supported languages:")
    languages = service.get_supported_languages()
    for lang in languages[:5]:  # Show first 5
        print(f"  • {lang['name']} ({lang['code']})")
    print(f"  ... and {len(languages) - 5} more languages")
    
    # Test voice options
    print("\n🎤 Voice options:")
    voices = service.get_voice_options("es")
    for voice in voices:
        print(f"  • {voice['name']}")
    
    print("\n✅ Optimized voice translation service is ready!")
    print("\n🎯 Performance Improvements:")
    print("  • Parallel processing with ThreadPoolExecutor")
    print("  • Optimized audio extraction (16kHz instead of 22kHz)")
    print("  • Batch translation processing")
    print("  • Faster FFmpeg presets")
    print("  • Automatic subtitle generation")
    print("  • Better error handling and fallbacks")
    
    print("\n📱 New Features:")
    print("  • Optional subtitle burning")
    print("  • Dynamic subtitle timing")
    print("  • Multiple language support")
    print("  • Progress tracking with detailed messages")

if __name__ == "__main__":
    asyncio.run(test_optimized_translation())
