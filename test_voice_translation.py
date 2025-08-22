#!/usr/bin/env python3
"""
Test script for voice translation functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_voice_translation_service():
    """Test the voice translation service"""
    try:
        from services.voice_translate import VoiceTranslationService
        
        print("✅ Voice translation service imported successfully")
        
        # Test service initialization
        service = VoiceTranslationService()
        print("✅ Service initialized successfully")
        
        # Test supported languages
        languages = service.get_supported_languages()
        print(f"✅ Found {len(languages)} supported languages")
        
        # Test voice options
        voice_options = service.get_voice_options("es")
        print(f"✅ Found {len(voice_options)} voice options")
        
        # Test translation preview
        test_text = "Hello, how are you?"
        translated = service.preview_translation(test_text, "es")
        print(f"✅ Translation preview: '{test_text}' -> '{translated}'")
        
        print("\n🎉 All voice translation tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Voice Translation Service...")
    success = test_voice_translation_service()
    sys.exit(0 if success else 1)
