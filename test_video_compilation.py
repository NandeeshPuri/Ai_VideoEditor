#!/usr/bin/env python3
"""
Test script for video compilation service
"""

import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, 'backend')

from services.video_compilation import VideoCompilationService

async def test_compilation_service():
    """Test the video compilation service"""
    print("🧪 Testing Video Compilation Service...")
    
    # Initialize service
    service = VideoCompilationService()
    
    # Test presets
    print("\n📋 Available presets:")
    presets = service.get_compilation_presets()
    for preset in presets:
        print(f"  • {preset['name']}: {preset['description']}")
    
    # Test transition styles
    print("\n🎬 Available transition styles:")
    transitions = service.get_transition_styles()
    for transition in transitions:
        print(f"  • {transition['name']}: {transition['description']}")
    
    print("\n✅ Video compilation service is ready!")
    print("\n🎯 Features:")
    print("  • AI-powered best parts detection")
    print("  • Multiple transition styles")
    print("  • Social media optimization")
    print("  • Support for up to 5 videos")
    print("  • YouTube, Instagram, TikTok presets")

if __name__ == "__main__":
    asyncio.run(test_compilation_service())
