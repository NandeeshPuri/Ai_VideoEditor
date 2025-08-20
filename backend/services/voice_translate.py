import whisper
import subprocess
import tempfile
import os
from pathlib import Path
import json
from googletrans import Translator
from TTS.api import TTS
from gtts import gTTS
import numpy as np
import wave
import struct

class VoiceTranslationService:
    def __init__(self):
        self.whisper_model = None  # lazy load
        self.translator = Translator()
        self.tts = None  # lazy load
    
    async def process(self, upload_id: str, processing_status: dict, target_language: str = "es", voice_gender: str = "female"):
        """Process voice translation and dubbing"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            processing_status[upload_id]["progress"] = 30
            audio_path = self._extract_audio(str(file_path))
            
            # Transcribe audio
            processing_status[upload_id]["progress"] = 40
            transcription = self._transcribe_audio(audio_path)
            
            # Translate text
            processing_status[upload_id]["progress"] = 50
            translated_text = self._translate_text(transcription, target_language)
            
            # Generate new audio
            processing_status[upload_id]["progress"] = 60
            new_audio_path = self._generate_speech(translated_text, target_language, voice_gender)
            
            # Replace audio in video
            processing_status[upload_id]["progress"] = 80
            output_path = output_dir / f"{upload_id}_dubbed_{target_language}.mp4"
            self._replace_audio(str(file_path), new_audio_path, str(output_path))
            
            # Clean up temporary files
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(new_audio_path):
                os.remove(new_audio_path)
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            processing_status[upload_id]["original_text"] = transcription
            processing_status[upload_id]["translated_text"] = translated_text
            processing_status[upload_id]["target_language"] = target_language
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Voice translation error: {e}")
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '22050', '-ac', '1',
            '-y', temp_audio.name
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_audio.name
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"].strip()
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text using Google Translate"""
        try:
            # Split text into sentences for better translation
            sentences = text.split('.')
            translated_sentences = []
            
            for sentence in sentences:
                if sentence.strip():
                    translated = self.translator.translate(
                        sentence.strip(), 
                        dest=target_language
                    )
                    translated_sentences.append(translated.text)
            
            return '. '.join(translated_sentences)
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def _generate_speech(self, text: str, language: str, voice_gender: str) -> str:
        """Generate speech from translated text using TTS"""
        try:
            if self.tts is None:
                # Try to init Coqui lazily; if it fails, fall back to gTTS
                try:
                    self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                except Exception:
                    self.tts = None
                    return self._generate_speech_gtts(text, language)
            
            # Create temporary file for output
            temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_output.close()
            
            # Generate speech
            self.tts.tts_to_file(text=text, file_path=temp_output.name)
            
            return temp_output.name
            
        except Exception as e:
            print(f"TTS error, falling back to gTTS: {e}")
            return self._generate_speech_gtts(text, language)

    def _generate_speech_gtts(self, text: str, language: str) -> str:
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output.close()
        # gTTS outputs mp3; create mp3 then convert to wav
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_mp3.close()
        try:
            gTTS(text=text or " ", lang=(language if len(language) == 2 else "en")).save(tmp_mp3.name)
        except Exception:
            # fallback to english
            gTTS(text=text or " ", lang="en").save(tmp_mp3.name)

        cmd = [
            'ffmpeg', '-i', tmp_mp3.name,
            '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
            '-y', temp_output.name
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        try:
            os.remove(tmp_mp3.name)
        except Exception:
            pass
        return temp_output.name
    
    def _generate_speech_espeak(self, text: str, language: str) -> str:
        """Generate speech using espeak as fallback"""
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output.close()
        
        # Map language codes to espeak voices
        voice_map = {
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "it": "italian",
            "pt": "portuguese",
            "ru": "russian",
            "ja": "japanese",
            "ko": "korean",
            "zh": "mandarin"
        }
        
        voice = voice_map.get(language, "english")
        
        cmd = [
            'espeak', '-w', temp_output.name,
            '-v', voice,
            text
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # If espeak fails, create a silent audio file
            self._create_silent_audio(temp_output.name, 3.0)  # 3 seconds of silence
        
        return temp_output.name
    
    def _create_silent_audio(self, output_path: str, duration: float):
        """Create a silent audio file as fallback"""
        sample_rate = 22050
        num_samples = int(duration * sample_rate)
        
        # Create silent audio data
        silent_data = [0] * num_samples
        
        # Write WAV file
        with wave.open(output_path, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert to bytes
            audio_bytes = struct.pack(f'<{num_samples}h', *silent_data)
            wav_file.writeframes(audio_bytes)
    
    def _replace_audio(self, video_path: str, audio_path: str, output_path: str):
        """Replace audio in video using FFmpeg and ensure at least 720p height."""
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-vf', "scale=-2:'if(gte(ih,720),ih,720)'",
            '-map', '0:v:0', '-map', '1:a:0',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac',
            '-shortest',
            '-movflags', '+faststart',
            '-y', output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages for translation"""
        return [
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese (Mandarin)"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "nl", "name": "Dutch"},
            {"code": "pl", "name": "Polish"},
            {"code": "tr", "name": "Turkish"}
        ]
    
    def get_voice_options(self, language: str) -> list:
        """Get available voice options for a language"""
        # This would typically return available TTS voices
        # For now, return basic options
        return [
            {"id": "female", "name": "Female Voice"},
            {"id": "male", "name": "Male Voice"}
        ]
    
    def preview_translation(self, text: str, target_language: str) -> str:
        """Preview translation without processing full video"""
        try:
            return self._translate_text(text, target_language)
        except Exception as e:
            return f"Translation error: {str(e)}"
