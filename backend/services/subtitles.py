import whisper
import subprocess
import tempfile
import os
from pathlib import Path
import json
from googletrans import Translator
import shutil

class SubtitleService:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.translator = Translator()
    
    async def process(self, upload_id: str, processing_status: dict, burn_in: bool = True, language: str = "en"):
        """Process subtitle generation and optional burn-in"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path("../" + processing_status[upload_id]["file_path"])
            output_dir = Path("../temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract audio
            processing_status[upload_id]["progress"] = 30
            audio_path = self._extract_audio(str(file_path))
            
            # Transcribe audio
            processing_status[upload_id]["progress"] = 50
            transcription = self._transcribe_audio(audio_path)
            
            # Generate SRT file
            processing_status[upload_id]["progress"] = 70
            srt_path = output_dir / f"{upload_id}_subtitles.srt"
            self._generate_srt(transcription, str(srt_path))
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            if burn_in:
                # Burn subtitles into video
                processing_status[upload_id]["progress"] = 85
                output_path = output_dir / f"{upload_id}_with_subtitles.mp4"
                self._burn_subtitles(str(file_path), str(srt_path), str(output_path))
                
                # Update status with video output
                processing_status[upload_id]["progress"] = 100
                processing_status[upload_id]["status"] = "completed"
                processing_status[upload_id]["output_path"] = str(output_path).replace("../", "")
                processing_status[upload_id]["srt_path"] = str(srt_path).replace("../", "")
            else:
                # Just return SRT file
                processing_status[upload_id]["progress"] = 100
                processing_status[upload_id]["status"] = "completed"
                processing_status[upload_id]["srt_path"] = str(srt_path).replace("../", "")
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Subtitle generation error: {e}")
    
    def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video using FFmpeg"""
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()
        
        cmd = [
            (shutil.which('ffmpeg') or 'ffmpeg'), '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', temp_audio.name
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return temp_audio.name
    
    def _transcribe_audio(self, audio_path: str) -> list:
        """Transcribe audio using Whisper"""
        result = self.model.transcribe(audio_path)
        
        # Format transcription for SRT
        segments = []
        for i, segment in enumerate(result["segments"]):
            start_time = self._format_timestamp(segment["start"])
            end_time = self._format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            segments.append({
                "index": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })
        
        return segments
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _generate_srt(self, segments: list, output_path: str):
        """Generate SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['index']}\n")
                f.write(f"{segment['start_time']} --> {segment['end_time']}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _burn_subtitles(self, video_path: str, srt_path: str, output_path: str):
        """Burn subtitles into video using FFmpeg"""
        srt_escaped = str(Path(srt_path)).replace('\\', '/')
        cmd = [
            (shutil.which('ffmpeg') or 'ffmpeg'), '-i', video_path,
            '-vf', f"subtitles='{srt_escaped}'",
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]

        subprocess.run(cmd, check=True, capture_output=True)
    
    def translate_subtitles(self, srt_path: str, target_language: str) -> str:
        """Translate subtitles to target language"""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT content
            segments = self._parse_srt(content)
            
            # Translate each text segment
            translated_segments = []
            for segment in segments:
                translated_text = self.translator.translate(
                    segment['text'], 
                    dest=target_language
                ).text
                
                segment['text'] = translated_text
                translated_segments.append(segment)
            
            # Generate new SRT file
            output_path = srt_path.replace('.srt', f'_{target_language}.srt')
            self._generate_srt(translated_segments, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Translation error: {e}")
            return srt_path
    
    def _parse_srt(self, srt_content: str) -> list:
        """Parse SRT content into segments"""
        segments = []
        lines = srt_content.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.isdigit():  # Segment index
                # Find timestamp line
                timestamp_line = lines[i + 1].strip()
                start_time, end_time = timestamp_line.split(' --> ')
                
                # Find text lines
                text_lines = []
                j = i + 2
                while j < len(lines) and lines[j].strip():
                    text_lines.append(lines[j].strip())
                    j += 1
                
                text = ' '.join(text_lines)
                
                segments.append({
                    'index': int(line),
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
                
                i = j + 1
            else:
                i += 1
        
        return segments
