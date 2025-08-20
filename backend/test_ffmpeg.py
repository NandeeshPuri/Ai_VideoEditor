import subprocess
import tempfile
import os
from pathlib import Path

def test_ffmpeg_concat():
    """Test FFmpeg concat functionality"""
    print("Testing FFmpeg concat functionality...")
    
    # Create a simple test concat file
    concat_list = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        # Test with a simple file reference
        test_path = "C:/test/video.mp4"
        concat_list.write(f"file '{test_path}'\n")
        concat_list.close()
        
        print(f"Created concat file: {concat_list.name}")
        print(f"Content: file '{test_path}'")
        
        # Test FFmpeg concat command (this will fail but show the error)
        cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list.name,
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-c:a", "aac",
            "-y", "test_output.mp4"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("FFmpeg concat successful!")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg concat failed with exit code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            
    finally:
        try:
            os.remove(concat_list.name)
        except Exception:
            pass

if __name__ == "__main__":
    test_ffmpeg_concat()
