import cv2
import numpy as np
from pathlib import Path
import subprocess
import os
import tempfile
from PIL import Image, ImageDraw

class ObjectRemovalService:
    def __init__(self):
        pass
    
    async def process(self, upload_id: str, processing_status: dict, bounding_box: list = [100, 100, 200, 200]):
        """Process object removal from video"""
        try:
            # Update status
            processing_status[upload_id]["progress"] = 20
            processing_status[upload_id]["status"] = "processing"
            
            # Get file paths
            file_path = Path(processing_status[upload_id]["file_path"])
            output_dir = Path("temp/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            processing_status[upload_id]["progress"] = 30
            frames = self._extract_frames(str(file_path))
            
            # Process frames to remove objects
            processing_status[upload_id]["progress"] = 50
            processed_frames = self._remove_objects_from_frames(frames, bounding_box, processing_status, upload_id)
            
            # Recombine frames into video
            processing_status[upload_id]["progress"] = 80
            output_path = output_dir / f"{upload_id}_object_removed.mp4"
            self._create_video_from_frames(processed_frames, str(output_path), original_video_path=str(file_path))
            
            # Update status
            processing_status[upload_id]["progress"] = 100
            processing_status[upload_id]["status"] = "completed"
            processing_status[upload_id]["output_path"] = str(output_path)
            processing_status[upload_id]["removed_objects"] = [bounding_box]
            
        except Exception as e:
            processing_status[upload_id]["status"] = "error"
            processing_status[upload_id]["error"] = str(e)
            print(f"Object removal error: {e}")
    
    def _extract_frames(self, video_path: str, max_frames: int = 1000000):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)

            frame_count += 1
        
        cap.release()
        return frames
    
    def _remove_objects_from_frames(self, frames: list, bounding_box: list, processing_status: dict, upload_id: str):
        """Remove objects from frames using inpainting"""
        processed_frames = []
        
        for i, frame in enumerate(frames):
            # Create mask for the object to remove
            mask = self._create_removal_mask(frame, bounding_box)
            
            # Apply inpainting to remove the object
            processed_frame = self._apply_inpainting(frame, mask)
            
            processed_frames.append(processed_frame)
            
            # Update progress
            if i % 10 == 0:
                progress = 50 + (i / len(frames)) * 30
                processing_status[upload_id]["progress"] = int(progress)
        
        return processed_frames
    
    def _create_removal_mask(self, frame: np.ndarray, bounding_box: list) -> np.ndarray:
        """Create a mask for the object to remove"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bounding_box
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        
        # Create rectangular mask
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask
    
    def _apply_inpainting(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply inpainting to remove objects"""
        try:
            # Use OpenCV's inpainting algorithm
            result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
            
            # Alternative: use Navier-Stokes inpainting
            # result = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)
            
            return result
            
        except Exception as e:
            print(f"Inpainting failed: {e}")
            # Fallback: simple blur and blend
            return self._fallback_object_removal(frame, mask)
    
    def _fallback_object_removal(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback method for object removal using blur and blending"""
        # Create a blurred version of the frame
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)
        
        # Blend the blurred area with the original frame
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_normalized = np.expand_dims(mask_normalized, axis=2)
        
        result = frame * (1 - mask_normalized) + blurred * mask_normalized
        
        return result.astype(np.uint8)
    
    def _create_video_from_frames(self, frames: list, output_path: str, fps: int = None, original_video_path: str = None):
        """Create video from processed frames"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        if fps is None and original_video_path:
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()
        if fps is None or fps <= 0:
            fps = 25
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Convert to MP4 using FFmpeg for better compatibility
        temp_path = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_path)
        
        vf = "scale=-2:'if(gte(ih,720),ih,720)'"
        cmd = [
            'ffmpeg', '-i', temp_path,
            '-i', original_video_path if original_video_path else temp_path,
            '-vf', vf,
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            '-y', output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_path)
        except subprocess.CalledProcessError:
            # If FFmpeg fails, keep the original
            os.rename(temp_path, output_path)
    
    def detect_objects(self, frame: np.ndarray) -> list:
        """Detect potential objects in a frame for easier selection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area
            min_area = 1000  # Minimum area to consider as object
            objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    objects.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "confidence": min(area / 10000, 1.0)  # Simple confidence based on area
                    })
            
            # Sort by confidence
            objects.sort(key=lambda x: x["confidence"], reverse=True)
            
            return objects[:10]  # Return top 10 objects
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    def create_preview_mask(self, frame: np.ndarray, bounding_box: list) -> np.ndarray:
        """Create a preview showing what will be removed"""
        preview = frame.copy()
        
        # Draw bounding box
        x1, y1, x2, y2 = bounding_box
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add label
        cv2.putText(preview, "Object to Remove", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return preview
    
    def get_removal_techniques(self) -> list:
        """Get available object removal techniques"""
        return [
            {
                "id": "inpaint_telea",
                "name": "Telea Inpainting",
                "description": "Fast inpainting using Telea algorithm",
                "quality": "high",
                "speed": "fast"
            },
            {
                "id": "inpaint_ns",
                "name": "Navier-Stokes Inpainting",
                "description": "High-quality inpainting using Navier-Stokes",
                "quality": "very_high",
                "speed": "slow"
            },
            {
                "id": "blur_blend",
                "name": "Blur and Blend",
                "description": "Simple blur and blend technique",
                "quality": "medium",
                "speed": "very_fast"
            }
        ]
    
    def process_multiple_objects(self, frame: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """Remove multiple objects from a single frame"""
        result = frame.copy()
        
        for bbox in bounding_boxes:
            mask = self._create_removal_mask(result, bbox)
            result = self._apply_inpainting(result, mask)
        
        return result
    
    def estimate_processing_time(self, video_duration: float, frame_count: int) -> dict:
        """Estimate processing time for object removal"""
        # Rough estimates based on processing complexity
        frames_per_second = frame_count / video_duration
        
        # Processing time per frame (in seconds)
        time_per_frame = 0.1  # 100ms per frame
        
        total_time = frame_count * time_per_frame
        
        return {
            "estimated_time_seconds": total_time,
            "estimated_time_minutes": total_time / 60,
            "frames_per_second": frames_per_second,
            "total_frames": frame_count
        }
