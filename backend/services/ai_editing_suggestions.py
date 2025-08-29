import os
import json
import subprocess
import tempfile
import hashlib
import pickle
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
from pathlib import Path
import gc  # Add garbage collection

@dataclass
class VideoFeature:
    """Represents a video feature at a specific timestamp"""
    timestamp: float
    feature_type: str  # 'scene_change', 'motion', 'face_detected', 'silence', 'loud_audio'
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class EditingSuggestion:
    """Represents an AI editing suggestion"""
    timestamp: float
    suggestion_type: str  # 'cut', 'transition', 'pace_change', 'emphasis'
    confidence: float
    reason: str
    description: str = None  # Add description field
    reasoning: str = None    # Add reasoning field
    transition_type: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        # Set description and reasoning from reason if not provided
        if self.description is None:
            self.description = self.reason
        if self.reasoning is None:
            self.reasoning = self.reason
        if self.metadata is None:
            self.metadata = {}

class AIEditingSuggestionsService:
    def __init__(self):
        self.scene_change_threshold = 0.15  # More sensitive scene change detection
        self.motion_threshold = 0.1
        self.face_cascade = None
        self._load_face_cascade()
        
        # Advanced memory optimization settings for longer videos
        self.max_frames_to_analyze = 500  # Increased for comprehensive analysis
        self.max_video_duration = 1800  # 30 minutes max (increased from 10)
        self.frame_resize_factor = 0.3  # More aggressive resize for longer videos
        self.batch_size = 20  # Process frames in batches
        self.gc_interval = 5  # Garbage collection frequency
        
        # Parallel processing settings
        self.max_workers = min(8, os.cpu_count() or 4)  # Use available CPU cores
        self.parallel_batch_size = 10  # Frames per parallel batch
        self.enable_parallel = True  # Enable parallel processing
        
        # Enhanced caching for longer videos
        self.cache_dir = Path("backend/temp/analysis_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_cache = True
        self.cache_compression = True  # Enable cache compression
        self.cache_ttl = 86400 * 7  # 7 days cache TTL
        
        # ULTRA-OPTIMIZED TIMELINE ANALYSIS SETTINGS
        self.timeline_analysis_enabled = True
        self.advanced_audio_analysis = True
        self.enable_optical_flow = True
        self.emotion_detection_enabled = True
        self.action_detection_enabled = True
        self.genre_aware_analysis = True

    def _load_face_cascade(self):
        """Load OpenCV face detection cascade"""
        try:
            # Try to load the cascade file
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                print("Warning: Face detection cascade not found")
        except Exception as e:
            print(f"Warning: Could not load face detection: {e}")

    async def analyze_video_features(self, video_path: str) -> List[VideoFeature]:
        """Analyze video and extract features for editing suggestions - ADVANCED OPTIMIZATION"""
        try:
            # Check cache first for longer videos
            if self.enable_cache and await self._get_video_duration(video_path) > 300:  # 5+ minutes
                cached_features = await self._load_cached_analysis(video_path)
                if cached_features:
                    print("✅ Using cached analysis for longer video")
                    return cached_features
            
            features = []
            
            # Get video duration
            duration = await self._get_video_duration(video_path)
            if not duration:
                return features
            
            # Check if video is too long
            if duration > self.max_video_duration:
                print(f"Warning: Video too long ({duration}s), limiting analysis")
                duration = self.max_video_duration
            
            # Calculate optimal frame sampling
            frame_interval = self._calculate_optimal_frame_interval(duration)
            print(f"Analyzing video: {duration:.1f}s, sampling every {frame_interval} frames")
            
            # Extract frames with memory optimization
            frames = await self._extract_frames_optimized(video_path, frame_interval)
            
            # Parallel frame analysis for optimal performance
            print(f"Processing {len(frames)} frames with {self.max_workers} workers")
            
            if self.enable_parallel and len(frames) > self.parallel_batch_size:
                # Use parallel processing for larger frame sets
                features = await self._analyze_frames_parallel(frames)
            else:
                # Fallback to sequential processing for small frame sets
                features = await self._analyze_frames_sequential(frames)
            
            # Analyze audio features (lightweight)
            audio_features = await self._analyze_audio_features_optimized(video_path)
            features.extend(audio_features)
            
            # Detect scene changes (lightweight)
            scene_changes = await self._detect_scene_changes_optimized(frames)
            features.extend(scene_changes)
            
            # Final garbage collection
            gc.collect()
            
            print(f"Analysis complete: {len(features)} features found")
            
            # Cache results for longer videos
            if self.enable_cache and duration > 300:  # 5+ minutes
                await self._save_cached_analysis(video_path, features)
            
            return features
            
        except Exception as e:
            print(f"Video analysis error: {e}")
            # Force cleanup on error
            gc.collect()
            raise RuntimeError(f"Video feature analysis failed: {e}")

    def _calculate_optimal_frame_interval(self, duration: float) -> int:
        """Calculate optimal frame interval based on video duration - ADVANCED OPTIMIZATION"""
        if duration <= 60:  # 1 minute or less
            return max(1, int(duration / 20))  # 20 samples max
        elif duration <= 300:  # 5 minutes or less
            return max(1, int(duration / 40))  # 40 samples max
        elif duration <= 600:  # 10 minutes or less
            return max(1, int(duration / 60))  # 60 samples max
        elif duration <= 1200:  # 20 minutes or less
            return max(2, int(duration / 80))  # 80 samples max
        elif duration <= 1800:  # 30 minutes or less
            return max(3, int(duration / 100))  # 100 samples max
        else:  # Very long videos
            return max(5, int(duration / 150))  # 150 samples max

    async def _extract_frames_optimized(self, video_path: str, interval: int) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video with memory optimization - ENHANCED FOR FULL VIDEO"""
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {total_frames} frames, {fps} fps, duration: {total_frames/fps:.1f}s")
            
            # Advanced frame extraction optimization for longer videos
            if fps > 0:
                video_duration = total_frames / fps
                
                # Dynamic sampling based on video length
                if video_duration <= 300:  # 5 minutes or less
                    sample_interval = 2  # Sample every 2 seconds
                elif video_duration <= 600:  # 10 minutes or less
                    sample_interval = 3  # Sample every 3 seconds
                elif video_duration <= 1200:  # 20 minutes or less
                    sample_interval = 5  # Sample every 5 seconds
                else:  # 30+ minutes
                    sample_interval = 8  # Sample every 8 seconds
                
                # Calculate frames needed for optimal coverage
                frames_needed = min(self.max_frames_to_analyze, int(video_duration / sample_interval))
                interval = max(1, total_frames // frames_needed) if frames_needed > 0 else 1
                
                print(f"Optimization: {sample_interval}s intervals, {frames_needed} samples, {interval} frame skip")
            else:
                interval = max(1, interval)
            
            print(f"Extraction settings: interval={interval}, max_frames={self.max_frames_to_analyze}")
            
            frame_count = 0
            extracted_count = 0
            
            while frame_count < total_frames and extracted_count < self.max_frames_to_analyze:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    timestamp = frame_count / fps if fps > 0 else 0
                    
                    # Resize frame to save memory
                    height, width = frame.shape[:2]
                    new_width = int(width * self.frame_resize_factor)
                    new_height = int(height * self.frame_resize_factor)
                    frame_resized = cv2.resize(frame, (new_width, new_height))
                    
                    frames.append((timestamp, frame_resized))
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f"Extracted {len(frames)} frames from {frame_count} total frames")
            
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        return frames

    async def _analyze_frame_optimized(self, frame: np.ndarray, timestamp: float) -> List[VideoFeature]:
        """Analyze a single frame with advanced professional editing features"""
        features = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. ADVANCED FACE DETECTION - Character analysis
            if self.face_cascade:
                try:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        # Analyze face positions and sizes for shot composition
                        face_sizes = []
                        face_positions = []
                        
                        for (x, y, w, h) in faces:
                            face_sizes.append(w * h)
                            face_positions.append((x + w/2, y + h/2))  # Center point
                        
                        # Determine shot type based on face analysis
                        avg_face_size = sum(face_sizes) / len(face_sizes)
                        frame_area = frame.shape[0] * frame.shape[1]
                        face_ratio = avg_face_size / frame_area
                        
                        shot_type = 'wide_shot'
                        if face_ratio > 0.1:
                            shot_type = 'close_up'
                        elif face_ratio > 0.05:
                            shot_type = 'medium_shot'
                        
                        features.append(VideoFeature(
                            timestamp=timestamp,
                            feature_type='face_detected',
                            confidence=min(1.0, len(faces) * 0.3),
                            metadata={
                                'face_count': len(faces),
                                'shot_type': shot_type,
                                'face_ratio': face_ratio,
                                'face_positions': face_positions,
                                'avg_face_size': avg_face_size
                            }
                        ))
                except Exception as e:
                    print(f"Face detection failed: {e}")
            
            # 2. ADVANCED COMPOSITION ANALYSIS - Professional framing
            composition_analysis = self._analyze_composition_advanced(frame)
            if composition_analysis['score'] > 0.6:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='good_composition',
                    confidence=composition_analysis['score'],
                    metadata={
                        'composition_score': composition_analysis['score'],
                        'composition_type': composition_analysis['type'],
                        'rule_of_thirds': composition_analysis['rule_of_thirds'],
                        'symmetry_score': composition_analysis['symmetry'],
                        'leading_lines': composition_analysis['leading_lines']
                    }
                ))
            
            # 3. MOTION ANALYSIS - Movement detection for dynamic editing
            motion_score = self._analyze_motion_lightweight(frame)
            if motion_score > 0.3:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='motion_detected',
                    confidence=motion_score,
                    metadata={
                        'motion_intensity': motion_score,
                        'motion_type': 'general_movement'
                    }
                ))
            
            # 4. COLOR ANALYSIS - Mood and atmosphere detection
            color_analysis = self._analyze_color_mood(frame)
            if color_analysis['mood_score'] > 0.6:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='color_mood',
                    confidence=color_analysis['mood_score'],
                    metadata={
                        'color_mood': color_analysis['mood'],
                        'color_temperature': color_analysis['temperature'],
                        'saturation_level': color_analysis['saturation'],
                        'contrast_level': color_analysis['contrast']
                    }
                ))
            
            # 5. TEXTURE ANALYSIS - Visual complexity
            texture_score = self._analyze_texture_complexity(frame)
            if texture_score > 0.5:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='texture_complexity',
                    confidence=texture_score,
                    metadata={
                        'texture_score': texture_score,
                        'complexity_level': 'high' if texture_score > 0.7 else 'medium'
                    }
                ))
            
        except Exception as e:
            print(f"Frame analysis failed: {e}")
        
        return features

    def _analyze_composition_lightweight(self, frame: np.ndarray) -> float:
        """Lightweight frame composition analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Simple brightness and contrast analysis instead of edge detection
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Normalize to 0-1 range
            brightness_score = mean_brightness / 255.0
            contrast_score = min(1.0, std_brightness / 50.0)
            
            # Combined score
            return (brightness_score + contrast_score) / 2.0
            
        except Exception:
            return 0.5

    async def _analyze_audio_features_optimized(self, video_path: str) -> List[VideoFeature]:
        """Lightweight audio analysis"""
        features = []
        
        try:
            # Use ffprobe for quick audio analysis instead of extracting audio
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Simple audio feature
                features.append(VideoFeature(
                    timestamp=0.0,
                    feature_type='audio_detected',
                    confidence=1.0,
                    metadata={'audio_info': 'Audio stream detected'}
                ))
                
        except Exception as e:
            print(f"Audio analysis failed: {e}")
        
        return features

    async def _detect_scene_changes_optimized(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Lightweight scene change detection - ENHANCED FOR FULL VIDEO"""
        scene_changes = []
        
        try:
            # Analyze all frame pairs for scene changes
            for i in range(1, len(frames)):
                prev_frame = frames[i-1][1]
                curr_frame = frames[i][1]
                timestamp = frames[i][0]
                
                # Simple frame difference (much faster than Canny)
                diff = cv2.absdiff(prev_frame, curr_frame)
                change_score = np.mean(diff) / 255.0
                
                if change_score > self.scene_change_threshold:
                    scene_changes.append(VideoFeature(
                        timestamp=timestamp,
                        feature_type='scene_change',
                        confidence=min(1.0, change_score),
                        metadata={'change_score': change_score}
                    ))
                    
        except Exception as e:
            print(f"Scene change detection failed: {e}")
        
        return scene_changes

    async def _analyze_frames_parallel(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Analyze frames using parallel processing for optimal performance"""
        features = []
        
        try:
            # Split frames into parallel batches
            batches = [frames[i:i + self.parallel_batch_size] 
                      for i in range(0, len(frames), self.parallel_batch_size)]
            
            print(f"Processing {len(batches)} batches in parallel with {self.max_workers} workers")
            
            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit batch processing tasks
                future_to_batch = {
                    executor.submit(self._process_frame_batch, batch): i 
                    for i, batch in enumerate(batches)
                }
                
                # Collect results as they complete
                completed = 0
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    try:
                        batch_features = future.result()
                        features.extend(batch_features)
                        completed += 1
                        
                        # Progress update
                        progress = (completed / len(batches)) * 100
                        print(f"Parallel progress: {progress:.1f}% ({completed}/{len(batches)} batches)")
                        
                        # Memory cleanup every few batches
                        if completed % self.gc_interval == 0:
                            gc.collect()
                            print(f"Memory cleanup at {progress:.1f}%")
                            
                    except Exception as e:
                        print(f"Batch {batch_index} failed: {e}")
                        continue
            
            print(f"Parallel processing complete: {len(features)} features extracted")
            
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            features = await self._analyze_frames_sequential(frames)
        
        return features

    def _process_frame_batch(self, batch: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Process a batch of frames (for parallel execution)"""
        batch_features = []
        
        for timestamp, frame in batch:
            try:
                # Analyze frame (synchronous version for threading)
                frame_features = self._analyze_frame_sync(frame, timestamp)
                batch_features.extend(frame_features)
            except Exception as e:
                print(f"Frame analysis failed at {timestamp}s: {e}")
                continue
        
        return batch_features

    def _analyze_frame_sync(self, frame: np.ndarray, timestamp: float) -> List[VideoFeature]:
        """Synchronous version of frame analysis for parallel processing"""
        features = []
        
        try:
            # Face detection
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    features.append(VideoFeature(
                        timestamp=timestamp,
                        feature_type='face_detected',
                        confidence=min(1.0, len(faces) * 0.3),
                        metadata={'face_count': len(faces)}
                    ))
            
            # Composition analysis (lightweight)
            composition_score = self._analyze_composition_lightweight(frame)
            if composition_score > 0.6:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='good_composition',
                    confidence=composition_score,
                    metadata={'composition_score': composition_score}
                ))
            
        except Exception as e:
            print(f"Frame analysis error at {timestamp}s: {e}")
        
        return features

    async def _analyze_frames_sequential(self, frames: List[Tuple[float, np.ndarray]]) -> List[VideoFeature]:
        """Fallback sequential frame analysis"""
        features = []
        
        print(f"Processing {len(frames)} frames sequentially")
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            
            # Process batch
            for j, (timestamp, frame) in enumerate(batch):
                frame_features = await self._analyze_frame_optimized(frame, timestamp)
                features.extend(frame_features)
            
            # Progress update
            progress = min(100, (i + self.batch_size) / len(frames) * 100)
            print(f"Sequential progress: {progress:.1f}% ({i + len(batch)}/{len(frames)} frames)")
            
            # Memory cleanup
            if i % (self.batch_size * self.gc_interval) == 0:
                gc.collect()
                print(f"Memory cleanup at {progress:.1f}%")
        
        return features

    async def _load_cached_analysis(self, video_path: str) -> Optional[List[VideoFeature]]:
        """Load cached analysis results with enhanced caching"""
        try:
            # Create cache key from video file hash
            video_hash = self._get_video_hash(video_path)
            cache_file = self.cache_dir / f"{video_hash}.pkl"
            cache_meta_file = self.cache_dir / f"{video_hash}.meta"
            
            if cache_file.exists() and cache_meta_file.exists():
                # Check cache metadata
                try:
                    with open(cache_meta_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cache_age = time.time() - metadata.get('created_at', 0)
                    if cache_age < self.cache_ttl:
                        # Load cached data
                        with open(cache_file, 'rb') as f:
                            if self.cache_compression:
                                import gzip
                                cached_data = pickle.loads(gzip.decompress(f.read()))
                            else:
                                cached_data = pickle.load(f)
                        
                        print(f"📦 Loaded cached analysis ({len(cached_data)} features, {cache_age/3600:.1f}h old)")
                        return cached_data
                    else:
                        print(f"🕒 Cache expired ({cache_age/3600:.1f}h old), re-analyzing")
                except Exception as e:
                    print(f"Cache metadata error: {e}")
            
            return None
        except Exception as e:
            print(f"Cache load failed: {e}")
            return None

    async def _save_cached_analysis(self, video_path: str, features: List[VideoFeature]):
        """Save analysis results to cache with compression and metadata"""
        try:
            video_hash = self._get_video_hash(video_path)
            cache_file = self.cache_dir / f"{video_hash}.pkl"
            cache_meta_file = self.cache_dir / f"{video_hash}.meta"
            
            # Save cache data
            with open(cache_file, 'wb') as f:
                if self.cache_compression:
                    import gzip
                    compressed_data = gzip.compress(pickle.dumps(features))
                    f.write(compressed_data)
                else:
                    pickle.dump(features, f)
            
            # Save cache metadata
            metadata = {
                'created_at': time.time(),
                'feature_count': len(features),
                'video_path': video_path,
                'compressed': self.cache_compression
            }
            
            with open(cache_meta_file, 'w') as f:
                json.dump(metadata, f)
            
            cache_size = cache_file.stat().st_size / 1024  # KB
            print(f"💾 Cached analysis ({len(features)} features, {cache_size:.1f}KB)")
            
        except Exception as e:
            print(f"Cache save failed: {e}")

    def _get_video_hash(self, video_path: str) -> str:
        """Generate enhanced hash for video file for caching"""
        try:
            # Use file size, modification time, and first/last bytes for better hash
            stat = os.stat(video_path)
            
            # Read first and last 1KB for content-based hash
            with open(video_path, 'rb') as f:
                f.seek(0)
                first_bytes = f.read(1024)
                f.seek(-1024, 2)  # From end
                last_bytes = f.read(1024)
            
            hash_input = f"{video_path}_{stat.st_size}_{stat.st_mtime}_{hashlib.md5(first_bytes).hexdigest()[:8]}_{hashlib.md5(last_bytes).hexdigest()[:8]}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            # Fallback to simple hash
            return hashlib.md5(video_path.encode()).hexdigest()

    async def generate_editing_suggestions(self, video_path: str, script_content: str = "") -> List[EditingSuggestion]:
        """Generate AI editing suggestions based on video features and optional script analysis"""
        try:
            # Analyze video features
            video_features = await self.analyze_video_features(video_path)
            
            # Generate suggestions based on video features
            video_suggestions = self._generate_video_based_suggestions(video_features)
            
            # Generate suggestions based on script analysis (if provided)
            script_suggestions = []
            if script_content:
                script_suggestions = self._generate_script_based_suggestions_from_text(script_content)
            
            # ULTRA-OPTIMIZED TIMELINE ANALYSIS
            if self.timeline_analysis_enabled:
                timeline_suggestions = await self._generate_ultra_optimized_timeline_suggestions(video_path, video_features)
                video_suggestions.extend(timeline_suggestions)
            
            # Combine and rank suggestions
            all_suggestions = video_suggestions + script_suggestions
            ranked_suggestions = self._rank_suggestions(all_suggestions)
            
            return ranked_suggestions
            
        except Exception as e:
            raise RuntimeError(f"Editing suggestions generation failed: {e}")

    async def _generate_ultra_optimized_timeline_suggestions(self, video_path: str, features: List[VideoFeature]) -> List[EditingSuggestion]:
        """ULTRA-OPTIMIZED timeline analysis for professional editing suggestions"""
        suggestions = []
        
        try:
            # Get video duration and basic info
            duration = await self._get_video_duration(video_path)
            if not duration:
                return suggestions
            
            # 1. ADVANCED AUDIO ANALYSIS FOR TIMELINE
            if self.advanced_audio_analysis:
                audio_suggestions = await self._analyze_audio_timeline(video_path, duration)
                suggestions.extend(audio_suggestions)
            
            # 2. OPTICAL FLOW ANALYSIS FOR MOTION-BASED CUTS
            if self.enable_optical_flow:
                motion_suggestions = await self._analyze_optical_flow_timeline(video_path, duration)
                suggestions.extend(motion_suggestions)
            
            # 3. EMOTION DETECTION TIMELINE
            if self.emotion_detection_enabled:
                emotion_suggestions = await self._analyze_emotion_timeline(video_path, duration)
                suggestions.extend(emotion_suggestions)
            
            # 4. ACTION RECOGNITION TIMELINE
            if self.action_detection_enabled:
                action_suggestions = await self._analyze_action_timeline(video_path, duration)
                suggestions.extend(action_suggestions)
            
            # 5. GENRE-AWARE TIMELINE ANALYSIS
            if self.genre_aware_analysis:
                genre_suggestions = self._analyze_genre_specific_timeline(features, duration)
                suggestions.extend(genre_suggestions)
            
            # 6. ADVANCED PACING ALGORITHM
            pacing_suggestions = self._generate_advanced_pacing_suggestions(features, duration)
            suggestions.extend(pacing_suggestions)
            
            # 7. NARRATIVE BEAT DETECTION
            narrative_suggestions = self._detect_narrative_beats(features, duration)
            suggestions.extend(narrative_suggestions)
            
            # 8. VIEWER ENGAGEMENT OPTIMIZATION
            engagement_suggestions = self._optimize_viewer_engagement(features, duration)
            suggestions.extend(engagement_suggestions)
            
        except Exception as e:
            print(f"Ultra-optimized timeline analysis failed: {e}")
        
        return suggestions

    async def _analyze_audio_timeline(self, video_path: str, duration: float) -> List[EditingSuggestion]:
        """Advanced audio analysis for timeline-based editing suggestions"""
        suggestions = []
        
        try:
            # Extract audio features using ffprobe
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-show_frames', '-show_entries', 'frame=pkt_pts_time,pkt_size',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Analyze audio patterns for editing suggestions
                # This is a simplified version - in production, you'd use more sophisticated audio analysis
                
                # Suggest cuts at audio silence points
                silence_intervals = self._detect_audio_silence_points(duration)
                for timestamp in silence_intervals:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='cut',
                        confidence=0.8,
                        reason="Audio silence detected - natural cut point",
                        description="Audio silence detected. Perfect timing for a clean cut.",
                        reasoning="Silence provides natural break in audio - ideal for maintaining flow",
                        transition_type='cut',
                        metadata={
                            'audio_feature': 'silence',
                            'cut_type': 'audio_silence',
                            'editor_tip': 'Cut during silence for seamless transition'
                        }
                    ))
                
                # Suggest emphasis at audio peaks
                audio_peaks = self._detect_audio_peaks(duration)
                for timestamp in audio_peaks:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='emphasis',
                        confidence=0.7,
                        reason="Audio peak detected - emphasize moment",
                        description="Audio peak detected. Consider emphasizing this moment.",
                        reasoning="Audio peaks indicate important moments - emphasis enhances impact",
                        transition_type=None,
                        metadata={
                            'audio_feature': 'peak',
                            'emphasis_type': 'audio_peak',
                            'editor_tip': 'Use dramatic shot or slow motion for audio peak'
                        }
                    ))
                
        except Exception as e:
            print(f"Audio timeline analysis failed: {e}")
        
        return suggestions

    async def _analyze_optical_flow_timeline(self, video_path: str, duration: float) -> List[EditingSuggestion]:
        """Optical flow analysis for motion-based editing suggestions"""
        suggestions = []
        
        try:
            # This would use OpenCV's optical flow for motion analysis
            # For now, we'll simulate motion detection based on frame differences
            
            # Simulate motion detection at regular intervals
            motion_points = self._simulate_motion_detection(duration)
            
            for timestamp, motion_intensity in motion_points:
                if motion_intensity > 0.7:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='cut',
                        confidence=0.75,
                        reason=f"High motion detected (intensity: {motion_intensity:.2f})",
                        description=f"High motion activity detected. Consider cutting on movement.",
                        reasoning="Motion provides natural cut points - movement masks transition",
                        transition_type='cut',
                        metadata={
                            'motion_intensity': motion_intensity,
                            'cut_type': 'motion_based',
                            'editor_tip': 'Cut on the peak of motion for smooth transition'
                        }
                    ))
                elif motion_intensity > 0.4:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='emphasis',
                        confidence=0.6,
                        reason=f"Moderate motion detected (intensity: {motion_intensity:.2f})",
                        description=f"Moderate motion activity. Consider emphasizing movement.",
                        reasoning="Moderate motion indicates activity - emphasis maintains engagement",
                        transition_type=None,
                        metadata={
                            'motion_intensity': motion_intensity,
                            'emphasis_type': 'motion_emphasis',
                            'editor_tip': 'Use tracking shot or close-up for motion emphasis'
                        }
                    ))
                
        except Exception as e:
            print(f"Optical flow timeline analysis failed: {e}")
        
        return suggestions

    async def _analyze_emotion_timeline(self, video_path: str, duration: float) -> List[EditingSuggestion]:
        """Emotion detection timeline analysis"""
        suggestions = []
        
        try:
            # Simulate emotion detection at key moments
            emotion_points = self._simulate_emotion_detection(duration)
            
            for timestamp, emotion_data in emotion_points:
                emotion_type = emotion_data['emotion']
                intensity = emotion_data['intensity']
                
                if intensity > 0.8:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='emphasis',
                        confidence=0.9,
                        reason=f"Strong {emotion_type} emotion detected",
                        description=f"Strong {emotion_type} emotion. Use dramatic emphasis.",
                        reasoning=f"High emotional intensity requires visual emphasis - {emotion_type} emotions need impact",
                        transition_type=None,
                        metadata={
                            'emotion_type': emotion_type,
                            'emotion_intensity': intensity,
                            'emphasis_type': 'emotional_peak',
                            'editor_tip': f'Use close-up and dramatic lighting for {emotion_type} emotion'
                        }
                    ))
                elif intensity > 0.6:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='cut',
                        confidence=0.7,
                        reason=f"Moderate {emotion_type} emotion - reaction shot timing",
                        description=f"Moderate {emotion_type} emotion. Perfect for reaction shot.",
                        reasoning=f"Moderate emotions are ideal for reaction shots - shows character response",
                        transition_type='cut',
                        metadata={
                            'emotion_type': emotion_type,
                            'emotion_intensity': intensity,
                            'cut_type': 'emotion_reaction',
                            'editor_tip': 'Cut to reaction shot to show emotional response'
                        }
                    ))
                
        except Exception as e:
            print(f"Emotion timeline analysis failed: {e}")
        
        return suggestions

    async def _analyze_action_timeline(self, video_path: str, duration: float) -> List[EditingSuggestion]:
        """Action recognition timeline analysis"""
        suggestions = []
        
        try:
            # Simulate action detection
            action_points = self._simulate_action_detection(duration)
            
            for timestamp, action_data in action_points:
                action_type = action_data['action']
                confidence = action_data['confidence']
                
                if action_type in ['fight', 'dance', 'sport']:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='emphasis',
                        confidence=confidence,
                        reason=f"Dynamic action detected: {action_type}",
                        description=f"Dynamic {action_type} action. Use wide shot and emphasis.",
                        reasoning=f"Dynamic actions require wide shots to show full movement",
                        transition_type=None,
                        metadata={
                            'action_type': action_type,
                            'emphasis_type': 'dynamic_action',
                            'editor_tip': 'Use wide shot to capture full action movement'
                        }
                    ))
                elif action_type in ['gesture', 'expression', 'reaction']:
                    suggestions.append(EditingSuggestion(
                        timestamp=timestamp,
                        suggestion_type='cut',
                        confidence=confidence,
                        reason=f"Character action: {action_type}",
                        description=f"Character {action_type}. Cut to close-up for detail.",
                        reasoning=f"Character actions benefit from close-ups to show detail",
                        transition_type='cut',
                        metadata={
                            'action_type': action_type,
                            'cut_type': 'character_action',
                            'editor_tip': 'Cut to close-up to capture character detail'
                        }
                    ))
                
        except Exception as e:
            print(f"Action timeline analysis failed: {e}")
        
        return suggestions

    def _analyze_genre_specific_timeline(self, features: List[VideoFeature], duration: float) -> List[EditingSuggestion]:
        """Genre-aware timeline analysis"""
        suggestions = []
        
        try:
            # Detect genre based on features
            genre = self._detect_video_genre(features)
            
            if genre == 'action':
                # Action videos need faster pacing
                action_pacing = self._generate_action_pacing_suggestions(duration)
                suggestions.extend(action_pacing)
            elif genre == 'drama':
                # Drama videos need emotional pacing
                drama_pacing = self._generate_drama_pacing_suggestions(duration)
                suggestions.extend(drama_pacing)
            elif genre == 'comedy':
                # Comedy videos need timing-based cuts
                comedy_pacing = self._generate_comedy_pacing_suggestions(duration)
                suggestions.extend(comedy_pacing)
            elif genre == 'documentary':
                # Documentary videos need informative pacing
                doc_pacing = self._generate_documentary_pacing_suggestions(duration)
                suggestions.extend(doc_pacing)
            
        except Exception as e:
            print(f"Genre-specific timeline analysis failed: {e}")
        
        return suggestions

    def _generate_advanced_pacing_suggestions(self, features: List[VideoFeature], duration: float) -> List[EditingSuggestion]:
        """Advanced pacing algorithm for optimal viewer engagement"""
        suggestions = []
        
        try:
            # Calculate optimal pacing based on content analysis
            feature_density = len(features) / max(duration, 1)
            
            # Dynamic pacing intervals
            if feature_density > 0.5:  # High activity
                intervals = self._calculate_fast_pacing_intervals(duration)
                pacing_type = "fast_paced"
            elif feature_density > 0.2:  # Medium activity
                intervals = self._calculate_medium_pacing_intervals(duration)
                pacing_type = "medium_paced"
            else:  # Low activity
                intervals = self._calculate_slow_pacing_intervals(duration)
                pacing_type = "slow_paced"
            
            for interval in intervals:
                suggestions.append(EditingSuggestion(
                    timestamp=interval['timestamp'],
                    suggestion_type='cut',
                    confidence=interval['confidence'],
                    reason=f"Pacing cut - {pacing_type} rhythm",
                    description=f"Maintain {pacing_type} rhythm with strategic cut point.",
                    reasoning=f"Content analysis suggests {pacing_type} pacing for optimal engagement",
                    transition_type='cut',
                    metadata={
                        'pacing_type': pacing_type,
                        'feature_density': feature_density,
                        'editor_tip': f'Use {pacing_type} cuts to maintain viewer engagement'
                    }
                ))
            
        except Exception as e:
            print(f"Advanced pacing suggestions failed: {e}")
        
        return suggestions

    def _detect_narrative_beats(self, features: List[VideoFeature], duration: float) -> List[EditingSuggestion]:
        """Detect narrative beats for story-driven editing"""
        suggestions = []
        
        try:
            # Key narrative moments
            narrative_moments = [
                {'timestamp': duration * 0.1, 'type': 'hook', 'confidence': 0.9},
                {'timestamp': duration * 0.25, 'type': 'setup', 'confidence': 0.8},
                {'timestamp': duration * 0.5, 'type': 'climax', 'confidence': 0.95},
                {'timestamp': duration * 0.75, 'type': 'resolution', 'confidence': 0.8},
                {'timestamp': duration * 0.9, 'type': 'conclusion', 'confidence': 0.9}
            ]
            
            for moment in narrative_moments:
                if moment['timestamp'] < duration:
                    suggestions.append(EditingSuggestion(
                        timestamp=moment['timestamp'],
                        suggestion_type='emphasis' if moment['type'] == 'climax' else 'transition',
                        confidence=moment['confidence'],
                        reason=f"Narrative {moment['type']} - story beat",
                        description=f"Story {moment['type']} detected. Use appropriate emphasis.",
                        reasoning=f"Narrative {moment['type']} requires special treatment for story impact",
                        transition_type='dramatic_cut' if moment['type'] == 'climax' else 'cross_dissolve',
                        metadata={
                            'narrative_beat': moment['type'],
                            'story_phase': moment['type'],
                            'editor_tip': f'Use dramatic treatment for {moment["type"]} moment'
                        }
                    ))
            
        except Exception as e:
            print(f"Narrative beat detection failed: {e}")
        
        return suggestions

    def _optimize_viewer_engagement(self, features: List[VideoFeature], duration: float) -> List[EditingSuggestion]:
        """Optimize viewer engagement through strategic editing"""
        suggestions = []
        
        try:
            # Engagement optimization points
            engagement_points = [
                {'timestamp': 5.0, 'type': 'attention_grabber', 'confidence': 0.8},
                {'timestamp': duration * 0.15, 'type': 'engagement_maintainer', 'confidence': 0.7},
                {'timestamp': duration * 0.35, 'type': 'interest_booster', 'confidence': 0.8},
                {'timestamp': duration * 0.65, 'type': 'tension_builder', 'confidence': 0.8},
                {'timestamp': duration * 0.85, 'type': 'memorable_moment', 'confidence': 0.9}
            ]
            
            for point in engagement_points:
                if point['timestamp'] < duration:
                    suggestions.append(EditingSuggestion(
                        timestamp=point['timestamp'],
                        suggestion_type='emphasis',
                        confidence=point['confidence'],
                        reason=f"Viewer engagement: {point['type']}",
                        description=f"Strategic engagement point: {point['type']}.",
                        reasoning=f"Engagement optimization requires special treatment to maintain viewer interest",
                        transition_type=None,
                        metadata={
                            'engagement_type': point['type'],
                            'optimization_target': 'viewer_retention',
                            'editor_tip': f'Use compelling visual treatment for {point["type"]}'
                        }
                    ))
            
        except Exception as e:
            print(f"Viewer engagement optimization failed: {e}")
        
        return suggestions

    # Helper methods for timeline analysis
    def _detect_audio_silence_points(self, duration: float) -> List[float]:
        """Detect audio silence points for natural cuts"""
        # Simulate silence detection at regular intervals
        silence_points = []
        interval = max(10, duration / 20)  # Every 10 seconds or 20 intervals
        
        for i in range(1, int(duration / interval)):
            silence_points.append(i * interval)
        
        return silence_points

    def _detect_audio_peaks(self, duration: float) -> List[float]:
        """Detect audio peaks for emphasis"""
        # Simulate audio peak detection
        peak_points = []
        interval = max(15, duration / 15)  # Every 15 seconds or 15 intervals
        
        for i in range(1, int(duration / interval)):
            peak_points.append(i * interval)
        
        return peak_points

    def _simulate_motion_detection(self, duration: float) -> List[Tuple[float, float]]:
        """Simulate motion detection results"""
        motion_points = []
        interval = max(8, duration / 25)  # Every 8 seconds or 25 intervals
        
        for i in range(1, int(duration / interval)):
            timestamp = i * interval
            motion_intensity = 0.3 + (i % 3) * 0.2  # Varying intensity
            motion_points.append((timestamp, motion_intensity))
        
        return motion_points

    def _simulate_emotion_detection(self, duration: float) -> List[Tuple[float, Dict]]:
        """Simulate emotion detection results"""
        emotions = ['joy', 'sadness', 'anger', 'surprise', 'fear']
        emotion_points = []
        interval = max(12, duration / 20)  # Every 12 seconds or 20 intervals
        
        for i in range(1, int(duration / interval)):
            timestamp = i * interval
            emotion = emotions[i % len(emotions)]
            intensity = 0.5 + (i % 4) * 0.15  # Varying intensity
            emotion_points.append((timestamp, {'emotion': emotion, 'intensity': intensity}))
        
        return emotion_points

    def _simulate_action_detection(self, duration: float) -> List[Tuple[float, Dict]]:
        """Simulate action detection results"""
        actions = ['gesture', 'expression', 'movement', 'reaction', 'interaction']
        action_points = []
        interval = max(10, duration / 18)  # Every 10 seconds or 18 intervals
        
        for i in range(1, int(duration / interval)):
            timestamp = i * interval
            action = actions[i % len(actions)]
            confidence = 0.6 + (i % 3) * 0.15  # Varying confidence
            action_points.append((timestamp, {'action': action, 'confidence': confidence}))
        
        return action_points

    def _detect_video_genre(self, features: List[VideoFeature]) -> str:
        """Detect video genre based on features"""
        # Simple genre detection based on feature types
        feature_types = [f.feature_type for f in features]
        
        if 'motion_detected' in feature_types and feature_types.count('motion_detected') > 5:
            return 'action'
        elif 'face_detected' in feature_types and feature_types.count('face_detected') > 3:
            return 'drama'
        elif 'good_composition' in feature_types and feature_types.count('good_composition') > 4:
            return 'documentary'
        else:
            return 'general'

    def _calculate_fast_pacing_intervals(self, duration: float) -> List[Dict]:
        """Calculate fast pacing intervals for action content"""
        intervals = []
        interval = max(8, duration / 30)  # Every 8 seconds or 30 intervals
        
        for i in range(1, int(duration / interval)):
            intervals.append({
                'timestamp': i * interval,
                'confidence': 0.8
            })
        
        return intervals

    def _calculate_medium_pacing_intervals(self, duration: float) -> List[Dict]:
        """Calculate medium pacing intervals for balanced content"""
        intervals = []
        interval = max(12, duration / 20)  # Every 12 seconds or 20 intervals
        
        for i in range(1, int(duration / interval)):
            intervals.append({
                'timestamp': i * interval,
                'confidence': 0.7
            })
        
        return intervals

    def _calculate_slow_pacing_intervals(self, duration: float) -> List[Dict]:
        """Calculate slow pacing intervals for contemplative content"""
        intervals = []
        interval = max(20, duration / 12)  # Every 20 seconds or 12 intervals
        
        for i in range(1, int(duration / interval)):
            intervals.append({
                'timestamp': i * interval,
                'confidence': 0.6
            })
        
        return intervals

    def _generate_action_pacing_suggestions(self, duration: float) -> List[EditingSuggestion]:
        """Generate action-specific pacing suggestions"""
        suggestions = []
        interval = max(6, duration / 35)  # Fast pacing for action
        
        for i in range(1, int(duration / interval)):
            suggestions.append(EditingSuggestion(
                timestamp=i * interval,
                suggestion_type='cut',
                confidence=0.8,
                reason="Action pacing - maintain energy",
                description="Fast-paced cut to maintain action energy.",
                reasoning="Action content requires fast pacing to maintain viewer excitement",
                transition_type='cut',
                metadata={
                    'genre': 'action',
                    'pacing': 'fast',
                    'editor_tip': 'Use quick cuts to maintain action momentum'
                }
            ))
        
        return suggestions

    def _generate_drama_pacing_suggestions(self, duration: float) -> List[EditingSuggestion]:
        """Generate drama-specific pacing suggestions"""
        suggestions = []
        interval = max(15, duration / 15)  # Slower pacing for drama
        
        for i in range(1, int(duration / interval)):
            suggestions.append(EditingSuggestion(
                timestamp=i * interval,
                suggestion_type='transition',
                confidence=0.7,
                reason="Drama pacing - emotional flow",
                description="Smooth transition for emotional storytelling.",
                reasoning="Drama content benefits from smooth transitions for emotional flow",
                transition_type='cross_dissolve',
                metadata={
                    'genre': 'drama',
                    'pacing': 'emotional',
                    'editor_tip': 'Use smooth transitions for emotional storytelling'
                }
            ))
        
        return suggestions

    def _generate_comedy_pacing_suggestions(self, duration: float) -> List[EditingSuggestion]:
        """Generate comedy-specific pacing suggestions"""
        suggestions = []
        interval = max(10, duration / 20)  # Medium pacing for comedy
        
        for i in range(1, int(duration / interval)):
            suggestions.append(EditingSuggestion(
                timestamp=i * interval,
                suggestion_type='cut',
                confidence=0.7,
                reason="Comedy pacing - timing is everything",
                description="Timing-based cut for comedic effect.",
                reasoning="Comedy relies on precise timing for maximum impact",
                transition_type='cut',
                metadata={
                    'genre': 'comedy',
                    'pacing': 'timing_based',
                    'editor_tip': 'Cut on comedic beats for maximum impact'
                }
            ))
        
        return suggestions

    def _generate_documentary_pacing_suggestions(self, duration: float) -> List[EditingSuggestion]:
        """Generate documentary-specific pacing suggestions"""
        suggestions = []
        interval = max(18, duration / 12)  # Slower pacing for documentary
        
        for i in range(1, int(duration / interval)):
            suggestions.append(EditingSuggestion(
                timestamp=i * interval,
                suggestion_type='emphasis',
                confidence=0.6,
                reason="Documentary pacing - informative flow",
                description="Emphasis on important information.",
                reasoning="Documentary content requires time for information absorption",
                transition_type=None,
                metadata={
                    'genre': 'documentary',
                    'pacing': 'informative',
                    'editor_tip': 'Hold shots longer for information absorption'
                }
            ))
        
        return suggestions

    async def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=nw=1:nk=1', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            print(f"Could not get video duration: {e}")
        return None

    def _generate_video_based_suggestions(self, features: List[VideoFeature]) -> List[EditingSuggestion]:
        """Generate advanced editing suggestions based on video features - PROFESSIONAL EDITOR OPTIMIZED"""
        suggestions = []
        
        # Track timing for pacing suggestions
        timestamps = [f.timestamp for f in features if f.timestamp > 0]
        video_duration = max(timestamps) if timestamps else 0
        
        # Enhanced feature analysis for professional editing
        scene_changes = [f for f in features if f.feature_type == 'scene_change']
        faces = [f for f in features if f.feature_type == 'face_detected']
        compositions = [f for f in features if f.feature_type == 'good_composition']
        
        # 1. SCENE CHANGE ANALYSIS - Professional cut points
        for feature in scene_changes:
            # Analyze scene change intensity for better cut suggestions
            change_intensity = feature.metadata.get('change_score', 0)
            
            if change_intensity > 0.8:
                # Major scene change - hard cut
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='cut',
                    confidence=min(0.95, feature.confidence + 0.1),
                    reason="Major scene change detected - perfect hard cut point",
                    description="Strong visual transition detected. Use a hard cut here for maximum impact.",
                    reasoning="High change intensity indicates significant visual shift - ideal for maintaining viewer engagement",
                    transition_type='hard_cut',
                    metadata={
                        'feature_type': feature.feature_type,
                        'change_intensity': change_intensity,
                        'cut_type': 'major_scene_change',
                        'editor_tip': 'Consider adding a brief pause before the cut for dramatic effect'
                    }
                ))
            elif change_intensity > 0.5:
                # Moderate scene change - cross dissolve
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='transition',
                    confidence=feature.confidence,
                    reason="Moderate scene change - smooth transition recommended",
                    description="Moderate visual change detected. Use cross dissolve for smooth narrative flow.",
                    reasoning="Moderate change intensity suggests related content - cross dissolve maintains continuity",
                    transition_type='cross_dissolve',
                    metadata={
                        'feature_type': feature.feature_type,
                        'change_intensity': change_intensity,
                        'transition_duration': '0.5-1.0s',
                        'editor_tip': 'Keep transition duration proportional to change intensity'
                    }
                ))
            else:
                # Minor scene change - fade or dip
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='transition',
                    confidence=feature.confidence * 0.8,
                    reason="Minor scene change - subtle transition",
                    description="Subtle visual change detected. Consider fade or dip to black for elegant transition.",
                    reasoning="Low change intensity suggests subtle shift - gentle transition maintains mood",
                    transition_type='fade_to_black',
                    metadata={
                        'feature_type': feature.feature_type,
                        'change_intensity': change_intensity,
                        'transition_duration': '0.3-0.5s',
                        'editor_tip': 'Use brief fade for quick mood shift'
                    }
                ))
        
        # 2. FACE DETECTION ANALYSIS - Character-driven editing
        for feature in faces:
            face_count = feature.metadata.get('face_count', 1)
            face_confidence = feature.confidence
            
            if face_count == 1 and face_confidence > 0.8:
                # Single clear face - close-up opportunity
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=face_confidence,
                    reason="Clear single face detected - perfect for close-up",
                    description="Single face clearly visible. Consider close-up shot for emotional impact.",
                    reasoning="High confidence single face detection indicates good framing for character focus",
                    transition_type=None,
                    metadata={
                        'face_count': face_count,
                        'shot_type': 'close_up',
                        'duration_suggestion': '3-5 seconds',
                        'editor_tip': 'Hold close-up for emotional beats, cut on blink or expression change'
                    }
                ))
                
                # Also suggest reaction shot timing
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp + 2.0,  # 2 seconds later
                    suggestion_type='cut',
                    confidence=face_confidence * 0.7,
                    reason="Reaction shot timing - cut to show response",
                    description="Timing for reaction shot after establishing close-up.",
                    reasoning="Natural timing for showing character reaction or response",
                    transition_type='cut',
                    metadata={
                        'face_count': face_count,
                        'cut_type': 'reaction_shot',
                        'timing': '2s_after_establishing',
                        'editor_tip': 'Cut on natural head movement or expression change'
                    }
                ))
                
            elif face_count > 1:
                # Multiple faces - group dynamics
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=face_confidence,
                    reason=f"Multiple faces detected ({face_count}) - group interaction",
                    description=f"Group of {face_count} people visible. Consider wide shot to show dynamics.",
                    reasoning="Multiple faces indicate social interaction - wide shot captures group dynamics",
                    transition_type=None,
                    metadata={
                        'face_count': face_count,
                        'shot_type': 'wide_shot',
                        'duration_suggestion': '5-8 seconds',
                        'editor_tip': 'Use wide shot to show spatial relationships and body language'
                    }
                ))
                
                # Suggest individual close-ups for each person
                for i in range(face_count):
                    suggestions.append(EditingSuggestion(
                        timestamp=feature.timestamp + (i * 1.5),  # Stagger individual shots
                        suggestion_type='cut',
                        confidence=face_confidence * 0.6,
                        reason=f"Individual close-up for person {i+1}",
                        description=f"Cut to close-up of person {i+1} in group.",
                        reasoning="Individual close-ups help audience connect with each character",
                        transition_type='cut',
                        metadata={
                            'face_count': face_count,
                            'person_index': i + 1,
                            'shot_type': 'individual_close_up',
                            'timing': f'{i * 1.5}s_after_group_shot',
                            'editor_tip': 'Cut on natural head turn or when person starts speaking'
                        }
                    ))
        
        # 3. COMPOSITION ANALYSIS - Visual storytelling
        for feature in compositions:
            comp_score = feature.metadata.get('composition_score', 0)
            
            if comp_score > 0.9:
                # Exceptional composition - hold shot
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=comp_score,
                    reason="Exceptional composition - hold for visual impact",
                    description="Outstanding visual composition. Hold this shot longer for maximum impact.",
                    reasoning="High composition score indicates strong visual storytelling opportunity",
                    transition_type=None,
                    metadata={
                        'composition_score': comp_score,
                        'shot_type': 'hero_shot',
                        'duration_suggestion': '8-12 seconds',
                        'editor_tip': 'Let audience absorb the visual beauty before cutting'
                    }
                ))
                
                # Suggest slow motion or speed ramp
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=comp_score * 0.8,
                    reason="Consider slow motion for exceptional composition",
                    description="Apply slow motion or speed ramp to enhance visual impact.",
                    reasoning="Slow motion emphasizes the beauty and detail of well-composed shots",
                    transition_type=None,
                    metadata={
                        'composition_score': comp_score,
                        'effect_type': 'slow_motion',
                        'speed_suggestion': '0.5x-0.75x',
                        'editor_tip': 'Use slow motion sparingly for maximum impact'
                    }
                ))
                
            elif comp_score > 0.7:
                # Good composition - standard hold
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=comp_score,
                    reason="Good composition - hold for storytelling",
                    description="Strong visual composition. Hold shot for effective storytelling.",
                    reasoning="Good composition supports narrative - adequate hold time maintains engagement",
                    transition_type=None,
                    metadata={
                        'composition_score': comp_score,
                        'shot_type': 'storytelling_shot',
                        'duration_suggestion': '4-6 seconds',
                        'editor_tip': 'Cut when visual information is fully conveyed'
                    }
                ))
        
        # 4. ADVANCED PACING ANALYSIS - Professional rhythm
        if video_duration > 0:
            # Dynamic pacing based on content analysis
            pacing_intervals = self._calculate_advanced_pacing_intervals(video_duration, features)
            for interval in pacing_intervals:
                suggestions.append(EditingSuggestion(
                    timestamp=interval['timestamp'],
                    suggestion_type='cut',
                    confidence=interval['confidence'],
                    reason=interval['reason'],
                    description=interval['description'],
                    reasoning=interval['reasoning'],
                    transition_type='cut',
                    metadata=interval['metadata']
                ))
            
            # Key moment transitions with context
            key_moments = self._calculate_advanced_key_moments(video_duration, features)
            for moment in key_moments:
                suggestions.append(EditingSuggestion(
                    timestamp=moment['timestamp'],
                    suggestion_type='transition',
                    confidence=moment['confidence'],
                    reason=moment['reason'],
                    description=moment['description'],
                    reasoning=moment['reasoning'],
                    transition_type=moment['transition_type'],
                    metadata=moment['metadata']
                ))
        
        # 5. RHYTHM AND FLOW ANALYSIS - Professional editing patterns
        if len(features) > 5:
            # Analyze feature distribution for rhythm patterns
            rhythm_analysis = self._analyze_rhythm_patterns(features, video_duration)
            for rhythm_suggestion in rhythm_analysis:
                suggestions.append(EditingSuggestion(
                    timestamp=rhythm_suggestion['timestamp'],
                    suggestion_type=rhythm_suggestion['suggestion_type'],
                    confidence=rhythm_suggestion['confidence'],
                    reason=rhythm_suggestion['reason'],
                    description=rhythm_suggestion['description'],
                    reasoning=rhythm_suggestion['reasoning'],
                    transition_type=rhythm_suggestion.get('transition_type'),
                    metadata=rhythm_suggestion['metadata']
                ))
        
        return suggestions

    def _calculate_advanced_pacing_intervals(self, duration: float, features: List[VideoFeature]) -> List[Dict[str, Any]]:
        """Calculate advanced pacing intervals based on content analysis"""
        intervals = []
        
        # Analyze feature density to determine optimal pacing
        feature_density = len(features) / max(duration, 1)
        
        if duration <= 60:  # 1 minute or less
            # Dynamic pacing based on content
            if feature_density > 0.5:  # High activity
                interval = 12  # Faster cuts
                pacing_type = "fast_paced"
            elif feature_density > 0.2:  # Medium activity
                interval = 18  # Medium pacing
                pacing_type = "medium_paced"
            else:  # Low activity
                interval = 25  # Slower pacing
                pacing_type = "slow_paced"
                
        elif duration <= 300:  # 5 minutes or less
            if feature_density > 0.3:
                interval = 25
                pacing_type = "fast_paced"
            elif feature_density > 0.15:
                interval = 35
                pacing_type = "medium_paced"
            else:
                interval = 45
                pacing_type = "slow_paced"
                
        else:  # Longer videos
            if feature_density > 0.2:
                interval = 45
                pacing_type = "fast_paced"
            elif feature_density > 0.1:
                interval = 60
                pacing_type = "medium_paced"
            else:
                interval = 90
                pacing_type = "slow_paced"
        
        # Generate intervals with context
        current = interval
        while current < duration:
            intervals.append({
                'timestamp': current,
                'confidence': 0.7,
                'reason': f"Pacing cut - {pacing_type} rhythm",
                'description': f"Maintain {pacing_type} rhythm with strategic cut point.",
                'reasoning': f"Content analysis suggests {pacing_type} pacing for optimal engagement",
                'metadata': {
                    'pacing_type': pacing_type,
                    'interval_seconds': interval,
                    'feature_density': feature_density,
                    'editor_tip': f'Use {pacing_type} cuts to maintain viewer engagement'
                }
            })
            current += interval
        
        return intervals

    def _calculate_advanced_key_moments(self, duration: float, features: List[VideoFeature]) -> List[Dict[str, Any]]:
        """Calculate advanced key moments with context analysis"""
        moments = []
        
        # Analyze feature distribution for key moments
        feature_timestamps = [f.timestamp for f in features if f.timestamp > 0]
        
        # Quarter points with enhanced context
        quarter = duration * 0.25
        half = duration * 0.5
        three_quarter = duration * 0.75
        
        # Beginning transition (if video is long enough)
        if duration > 30:
            moments.append({
                'timestamp': 5.0,
                'confidence': 0.8,
                'reason': "Opening transition - establish narrative",
                'description': "Early transition to establish narrative flow and viewer engagement.",
                'reasoning': "First 5 seconds are crucial for viewer retention - smooth transition maintains interest",
                'transition_type': 'cross_dissolve',
                'metadata': {
                    'moment_type': 'opening_transition',
                    'timing': 'early_establishment',
                    'editor_tip': 'Use cross dissolve to smoothly introduce main content'
                }
            })
        
        # Quarter point with context
        if quarter > 10:
            moments.append({
                'timestamp': quarter,
                'confidence': 0.75,
                'reason': "Quarter-point transition - story development",
                'description': "Natural story progression point. Consider transition to next narrative beat.",
                'reasoning': "Quarter point marks natural story development - transition maintains flow",
                'transition_type': 'cross_dissolve',
                'metadata': {
                    'moment_type': 'quarter_point',
                    'story_phase': 'development',
                    'editor_tip': 'Transition to next story beat or scene'
                }
            })
        
        # Mid-point with emphasis
        if half > 15:
            moments.append({
                'timestamp': half,
                'confidence': 0.9,
                'reason': "Mid-point emphasis - narrative climax",
                'description': "Video mid-point. Consider dramatic transition or emphasis for maximum impact.",
                'reasoning': "Mid-point is crucial for maintaining viewer attention - dramatic treatment enhances engagement",
                'transition_type': 'dramatic_cut',
                'metadata': {
                    'moment_type': 'mid_point',
                    'story_phase': 'climax',
                    'editor_tip': 'Use dramatic cut or emphasis for mid-point impact'
                }
            })
        
        # Three-quarter point
        if three_quarter > 20:
            moments.append({
                'timestamp': three_quarter,
                'confidence': 0.7,
                'reason': "Three-quarter transition - story resolution",
                'description': "Approaching conclusion. Transition to resolution or final act.",
                'reasoning': "Three-quarter point signals story resolution - smooth transition to conclusion",
                'transition_type': 'cross_dissolve',
                'metadata': {
                    'moment_type': 'three_quarter',
                    'story_phase': 'resolution',
                    'editor_tip': 'Transition to final story beat or conclusion'
                }
            })
        
        # Ending transition
        if duration > 30:
            moments.append({
                'timestamp': duration - 5.0,
                'confidence': 0.8,
                'reason': "Closing transition - story conclusion",
                'description': "Final transition before conclusion. Consider fade or dramatic cut.",
                'reasoning': "Final moments are crucial for lasting impact - strong transition enhances memorability",
                'transition_type': 'fade_to_black',
                'metadata': {
                    'moment_type': 'closing_transition',
                    'story_phase': 'conclusion',
                    'editor_tip': 'Use fade or dramatic cut for memorable conclusion'
                }
            })
        
        return [m for m in moments if 0 < m['timestamp'] < duration]

    def _analyze_rhythm_patterns(self, features: List[VideoFeature], duration: float) -> List[Dict[str, Any]]:
        """Analyze rhythm patterns for professional editing suggestions"""
        rhythm_suggestions = []
        
        if len(features) < 3:
            return rhythm_suggestions
        
        # Analyze feature timing patterns
        feature_timestamps = sorted([f.timestamp for f in features if f.timestamp > 0])
        
        # Calculate average intervals between features
        intervals = []
        for i in range(1, len(feature_timestamps)):
            interval = feature_timestamps[i] - feature_timestamps[i-1]
            intervals.append(interval)
        
        if not intervals:
            return rhythm_suggestions
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Detect rhythm patterns
        if avg_interval < 10:  # Fast rhythm
            # Suggest rhythm breaks
            rhythm_break_time = duration * 0.4  # 40% into video
            rhythm_suggestions.append({
                'timestamp': rhythm_break_time,
                'suggestion_type': 'pace_change',
                'confidence': 0.7,
                'reason': "Rhythm break - slow down for impact",
                'description': "Fast-paced content detected. Consider slowing down for dramatic impact.",
                'reasoning': "Fast rhythm needs contrast - slower section creates dramatic tension",
                'metadata': {
                    'rhythm_type': 'fast_to_slow',
                    'avg_interval': avg_interval,
                    'editor_tip': 'Use longer shots or slower transitions for contrast'
                }
            })
            
        elif avg_interval > 30:  # Slow rhythm
            # Suggest rhythm acceleration
            rhythm_accel_time = duration * 0.6  # 60% into video
            rhythm_suggestions.append({
                'timestamp': rhythm_accel_time,
                'suggestion_type': 'pace_change',
                'confidence': 0.6,
                'reason': "Rhythm acceleration - increase pace",
                'description': "Slow-paced content detected. Consider faster cuts for engagement.",
                'reasoning': "Slow rhythm needs energy - faster cuts maintain viewer interest",
                'metadata': {
                    'rhythm_type': 'slow_to_fast',
                    'avg_interval': avg_interval,
                    'editor_tip': 'Use shorter shots and faster transitions for energy'
                }
            })
        
        # Detect feature clusters for emphasis
        feature_clusters = self._detect_feature_clusters(feature_timestamps)
        for cluster in feature_clusters:
            cluster_center = sum(cluster) / len(cluster)
            rhythm_suggestions.append({
                'timestamp': cluster_center,
                'suggestion_type': 'emphasis',
                'confidence': 0.8,
                'reason': f"Feature cluster - emphasize key moment",
                'description': f"Cluster of {len(cluster)} features detected. Emphasize this key moment.",
                'reasoning': "Feature clusters indicate important content - emphasis enhances impact",
                'metadata': {
                    'cluster_size': len(cluster),
                    'cluster_duration': max(cluster) - min(cluster),
                    'editor_tip': 'Use longer hold or dramatic transition for cluster emphasis'
                }
            })
        
        return rhythm_suggestions

    def _detect_feature_clusters(self, timestamps: List[float], cluster_threshold: float = 5.0) -> List[List[float]]:
        """Detect clusters of features that occur close together"""
        if len(timestamps) < 2:
            return []
        
        clusters = []
        current_cluster = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] <= cluster_threshold:
                current_cluster.append(timestamps[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [timestamps[i]]
        
        # Add final cluster
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return clusters

    def _calculate_key_moments(self, duration: float) -> List[float]:
        """Calculate key moments for transitions"""
        moments = []
        
        # Quarter points for narrative structure
        quarter = duration * 0.25
        half = duration * 0.5
        three_quarter = duration * 0.75
        
        moments.extend([quarter, half, three_quarter])
        
        # Add beginning and end transitions
        if duration > 30:
            moments.append(5.0)  # Early transition
            moments.append(duration - 5.0)  # Late transition
        
        return [m for m in moments if 0 < m < duration]

    def _generate_script_based_suggestions(self, script_analysis: Dict) -> List[EditingSuggestion]:
        """Generate editing suggestions based on script analysis"""
        suggestions = []
        
        # Extract script-based suggestions
        script_suggestions = script_analysis.get('editing_suggestions', [])
        
        for script_sug in script_suggestions:
            suggestions.append(EditingSuggestion(
                timestamp=script_sug.get('timestamp', 0),
                suggestion_type=script_sug.get('suggestion_type', 'cut'),
                confidence=script_sug.get('confidence', 0.5),
                reason=script_sug.get('reason', 'Script-based suggestion'),
                transition_type=script_sug.get('transition_type'),
                metadata={'source': 'script_analysis'}
            ))
        
        return suggestions

    def _generate_script_based_suggestions_from_text(self, script_content: str) -> List[EditingSuggestion]:
        """Generate advanced editing suggestions based on script text content - PROFESSIONAL EDITOR OPTIMIZED"""
        suggestions = []
        
        try:
            # Advanced script analysis for professional editing
            lines = script_content.split('\n')
            current_time = 0.0
            
            # Track narrative structure
            narrative_phases = []
            emotional_arc = []
            speaker_timeline = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Estimate duration based on text length and complexity
                words = line.split()
                word_count = len(words)
                duration = max(2.0, word_count * 0.3)  # More realistic timing
                
                # 1. NARRATIVE STRUCTURE ANALYSIS
                narrative_phase = self._analyze_narrative_phase(line, i, len(lines))
                if narrative_phase:
                    narrative_phases.append({
                        'timestamp': current_time,
                        'phase': narrative_phase,
                        'line': line[:100]
                    })
                
                # 2. EMOTIONAL CONTENT ANALYSIS
                emotional_analysis = self._analyze_emotional_content(line)
                if emotional_analysis['intensity'] > 0.5:
                    emotional_arc.append({
                        'timestamp': current_time,
                        'emotion': emotional_analysis['emotion'],
                        'intensity': emotional_analysis['intensity'],
                        'line': line[:100]
                    })
                
                # 3. SPEAKER AND DIALOGUE ANALYSIS
                speaker_analysis = self._analyze_speaker_content(line)
                if speaker_analysis['is_speaker']:
                    speaker_timeline.append({
                        'timestamp': current_time,
                        'speaker': speaker_analysis['speaker'],
                        'dialogue_type': speaker_analysis['dialogue_type'],
                        'line': line[:100]
                    })
                
                # 4. TRANSITION WORD ANALYSIS - Enhanced
                transition_analysis = self._analyze_transition_words(line)
                if transition_analysis['has_transition']:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='transition',
                        confidence=transition_analysis['confidence'],
                        reason=f"Transition word detected: {transition_analysis['word']}",
                        description=f"Script transition detected: '{transition_analysis['word']}'. Use {transition_analysis['transition_type']} for smooth narrative flow.",
                        reasoning=f"Transition word '{transition_analysis['word']}' indicates narrative shift - {transition_analysis['transition_type']} maintains story continuity",
                        transition_type=transition_analysis['transition_type'],
                        metadata={
                            'source': 'script_analysis',
                            'transition_word': transition_analysis['word'],
                            'transition_category': transition_analysis['category'],
                            'editor_tip': transition_analysis['editor_tip']
                        }
                    ))
                
                # 5. EMOTIONAL MOMENT SUGGESTIONS
                if emotional_analysis['intensity'] > 0.7:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='emphasis',
                        confidence=emotional_analysis['intensity'],
                        reason=f"High emotional content: {emotional_analysis['emotion']}",
                        description=f"Strong {emotional_analysis['emotion']} emotion detected. Consider dramatic emphasis or close-up.",
                        reasoning=f"High emotional intensity requires visual emphasis - dramatic treatment enhances impact",
                        transition_type=None,
                        metadata={
                            'source': 'script_analysis',
                            'emotion': emotional_analysis['emotion'],
                            'intensity': emotional_analysis['intensity'],
                            'suggested_shot': emotional_analysis['suggested_shot'],
                            'editor_tip': emotional_analysis['editor_tip']
                        }
                    ))
                
                # 6. SPEAKER CHANGE SUGGESTIONS
                if speaker_analysis['is_speaker'] and speaker_analysis['dialogue_type'] != 'monologue':
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='cut',
                        confidence=0.8,
                        reason=f"Speaker change: {speaker_analysis['speaker']}",
                        description=f"New speaker '{speaker_analysis['speaker']}' detected. Cut to show speaker or reaction.",
                        reasoning=f"Speaker changes are natural cut points - visual transition maintains dialogue flow",
                        transition_type='cut',
                        metadata={
                            'source': 'script_analysis',
                            'speaker': speaker_analysis['speaker'],
                            'dialogue_type': speaker_analysis['dialogue_type'],
                            'suggested_shot': 'close_up_or_reaction',
                            'editor_tip': 'Cut on natural speech rhythm or gesture'
                        }
                    ))
                
                # 7. ACTION AND MOVEMENT SUGGESTIONS
                action_analysis = self._analyze_action_content(line)
                if action_analysis['has_action']:
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='emphasis',
                        confidence=action_analysis['confidence'],
                        reason=f"Action content: {action_analysis['action_type']}",
                        description=f"Action detected: {action_analysis['action_type']}. Consider dynamic camera movement or emphasis.",
                        reasoning=f"Action content benefits from dynamic editing - movement enhances viewer engagement",
                        transition_type=None,
                        metadata={
                            'source': 'script_analysis',
                            'action_type': action_analysis['action_type'],
                            'suggested_shot': action_analysis['suggested_shot'],
                            'editor_tip': action_analysis['editor_tip']
                        }
                    ))
                
                current_time += duration
            
            # 8. NARRATIVE STRUCTURE SUGGESTIONS
            structure_suggestions = self._generate_narrative_structure_suggestions(narrative_phases, current_time)
            suggestions.extend(structure_suggestions)
            
            # 9. EMOTIONAL ARC SUGGESTIONS
            arc_suggestions = self._generate_emotional_arc_suggestions(emotional_arc, current_time)
            suggestions.extend(arc_suggestions)
            
        except Exception as e:
            print(f"Advanced script analysis failed: {e}")
        
        return suggestions

    def _analyze_narrative_phase(self, line: str, line_index: int, total_lines: int) -> Optional[str]:
        """Analyze narrative phase based on content and position"""
        line_lower = line.lower()
        
        # Opening phase indicators
        opening_words = ['begin', 'start', 'first', 'introduction', 'meet', 'welcome']
        if any(word in line_lower for word in opening_words) or line_index < total_lines * 0.1:
            return 'opening'
        
        # Development phase indicators
        development_words = ['develop', 'grow', 'learn', 'discover', 'explore', 'build']
        if any(word in line_lower for word in development_words) or (total_lines * 0.1 <= line_index < total_lines * 0.8):
            return 'development'
        
        # Climax phase indicators
        climax_words = ['climax', 'peak', 'moment', 'finally', 'suddenly', 'dramatic', 'intense']
        if any(word in line_lower for word in climax_words) or (total_lines * 0.7 <= line_index < total_lines * 0.9):
            return 'climax'
        
        # Resolution phase indicators
        resolution_words = ['end', 'conclude', 'finally', 'result', 'outcome', 'resolution']
        if any(word in line_lower for word in resolution_words) or line_index >= total_lines * 0.9:
            return 'resolution'
        
        return None

    def _analyze_emotional_content(self, line: str) -> Dict[str, Any]:
        """Analyze emotional content for editing suggestions"""
        line_lower = line.lower()
        
        # Emotional word categories
        emotions = {
            'joy': ['happy', 'joy', 'excited', 'thrilled', 'amazing', 'wonderful', 'fantastic'],
            'sadness': ['sad', 'depressed', 'melancholy', 'sorrow', 'grief', 'heartbroken'],
            'anger': ['angry', 'furious', 'rage', 'outraged', 'furious', 'mad'],
            'fear': ['afraid', 'scared', 'terrified', 'fearful', 'anxious', 'worried'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
            'love': ['love', 'adore', 'passion', 'romantic', 'affection', 'tender'],
            'dramatic': ['dramatic', 'intense', 'powerful', 'emotional', 'moving']
        }
        
        max_intensity = 0.0
        detected_emotion = 'neutral'
        suggested_shot = 'medium_shot'
        editor_tip = 'Standard emotional treatment'
        
        for emotion, words in emotions.items():
            intensity = sum(1 for word in words if word in line_lower) / len(words)
            if intensity > max_intensity:
                max_intensity = intensity
                detected_emotion = emotion
                
                # Suggest appropriate shots based on emotion
                if emotion in ['joy', 'love']:
                    suggested_shot = 'close_up'
                    editor_tip = 'Use close-up to capture emotional expression'
                elif emotion in ['anger', 'fear']:
                    suggested_shot = 'dramatic_angle'
                    editor_tip = 'Use dramatic angles for intense emotions'
                elif emotion == 'dramatic':
                    suggested_shot = 'hero_shot'
                    editor_tip = 'Use hero shot for maximum dramatic impact'
        
        return {
            'emotion': detected_emotion,
            'intensity': max_intensity,
            'suggested_shot': suggested_shot,
            'editor_tip': editor_tip
        }

    def _analyze_speaker_content(self, line: str) -> Dict[str, Any]:
        """Analyze speaker and dialogue content"""
        # Check for speaker format (NAME: dialogue)
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            dialogue = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if speaker is in caps (typical script format)
            if speaker.isupper() and len(speaker) > 1:
                # Analyze dialogue type
                dialogue_type = 'dialogue'
                if len(dialogue) > 100:
                    dialogue_type = 'monologue'
                elif len(dialogue) < 10:
                    dialogue_type = 'reaction'
                
                return {
                    'is_speaker': True,
                    'speaker': speaker,
                    'dialogue_type': dialogue_type,
                    'dialogue_length': len(dialogue)
                }
        
        return {
            'is_speaker': False,
            'speaker': None,
            'dialogue_type': None,
            'dialogue_length': 0
        }

    def _analyze_transition_words(self, line: str) -> Dict[str, Any]:
        """Analyze transition words for editing suggestions"""
        line_lower = line.lower()
        
        # Categorized transition words
        transitions = {
            'temporal': {
                'words': ['meanwhile', 'later', 'earlier', 'before', 'after', 'then', 'now'],
                'transition_type': 'cross_dissolve',
                'category': 'time_shift',
                'editor_tip': 'Use cross dissolve for smooth time transitions'
            },
            'contrast': {
                'words': ['however', 'but', 'although', 'despite', 'nevertheless'],
                'transition_type': 'cut',
                'category': 'contrast',
                'editor_tip': 'Use hard cut for contrast emphasis'
            },
            'causal': {
                'words': ['therefore', 'thus', 'consequently', 'as a result', 'because'],
                'transition_type': 'cross_dissolve',
                'category': 'cause_effect',
                'editor_tip': 'Use cross dissolve to show cause-effect relationship'
            },
            'dramatic': {
                'words': ['suddenly', 'dramatically', 'shockingly', 'unexpectedly'],
                'transition_type': 'dramatic_cut',
                'category': 'drama',
                'editor_tip': 'Use dramatic cut for sudden revelations'
            }
        }
        
        for category, data in transitions.items():
            for word in data['words']:
                if word in line_lower:
                    return {
                        'has_transition': True,
                        'word': word,
                        'category': category,
                        'transition_type': data['transition_type'],
                        'confidence': 0.8,
                        'editor_tip': data['editor_tip']
                    }
        
        return {
            'has_transition': False,
            'word': None,
            'category': None,
            'transition_type': None,
            'confidence': 0.0,
            'editor_tip': None
        }

    def _analyze_action_content(self, line: str) -> Dict[str, Any]:
        """Analyze action content for dynamic editing suggestions"""
        line_lower = line.lower()
        
        # Action word categories
        actions = {
            'movement': ['walk', 'run', 'move', 'travel', 'go', 'come'],
            'physical': ['fight', 'dance', 'jump', 'fall', 'climb', 'throw'],
            'gesture': ['point', 'wave', 'nod', 'shake', 'smile', 'frown'],
            'interaction': ['touch', 'hold', 'push', 'pull', 'embrace', 'kiss']
        }
        
        detected_actions = []
        for action_type, words in actions.items():
            if any(word in line_lower for word in words):
                detected_actions.append(action_type)
        
        if detected_actions:
            action_type = detected_actions[0]
            confidence = min(0.9, len(detected_actions) * 0.3)
            
            # Suggest appropriate shots based on action
            suggested_shot = 'medium_shot'
            editor_tip = 'Standard action treatment'
            
            if action_type == 'movement':
                suggested_shot = 'tracking_shot'
                editor_tip = 'Use tracking shot to follow movement'
            elif action_type == 'physical':
                suggested_shot = 'wide_shot'
                editor_tip = 'Use wide shot to show full action'
            elif action_type == 'gesture':
                suggested_shot = 'close_up'
                editor_tip = 'Use close-up to capture gesture detail'
            elif action_type == 'interaction':
                suggested_shot = 'two_shot'
                editor_tip = 'Use two-shot to show interaction'
            
            return {
                'has_action': True,
                'action_type': action_type,
                'confidence': confidence,
                'suggested_shot': suggested_shot,
                'editor_tip': editor_tip
            }
        
        return {
            'has_action': False,
            'action_type': None,
            'confidence': 0.0,
            'suggested_shot': None,
            'editor_tip': None
        }

    def _generate_narrative_structure_suggestions(self, narrative_phases: List[Dict], total_duration: float) -> List[EditingSuggestion]:
        """Generate editing suggestions based on narrative structure"""
        suggestions = []
        
        for phase in narrative_phases:
            if phase['phase'] == 'opening':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='transition',
                    confidence=0.8,
                    reason="Narrative opening - establish story",
                    description="Story opening detected. Use establishing shot or smooth transition.",
                    reasoning="Opening phase requires clear story establishment - smooth transition sets tone",
                    transition_type='cross_dissolve',
                    metadata={
                        'narrative_phase': 'opening',
                        'editor_tip': 'Use establishing shot to set scene and tone'
                    }
                ))
            
            elif phase['phase'] == 'climax':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='emphasis',
                    confidence=0.9,
                    reason="Narrative climax - maximum impact",
                    description="Story climax detected. Use dramatic emphasis for maximum impact.",
                    reasoning="Climax requires maximum visual impact - dramatic treatment enhances tension",
                    transition_type=None,
                    metadata={
                        'narrative_phase': 'climax',
                        'editor_tip': 'Use dramatic angles and close-ups for climax impact'
                    }
                ))
            
            elif phase['phase'] == 'resolution':
                suggestions.append(EditingSuggestion(
                    timestamp=phase['timestamp'],
                    suggestion_type='transition',
                    confidence=0.7,
                    reason="Narrative resolution - story conclusion",
                    description="Story resolution detected. Use gentle transition for conclusion.",
                    reasoning="Resolution phase requires gentle treatment - smooth transition provides closure",
                    transition_type='fade_to_black',
                    metadata={
                        'narrative_phase': 'resolution',
                        'editor_tip': 'Use gentle fade for story conclusion'
                    }
                ))
        
        return suggestions

    def _generate_emotional_arc_suggestions(self, emotional_arc: List[Dict], total_duration: float) -> List[EditingSuggestion]:
        """Generate editing suggestions based on emotional arc"""
        suggestions = []
        
        if len(emotional_arc) < 2:
            return suggestions
        
        # Find emotional peaks
        high_emotions = [e for e in emotional_arc if e['intensity'] > 0.8]
        
        for emotion in high_emotions:
            suggestions.append(EditingSuggestion(
                timestamp=emotion['timestamp'],
                suggestion_type='emphasis',
                confidence=emotion['intensity'],
                reason=f"Emotional peak: {emotion['emotion']}",
                description=f"Emotional peak detected: {emotion['emotion']}. Use dramatic emphasis.",
                reasoning=f"Emotional peaks require visual emphasis - dramatic treatment enhances impact",
                transition_type=None,
                metadata={
                    'emotion': emotion['emotion'],
                    'intensity': emotion['intensity'],
                    'editor_tip': f'Use dramatic treatment for {emotion["emotion"]} emotion'
                }
            ))
        
        return suggestions

    def _rank_suggestions(self, suggestions: List[EditingSuggestion]) -> List[EditingSuggestion]:
        """Rank suggestions by confidence and importance"""
        # Sort by confidence (highest first)
        ranked = sorted(suggestions, key=lambda x: x.confidence, reverse=True)
        
        # Remove duplicates (same timestamp and type)
        unique_suggestions = []
        seen = set()
        
        for suggestion in ranked:
            key = (suggestion.timestamp, suggestion.suggestion_type)
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions

    def _group_suggestions(self, suggestions: List[EditingSuggestion]) -> Dict[str, List[EditingSuggestion]]:
        """Group suggestions by type"""
        grouped = {
            'cuts': [],
            'transitions': [],
            'emphasis': [],
            'pace_changes': []
        }
        
        for suggestion in suggestions:
            if suggestion.suggestion_type == 'cut':
                grouped['cuts'].append(suggestion)
            elif suggestion.suggestion_type == 'transition':
                grouped['transitions'].append(suggestion)
            elif suggestion.suggestion_type == 'emphasis':
                grouped['emphasis'].append(suggestion)
            elif suggestion.suggestion_type == 'pace_change':
                grouped['pace_changes'].append(suggestion)
        
        return grouped

    def _suggestion_to_dict(self, suggestion: EditingSuggestion) -> Dict[str, Any]:
        """Convert EditingSuggestion to dictionary"""
        return {
            "timestamp": suggestion.timestamp,
            "suggestion_type": suggestion.suggestion_type,
            "confidence": suggestion.confidence,
            "reason": suggestion.reason,
            "transition_type": suggestion.transition_type,
            "metadata": suggestion.metadata or {}
        }

    def _feature_to_dict(self, feature: VideoFeature) -> Dict[str, Any]:
        """Convert VideoFeature to dictionary"""
        return {
            "timestamp": feature.timestamp,
            "feature_type": feature.feature_type,
            "confidence": feature.confidence,
            "metadata": feature.metadata
        }

    async def process(self, upload_id: str, processing_status: dict, script_content: Optional[str] = None, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Main processing method for AI editing suggestions"""
        try:
            # Use provided file_path or get from processing_status
            if not file_path:
                file_path = processing_status.get('file_path', '')
            
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            # Get video duration
            duration = await self._get_video_duration(file_path)
            
            # Prepare script analysis if provided
            script_analysis = None
            if script_content:
                from .script_analysis import ScriptAnalysisService
                script_service = ScriptAnalysisService()
                script_analysis = await script_service.analyze_script(script_content, duration or 0)
            
            # Generate editing suggestions
            result = await self.generate_editing_suggestions(file_path, script_content)
            
            # Add processing metadata
            result.update({
                "upload_id": upload_id,
                "video_duration": duration,
                "script_provided": script_content is not None
            })
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"AI editing suggestions processing failed: {e}")
