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
        """Analyze a single frame with memory optimization"""
        features = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces (only if cascade is loaded)
            if self.face_cascade:
                try:
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        features.append(VideoFeature(
                            timestamp=timestamp,
                            feature_type='face_detected',
                            confidence=min(1.0, len(faces) * 0.3),
                            metadata={'face_count': len(faces)}
                        ))
                except Exception as e:
                    print(f"Face detection failed: {e}")
            
            # Lightweight composition analysis
            composition_score = self._analyze_composition_lightweight(frame)
            if composition_score > 0.7:
                features.append(VideoFeature(
                    timestamp=timestamp,
                    feature_type='good_composition',
                    confidence=composition_score,
                    metadata={'composition_score': composition_score}
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
            
            # Combine and rank suggestions
            all_suggestions = video_suggestions + script_suggestions
            ranked_suggestions = self._rank_suggestions(all_suggestions)
            
            return ranked_suggestions
            
        except Exception as e:
            raise RuntimeError(f"Editing suggestions generation failed: {e}")

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
        """Generate editing suggestions based on video features - ENHANCED VERSION"""
        suggestions = []
        
        # Track timing for pacing suggestions
        timestamps = [f.timestamp for f in features if f.timestamp > 0]
        video_duration = max(timestamps) if timestamps else 0
        
        for feature in features:
            if feature.feature_type == 'scene_change':
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='cut',
                    confidence=feature.confidence,
                    reason="Scene change detected",
                    description="Scene change detected - natural cut point",
                    reasoning="Scene change detected - natural cut point",
                    transition_type='cut',
                    metadata={'feature_type': feature.feature_type}
                ))
            
            elif feature.feature_type == 'face_detected':
                # Face detection can suggest both emphasis and cuts
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=feature.confidence,
                    reason="Face detected - consider close-up or emphasis",
                    description="Face detected - consider close-up or emphasis",
                    reasoning="Face detected - consider close-up or emphasis",
                    transition_type=None,
                    metadata={'face_count': feature.metadata.get('face_count', 1)}
                ))
                
                # Also suggest a cut if multiple faces or high confidence
                if feature.metadata.get('face_count', 1) > 1 or feature.confidence > 0.7:
                    suggestions.append(EditingSuggestion(
                        timestamp=feature.timestamp,
                        suggestion_type='cut',
                        confidence=feature.confidence * 0.8,
                        reason="Multiple faces or high-confidence face detection - good cut point",
                        description="Multiple faces or high-confidence face detection - good cut point",
                        reasoning="Multiple faces or high-confidence face detection - good cut point",
                        transition_type='cut',
                        metadata={'face_count': feature.metadata.get('face_count', 1)}
                    ))
            
            elif feature.feature_type == 'good_composition':
                suggestions.append(EditingSuggestion(
                    timestamp=feature.timestamp,
                    suggestion_type='emphasis',
                    confidence=feature.confidence,
                    reason="Strong composition - consider holding this shot",
                    description="Strong composition - consider holding this shot",
                    reasoning="Strong composition - consider holding this shot",
                    transition_type=None,
                    metadata={'composition_score': feature.metadata.get('composition_score', 0)}
                ))
                
                # Also suggest a transition if composition is very good
                if feature.confidence > 0.8:
                    suggestions.append(EditingSuggestion(
                        timestamp=feature.timestamp,
                        suggestion_type='transition',
                        confidence=feature.confidence * 0.7,
                        reason="Excellent composition - consider smooth transition",
                        description="Excellent composition - consider smooth transition",
                        reasoning="Excellent composition - consider smooth transition",
                        transition_type='cross_dissolve',
                        metadata={'composition_score': feature.metadata.get('composition_score', 0)}
                    ))
        
        # Add automatic pacing suggestions based on video structure
        if video_duration > 0:
            # Suggest cuts at regular intervals for pacing
            pacing_intervals = self._calculate_pacing_intervals(video_duration)
            for interval in pacing_intervals:
                suggestions.append(EditingSuggestion(
                    timestamp=interval,
                    suggestion_type='cut',
                    confidence=0.6,
                    reason="Pacing cut - maintain viewer engagement",
                    description="Pacing cut - maintain viewer engagement",
                    reasoning="Pacing cut - maintain viewer engagement",
                    transition_type='cut',
                    metadata={'pacing_type': 'regular_interval'}
                ))
            
            # Suggest transitions at key moments
            key_moments = self._calculate_key_moments(video_duration)
            for moment in key_moments:
                suggestions.append(EditingSuggestion(
                    timestamp=moment,
                    suggestion_type='transition',
                    confidence=0.7,
                    reason="Key moment transition - enhance narrative flow",
                    description="Key moment transition - enhance narrative flow",
                    reasoning="Key moment transition - enhance narrative flow",
                    transition_type='cross_dissolve',
                    metadata={'moment_type': 'key_timing'}
                ))
            
            # Add more comprehensive pacing suggestions for longer videos
            if video_duration > 300:  # 5+ minutes
                # Add mid-point emphasis
                mid_point = video_duration * 0.5
                suggestions.append(EditingSuggestion(
                    timestamp=mid_point,
                    suggestion_type='emphasis',
                    confidence=0.8,
                    reason="Mid-point emphasis - maintain viewer attention",
                    description="Mid-point emphasis - maintain viewer attention",
                    reasoning="Mid-point emphasis - maintain viewer attention",
                    transition_type=None,
                    metadata={'pacing_type': 'mid_point_emphasis'}
                ))
                
                # Add quarter-point transitions
                quarter_points = [video_duration * 0.25, video_duration * 0.75]
                for point in quarter_points:
                    suggestions.append(EditingSuggestion(
                        timestamp=point,
                        suggestion_type='transition',
                        confidence=0.6,
                        reason="Quarter-point transition - natural story progression",
                        description="Quarter-point transition - natural story progression",
                        reasoning="Quarter-point transition - natural story progression",
                        transition_type='cross_dissolve',
                        metadata={'pacing_type': 'quarter_point'}
                    ))
        
        # Add variety suggestions based on feature distribution
        if len(features) > 10:
            # Suggest pace changes if many features detected
            suggestions.append(EditingSuggestion(
                timestamp=video_duration * 0.3,  # 30% into video
                suggestion_type='pace_change',
                confidence=0.6,
                reason="High activity detected - consider faster pacing",
                description="High activity detected - consider faster pacing",
                reasoning="High activity detected - consider faster pacing",
                transition_type=None,
                metadata={'pace_direction': 'faster'}
            ))
        
        return suggestions

    def _calculate_pacing_intervals(self, duration: float) -> List[float]:
        """Calculate optimal pacing cut intervals"""
        intervals = []
        
        if duration <= 60:  # 1 minute or less
            # Cut every 15-20 seconds
            interval = max(15, duration / 4)
            current = interval
            while current < duration:
                intervals.append(current)
                current += interval
                
        elif duration <= 300:  # 5 minutes or less
            # Cut every 30-45 seconds
            interval = max(30, duration / 8)
            current = interval
            while current < duration:
                intervals.append(current)
                current += interval
                
        else:  # Longer videos
            # Cut every 60-90 seconds
            interval = max(60, duration / 12)
            current = interval
            while current < duration:
                intervals.append(current)
                current += interval
        
        return intervals

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
        """Generate editing suggestions based on script text content"""
        suggestions = []
        
        try:
            # Simple script analysis - look for transition words and emotional content
            lines = script_content.split('\n')
            current_time = 0.0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Estimate duration based on text length
                duration = max(2.0, len(line) * 0.1)  # At least 2 seconds, 0.1s per character
                
                # Look for transition words
                transition_words = ['meanwhile', 'suddenly', 'finally', 'however', 'therefore', 'meanwhile', 'later', 'earlier']
                if any(word in line.lower() for word in transition_words):
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='transition',
                        confidence=0.7,
                        reason=f"Transition word detected: {line[:50]}...",
                        description=f"Transition word detected: {line[:50]}...",
                        reasoning=f"Transition word detected: {line[:50]}...",
                        transition_type='cross_dissolve',
                        metadata={'source': 'script_analysis', 'transition_word': True}
                    ))
                
                # Look for emotional content
                emotional_words = ['dramatic', 'exciting', 'amazing', 'incredible', 'thrilling', 'shocking', 'surprising']
                if any(word in line.lower() for word in emotional_words):
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='emphasis',
                        confidence=0.6,
                        reason=f"Emotional content detected: {line[:50]}...",
                        description=f"Emotional content detected: {line[:50]}...",
                        reasoning=f"Emotional content detected: {line[:50]}...",
                        transition_type=None,
                        metadata={'source': 'script_analysis', 'emotional_content': True}
                    ))
                
                # Look for speaker changes
                if ':' in line and line.split(':')[0].strip().isupper():
                    suggestions.append(EditingSuggestion(
                        timestamp=current_time,
                        suggestion_type='cut',
                        confidence=0.5,
                        reason=f"Speaker change detected: {line[:50]}...",
                        description=f"Speaker change detected: {line[:50]}...",
                        reasoning=f"Speaker change detected: {line[:50]}...",
                        transition_type='cut',
                        metadata={'source': 'script_analysis', 'speaker_change': True}
                    ))
                
                current_time += duration
            
        except Exception as e:
            print(f"Script analysis failed: {e}")
        
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
