# AI Video Editor

A powerful AI-powered video editing application that combines multiple cutting-edge technologies to transform your videos with minimal effort.

## 🚀 Features

- **AI Video Optimization**: Complete AI pipeline with scene analysis, silence removal, subtitle generation, and quality optimization
- **AI Auto-Cut & Transitions**: Scene-aware cuts with smooth transitions and intelligent merging
- **AI Background Removal**: Advanced background removal with replacement options
- **AI Subtitle Generation**: Multi-language speech-to-text with SRT generation
- **AI Scene Detection**: Automatic video splitting with individual download links
- **Voice Translation & Dubbing**: Speech recognition, translation, and AI voice generation
- **AI Style Filters**: Neural style transfer with multiple artistic styles
- **AI Object Removal**: Bounding box selection with advanced inpainting

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.10+, FFmpeg, OpenCV
- **AI/ML**: OpenAI Whisper, Rembg, PySceneDetect, Coqui TTS, Google Translate, MediaPipe
- **Frontend**: Next.js 14, TypeScript, Tailwind CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- FFmpeg installed and in PATH
- Node.js 18 or higher

### Installation

1. **Clone and setup**
   ```bash
   git clone <repository-url>
   cd videoeditor
   pip install -r requirements.txt
   ```

2. **Start the application**
   ```bash
   # Start backend
   python run_backend.py
   
   # Or use the provided scripts
   ./start.bat          # Windows
   ./start.ps1          # PowerShell
   ```

3. **Access the application**
   - Open your browser to `http://localhost:8000`
   - Upload a video and start editing!

## 📖 Usage

1. **Upload Video**: Drag and drop or click to upload (MP4, MOV, WebM formats)
2. **Select Features**: Choose from individual AI features or use "AI Video Optimization"
3. **Process Video**: Click "Process Video" and watch real-time progress
4. **Download Results**: Download buttons appear automatically after processing

## 🔧 Configuration

Set environment variables for enhanced features:
```bash
OPENAI_API_KEY=your_openai_api_key      # For Whisper
GOOGLE_TRANSLATE_API_KEY=your_key       # For translation
```

## 🐛 Troubleshooting

- **FFmpeg not found**: Install FFmpeg and add to PATH
- **Processing fails**: Check video format and size (max 500MB, 10 minutes)
- **Download issues**: Check browser settings and disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**AI Video Editor** - Transform your videos with the power of AI! 🎬✨
