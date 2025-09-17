# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an anime character dialogue extraction system that uses UVR5 for vocal separation and speaker identification to automatically extract and classify different character dialogues from anime videos based on subtitles.

## Common Commands

### Development
```bash
# Install all dependencies (backend + frontend)
yarn install-all

# Start both backend and frontend in development mode
yarn dev

# Start backend only (Python Flask server on port 5000)
yarn server

# Start frontend only (React app on port 3000)
yarn client

# Build frontend for production
yarn build
```

### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend (React + TypeScript)
```bash
cd frontend
yarn install
yarn start    # Development server
yarn build    # Production build
yarn test     # Run tests
```

## Architecture

### Backend (`backend/`)
- **Flask API server** (`app.py`) - Main web API with file upload, processing endpoints
- **Core processing modules** (`core/`):
  - `main_processor.py` - Orchestrates the entire pipeline
  - `subtitle_splitter.py` - Extracts audio and splits by subtitle timing
  - `uvr5_processor.py` - Handles UVR5 vocal separation integration
  - `speaker_diarization.py` - Speaker identification and clustering using ML
  - `dialogue_extractor.py` - Exports organized dialogue files
- **Configuration** (`config.py`) - System settings and file paths
- **System utilities** (`utils/system_check.py`) - Environment validation

### Frontend (`frontend/src/`)
- **React + TypeScript** with Ant Design UI components
- **Main components**:
  - `FileUpload.tsx` - Video and subtitle file upload interface
  - `ProcessingProgress.tsx` - Real-time processing status display
  - `ResultDisplay.tsx` - Results visualization and download
- **API communication** via Axios for backend integration

### Processing Pipeline
1. **Subtitle-based audio splitting** - Extracts audio track and splits by subtitle timings
2. **UVR5 vocal separation** - Isolates vocals from background music (optional)
3. **Speaker embedding extraction** - Extracts ML features for speaker identification
4. **Speaker clustering** - Groups audio segments by speaker using ML algorithms
5. **Dialogue export** - Organizes and exports speaker-separated audio/text files

### Key Dependencies
- **Backend**: Flask, MoviePy, librosa, scikit-learn, torch, pyannote.audio
- **Frontend**: React 18, TypeScript, Ant Design, Axios
- **External**: FFmpeg (required), UVR5 (optional for best vocal separation)

### File Structure
- Input: Video files (MP4, AVI, MKV, etc.) + Subtitle files (SRT, ASS, VTT, etc.)
- Output: `output/speaker_XX/` directories with separated audio files and transcripts
- Temp files automatically cleaned up after processing

### API Endpoints
- `POST /api/upload` - File upload
- `POST /api/process` - Start processing
- `GET /api/progress` - Get processing status
- `GET /api/result` - Get final results
- `GET /api/download/<path>` - Download output files
- `POST /api/reset` - Reset system state

## Development Notes

- The system requires FFmpeg to be installed system-wide for audio processing
- UVR5 integration is optional but recommended for better vocal separation quality
- Processing is handled asynchronously with progress callbacks for UI updates
- All file uploads are validated for type and size limits
- Temporary files are automatically cleaned up after processing