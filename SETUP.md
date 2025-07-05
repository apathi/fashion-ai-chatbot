# Quick Setup Guide

## Prerequisites Check

Before starting, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] OpenAI API key (from https://platform.openai.com/api-keys)
- [ ] Terminal/command line access

## Installation Steps

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd fashion-ai-chatbot
```

### 2. Create Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your favorite editor
# Add your OpenAI API key
nano .env
```

**Minimum required in .env:**
```env
OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Verify Setup
```bash
# Quick test to ensure everything works
python -c "from src.chatbot import create_chatbot; print('Setup successful!')"
```

## Running the App

### Start the Server
```bash
python app.py
```

### Access the Interface
Open your browser and go to:
```
http://localhost:5000
```

## Testing

### Run Tests
```bash
pytest tests/ -v
```

### Run Evaluation
```bash
python scripts/evaluate.py
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'openai'"
**Solution:** Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"
**Solution:** Check that .env file exists and contains your API key:
```bash
cat .env  # Should show OPENAI_API_KEY=sk-...
```

### Issue: "Port 5000 already in use"
**Solution:** This is common on macOS where AirPlay Receiver uses port 5000.

**Quick Fix - Use Port 8000 (Recommended):**
The .env file has been updated to use port 8000. Just run:
```bash
python app.py
# Then access http://localhost:8000
```

**Alternative - Disable AirPlay Receiver:**
```
System Preferences → General → AirDrop & Handoff →
Uncheck "AirPlay Receiver"
```

### Issue: Slow initial load
**Solution:** This is normal! The first run generates embeddings for 82 trends.
Subsequent runs will be much faster due to caching.

## Next Steps

1. ✅ Try the sample questions in the UI
2. ✅ Toggle between RAG and Basic modes
3. ✅ Use "Compare RAG vs Basic" to see the difference
4. ✅ Check the evaluation results with `python scripts/evaluate.py`
5. ✅ Read the full README.md for detailed documentation

## Need Help?

- Check the [README.md](README.md) for comprehensive documentation
- Review the [API Documentation](README.md#api-documentation) section
- Look at sample code in `tests/test_chatbot.py`
- Open an issue on GitHub

Happy coding! 🎨
