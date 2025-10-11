#!/bin/bash
# Start the LLM Assistant application

echo "Starting LLM Assistant..."
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
echo "Installing/updating dependencies..."
pip install -r requirements.txt
echo

# Start the application
echo "Starting FastAPI server..."
python -m app.main
