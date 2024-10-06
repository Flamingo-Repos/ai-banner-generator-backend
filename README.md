# Ad Image Generator

This project generates ad images based on user input using GPT-4 for prompt generation and Fal.AI for image creation.

## Setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r backend/requirements.txt`
5. Set up your `.env` file with your OpenAI and Fal.AI API keys
6. Run the application: `python src/main.py`

## Usage

Send a POST request to `http://localhost:8000/generate-ad` with the following JSON body:
