## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   uv venv # recommended
   
   .venv\Scripts\activate  # On Windows
   
   source .venv/bin/activate  # For Mac/Linux:
   ```
3. Install dependencies (create a requirements.txt file with the following):
   ```
   python -m pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

## Usage

Run the application with:
```
python main.py
```
