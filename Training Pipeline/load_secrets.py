"""
Helper script to load HuggingFace token from environment or .env file
"""
import os
from pathlib import Path

def get_hf_token():
    """
    Get HuggingFace token from environment variable or .env file.

    Priority:
    1. Environment variable HF_TOKEN
    2. Google Colab secrets (if in Colab)
    3. .env file in project root

    Returns:
        str: HuggingFace token or None if not found
    """
    # First check environment variable
    token = os.environ.get('HF_TOKEN')
    if token:
        return token

    # Check if we're in Google Colab
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token
    except:
        pass

    # Try loading from .env file
    try:
        env_path = Path(__file__).parent.parent / '.env'
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('HF_TOKEN='):
                        return line.split('=', 1)[1].strip()
    except:
        pass

    # If still not found, provide helpful message
    print("WARNING: HF_TOKEN not found!")
    print("Please set it using one of these methods:")
    print("1. Environment variable: export HF_TOKEN='your_token_here'")
    print("2. Google Colab: Add HF_TOKEN to Colab secrets")
    print("3. Create .env file in project root with: HF_TOKEN=your_token_here")
    return None

# For backward compatibility
hf_token = get_hf_token()
