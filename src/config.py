import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from a .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file or environment variables.")
