import pandas as pd
import re

def extract_subject(msg):
    """Extracts the subject line from raw email message."""
    match = re.search(r"Subject:\s*(.*)", msg)
    return match.group(1).strip() if match else "(No Subject)"

def load_email_data(path: str):
    """Load email data and extract subject and label."""
    df = pd.read_csv(path)
    df['subject'] = df['message'].apply(extract_subject)
    df['label'] = "unknown"  # No label in dataset, placeholder
    return df[['subject', 'message', 'label']]
