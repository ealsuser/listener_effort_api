import re
import numpy as np
from lepm_api.utils import get_logger
logger = get_logger()

def replace_contractions(text):
    """
    Replace contractions and long forms in a text with their normalized format without apostrophes.

    This function replaces:
    - Short contractions with apostrophes (e.g., "couldn't") to their target format without apostrophes (e.g., "couldnt").
    - Long formats (e.g., "could not") to the same target format (e.g., "couldnt").
    - Handles case-insensitive matches.

    Parameters:
    text (str): The input text to normalize.

    Returns:
    str: The text with contractions replaced by their normalized format.

    Examples:
    >>> replace_contractions("Could not you see? I couldn't tell if you'll be there.")
    'Couldnt you see? I couldnt tell if youll be there.'
    >>> replace_contractions("They are not sure. Aren't they?")
    'They arent sure. Arent they?'
    """
    contractions = {
        "could not": "couldnt",
        "couldn't": "couldnt",
        "did not": "didnt",
        "didn't": "didnt",
        "you will": "youll",
        "you'll": "youll",
        "are not": "arent",
        "aren't": "arent"
    }
    
    # Use regex to handle case-insensitivity and word boundaries
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda match: contractions[match.group(0).lower()], text)

def clean_text(text):
    try:
        text = text.lower().strip()
        text = replace_contractions(text)
        text = text.replace(",", "").replace(".", "")
        for ch in ["!", "$", "&", "|", "ü", "ł", "土", "家", "石", "<", ">", "?"]:
            text = text.replace(ch, " ")
        return text
    except Exception as e:
        logger.info(f"Error in clean_text: {e}")
        return np.nan