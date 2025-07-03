import re
def standardize_reference_text(text):
    return re.sub(" '","'",text)
def clean_punctuations_transcript(text):
    text= text.strip().lower()
    text = re.sub(r' --+', '', text)           # remove 2 or more dashes
    text = re.sub(r'(?<!-)-(?!-)', ' ', text) # replace single dash with space
    text = re.sub(r"[,.?!:]", '', text)       # remove other common punctuation
    return text
