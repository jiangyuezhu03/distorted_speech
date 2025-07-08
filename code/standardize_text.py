import re
def standardize_reference_text(text):
    return re.sub(" '","'",text)

def clean_punctuations_transcript(text):
    text= text.strip().lower()
    text = re.sub(r' --+', '', text)           # remove 2 or more dashes, only owsm needs it
    text = re.sub(r'(?<!-)-(?!-)', ' ', text) # replace single dash with space, both models need it
    text = re.sub(r"[,.?!:\"]", '', text)       # remove other common punctuation
    text = re.sub(r" \(laughter", "", text) # only owsm problem
    return text

def clean_punctuations_transcript_owsm(text):
    text= text.strip().lower()
    text = re.sub(r' --+', '', text)           # remove 2 or more dashes, only owsm needs it
    text = re.sub(r'(?<!-)-(?!-)', ' ', text) # replace single dash with space, both models need it
    text = re.sub(r"[,.?!:\"]", '', text)       # remove other common punctuation, both models need it
    text = re.sub(r" \(laughter", "", text) # removes laughter label, only owsm problem
    return text
def clean_punctuations_transcript_whspr(text):
    text= text.strip().lower()
    text = re.sub(r'(?<!-)-(?!-)', ' ', text) # replace single dash with space
    text = re.sub(r"[,.?!:\"]", '', text)       # remove other common punctuation
    return text