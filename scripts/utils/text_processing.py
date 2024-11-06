import re

# Text Processing and Filtering
def preprocess_text_pairs(data_pairs):
    """
    Preprocess the text data by cleaning up common errors and symbols.

    Args:
        data_pairs (list): A list of tuples containing source and target text pairs.

    Returns:
        list: A list of cleaned text pairs.
    """
    cleaned_data = []
    for src_text, tgt_text in data_pairs:
        src_text = fix_text_content(src_text)
        tgt_text = fix_text_content(tgt_text)
        cleaned_data.append((src_text, tgt_text))
    return cleaned_data


def fix_text_content(text):
    """
    Fix common contractions and HTML character entities in the text.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    replacements = {
        "won &apos;t": "will not",
        "can &apos;t": "cannot",
        "couldn &apos;t": "could not",
        "don &apos;t": "do not",
        "didn &apos;t": "did not",
        "&apos;d": "would",
        "&apos;s": "is",
        "&apos;re": "are",
        "&apos;m": "am",
        "&apos;ve": "have",
        "&apos;ll": "will",
        "&quot;": '"',
        "&amp;": "&",
    }

    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    text = pattern.sub(lambda match: replacements[match.group(0)], text)
    
    return text


def filter_long_sequences(data_pairs, tokenizer, max_length):
    """
    Filter out text pairs that exceed the maximum length allowed after tokenization.

    Args:
        data_pairs (list): A list of tuples containing source and target text pairs.
        tokenizer (PreTrainedTokenizer): The tokenizer for both source and target languages.
        max_length (int): The maximum allowed length for the tokenized sequences.

    Returns:
        list: A filtered list of text pairs that are within the max length.
    """
    filtered_data = []
    for src_text, tgt_text in data_pairs:
        src_token_ids = tokenizer.encode(src_text, add_special_tokens=True, truncation=False)
        tgt_token_ids = tokenizer.encode(tgt_text, add_special_tokens=True, truncation=False)
        if len(src_token_ids) <= max_length and len(tgt_token_ids) <= max_length:
            filtered_data.append((src_text, tgt_text))
    return filtered_data