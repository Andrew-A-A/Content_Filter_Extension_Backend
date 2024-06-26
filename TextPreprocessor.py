import pandas as pd
import re
from transformers import DistilBertTokenizer
from string import digits
import unicodedata

def pipelineText(text):
    slang= pd.read_csv('data/slang.csv',index_col=False)
    slang.set_index('acronym',drop=True,inplace=True)
    slang.drop('Unnamed: 0',axis=1, inplace=True)
    slang=slang.to_dict()
    inner_dict=slang['expansion']
    slang = {acronym: meaning for acronym, meaning in inner_dict.items()}
    text= remove_url(text)
    text= remove_days_months(text)
    text= remove_unicode_variations(text)
    text= remove_mentions(text)
    text= remove_hashtags(text)
    text= remove_special_characters(text)
    text= remove_redundant_characters_in_row(text)
    text= replace_acronyms_with_meanings(text,slang)
    text= remove_numbers(text)
    text= text.lower()
    return text
def replace_acronyms_with_meanings(text, acronym_dict):
    def replace_acronym(match):
        acronym = match.group(0)
        meaning = acronym_dict.get(acronym.lower(), acronym)
        return meaning

    pattern = r'\b[A-Z]{2,}\b'
    if isinstance(text, str):
        updated_text = re.sub(pattern, replace_acronym, text, flags=re.IGNORECASE)
        return updated_text
def remove_redundant_characters_in_row(row):
    if isinstance(row, str):
        words = row.split()
        cleaned_words = []
        for word in words:
            cleaned_word = ""
            prev_char = None
            count = 0

            for char in word:
                if char == prev_char:
                    count += 1
                    if count <= 2:
                        cleaned_word += char
                else:
                    cleaned_word += char
                    count = 1
                prev_char = char

            cleaned_words.append(cleaned_word)

        return " ".join(cleaned_words)
def remove_unicode_variations(input_string):
    if isinstance(input_string, str):
        normalized_string = unicodedata.normalize('NFKD', input_string)
        ascii_string = normalized_string.encode('ascii', 'ignore').decode('utf-8')
        return ascii_string
def remove_numbers(text):
    if isinstance(text, str):
        rem = str.maketrans('', '', digits)
        res = text.translate(rem)
        return res
def remove_special_characters(text):
    if isinstance(text, str):
        special_characters = r'[!@#$%*&()-+\n,.?/:;{}\'"><=`-]'
        pattern = re.compile(special_characters)
        return pattern.sub('', text)
def remove_url(text):
    if isinstance(text, str):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        text=url_pattern.sub('', text)
    return text
def remove_days_months(text):
    if isinstance(text, str):
        day_names_pattern = r'\b(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\b'
        day_names = r'\b(?:mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b'
        month_names_pattern = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'
        month_names = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
        combined_pattern = f'{day_names_pattern}|{month_names_pattern}|{day_names}|{month_names}'
        result = re.sub(combined_pattern, '', text)
        return result
def remove_hashtags(text):
    if isinstance(text, str):
        forms = [r'#\w+',  # form1(# followed by letters whether small or capital and/or numbers)
                 r'#([A-Z][a-z]+)([A-Z][a-z]+)'
                 # form2(# followed by capital letter,set of small letters , another capital letters and set of small letters -> part the 2 words)
                 ]
        all_forms = '|'.join(forms)
        pattern = re.compile(all_forms)
        return pattern.sub('', text)
    return text
def remove_mentions(text):

    if isinstance(text, str):
        forms = [r'@[A-Za-z0-9]+',  # form1
                 r'@[A-Za-z0-9]+/[A-Za-z0-9]+',  # form2
                 r'@[A-Za-z0-9]+[^\w\s]',  # form3
                 r'@[A-Za-z0-9]+:\s?'  # form4
                 ]
        all_forms = '|'.join(forms)
        pattern = re.compile(all_forms)
        text = pattern.sub('', text)
    return text
def preprocess_text_list(text_list):
    # Initialize the DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_texts = tokenizer(text_list, padding=True, truncation=True, return_tensors="tf")
    padding_mask = tokenized_texts["attention_mask"]
    token_ids = tokenized_texts["input_ids"]
    return padding_mask, token_ids
def filter_strings(string_list):
    if not string_list:
        return []
    filtered_list = [s for s in string_list if len(s) > 1]
    return filtered_list