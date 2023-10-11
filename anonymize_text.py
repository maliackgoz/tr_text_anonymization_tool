import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

name_checkpoint = "deprem-ml/name_anonymization"

ner_checkpoint = "akdeniz27/bert-base-turkish-cased-ner"

phone_regex = r"(?:\(\+90\)\s?)?(?:\d{11}|\d{3}\s?\d{3}\s?\d{4})"
email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
date_regex = (r'\b((\d{1,2}[./]\d{1,2}[./]\d{4})|((?P<LongDate>(?P<Day>\d+)[\s ]+('
              r'?P<Month>Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)[\s ]+('
              r'?P<Year>\d{4}))))\b')
turkish_id_regex = r'\b[1-9][0-9]{9}[02468]\b'

# Specify the file path
file_path = 'test_text.txt'

# Open the file in read mode
with open(file_path, 'r', encoding='utf-8') as file:
    # Read the file contents into a string
    test_text = file.read()


def mask_phone_numbers(input_text):
    # Define a regex pattern to find phone numbers in the input text
    regex_phone = re.compile(phone_regex)

    # Find phone numbers in the input text
    matches = re.finditer(regex_phone, input_text)

    # Replace phone numbers with masked versions
    masked_text = input_text
    for match in matches:
        start, end = match.span()
        phone_number = match.group(0)
        masked_number = 'X' * (len(phone_number) - 4) + phone_number[-4:]
        masked_text = masked_text[:start] + masked_number + masked_text[end:]

    return masked_text


def extract_turkish_id(text):
    # Define the regex pattern
    pattern = turkish_id_regex

    # Use re.findall to extract all matching Turkish IDs from the text
    turkish_ids = re.findall(pattern, text)

    return turkish_ids


def mask_emails(input_text):
    # Find email addresses in the input text
    matches = re.finditer(email_regex, input_text)

    # Replace email addresses with masked versions
    masked_text = input_text
    for match in matches:
        start, end = match.span()
        email = match.group(0)
        parts = email.split('@')
        username = parts[0]
        domain = parts[1]
        masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
        masked_email = masked_username + '@' + domain
        masked_text = masked_text[:start] + masked_email + masked_text[end:]

    return masked_text


def mask_dates(input_text):
    def replace_date(match):
        return "(İlgili Tarih)"

    # Replace dates with "ilgili tarih"
    masked_text = re.sub(date_regex, replace_date, input_text)

    return masked_text


def regex_mask(input_text):
    try:
        masked_text = mask_phone_numbers(input_text)
    except Exception as e:
        print(f"Error masking phone numbers: {e}")
        masked_text = input_text

    try:
        masked_text = mask_emails(masked_text)
    except Exception as e:
        print(f"Error masking email addresses: {e}")

    try:
        masked_text = mask_dates(masked_text)
    except Exception as e:
        print(f"Error masking dates: {e}")

    return masked_text


def ner_mask(text, checkpoint):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")

    ner_results = ner(text)

    # Define a mapping of entity types to replacement labels
    replacement_labels = {
        # "PER": "(İNSAN)",
        "ORG": "(KURULUŞ)",
        # "LOC": "(YER)",
    }

    # Replace named entity with their corresponding labels
    cleaned_text = text  # Initialize with the original text
    for entity in ner_results:
        entity_label = entity["entity_group"]
        if entity_label in replacement_labels:
            cleaned_text = cleaned_text.replace(entity["word"], replacement_labels[entity_label])

    return cleaned_text


def ner_extract(text, checkpoint):
    # Initialize the NER pipeline
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")

    # Perform NER on the input text
    ner_results = ner(text)

    # Define a list to store extracted entities
    extracted_entities = []

    # Iterate through the NER results and extract entities
    for entity in ner_results:
        entity_text = entity["word"]
        entity_label = entity["entity_group"]
        extracted_entities.append((entity_text, entity_label))

    return extracted_entities


def remove_apostrophes(text):
    cleaned_text = text.replace("'", "")
    return cleaned_text


def ner_name_mask(text, checkpoint):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")

    text_test = remove_apostrophes(text)

    ner_results = ner(text_test)

    # Define a mapping of entity types to replacement labels
    replacement_labels = {
        "ad": "İNSAN",
    }

    # Initialize cleaned text with the original text
    cleaned_text = text

    # Create a list to store the last names for later replacement
    last_names = []

    # Process NER results
    for entity in ner_results:
        entity_label = entity["entity_group"]
        if entity_label in replacement_labels:
            if "İNSAN" in replacement_labels[entity_label]:  # Check if it's a person's name
                name_parts = entity["word"].split()  # Split the name into parts
                if len(name_parts) == 2:  # Check if it's a full name (first name and last name)
                    first_name, last_name = name_parts
                    # Keep the first 2 letters of the last name and mask the rest with asterisks
                    masked_last_name = last_name[:2] + '*' * (len(last_name) - 2)
                    last_names.append(masked_last_name)  # Add masked last name to the list
                    cleaned_text = cleaned_text.replace(entity["word"], first_name + " " + masked_last_name)

    return cleaned_text


def ner_name_extract(text, checkpoint):
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="first")

    text_test = remove_apostrophes(text)

    ner_results = ner(text_test)

    # Define a list to store the extracted names
    extracted_names = []

    # Process NER results
    for entity in ner_results:
        entity_label = entity["entity_group"]
        if entity_label == "İNSAN":  # Check if it's a person's name
            name_parts = entity["word"].split()  # Split the name into parts
            if len(name_parts) == 2:  # Check if it's a full name (first name and last name)
                first_name, last_name = name_parts
                extracted_names.append((first_name, last_name))  # Add name to the list as a tuple

    return extracted_names


def anonymize_text(input_text, name_checkpoint, ner_checkpoint):
    cleaned_text = regex_mask(input_text)
    cleaned_text = ner_mask(cleaned_text, ner_checkpoint)
    cleaned_text = ner_name_mask(cleaned_text, name_checkpoint)

    return cleaned_text


anonymized_text = anonymize_text(test_text, name_checkpoint, ner_checkpoint)

print("Original Text:" + test_text, end="\n\n")
print("Anonymized Text:" + anonymized_text)
