# Turkish Text Anonymization Tool

**Project Overview:** This project is a Turkish text anonymization tool that utilizes two different named entity recognition models and regular expressions to identify and anonymize sensitive information in text, such as phone numbers, email addresses, dates, Turkish identification numbers, and names.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Features

- Anonymizes sensitive information such as phone numbers, email addresses, dates, Turkish identification numbers, and names.
- Utilizes two different named entity recognition models for entity identification and replacement.
- Supports Turkish text.

## Dependencies

- Python 3.x
- [Transformers Library](https://github.com/huggingface/transformers) - Used for the named entity recognition models.
- You can install the required Python packages using the following command:
```
pip install transformers
```
## Usage

1. Clone the repository to your local machine:
```
git clone https://github.com/maliackgoz/tr_text_anonymization_tool.git
```
2. Navigate to the project directory:
```
cd tr_text_anonymization_tool
```
3. Place the text you want to anonymize in a file (e.g., `test_text.txt`) in the project directory.

4. Open the Python script (`anonymize_text.py`) and specify the paths to your NER models and the input file.

5. Run the script:
```
python anonymize_text.py
```

## Acknowledgments

- [Name Recognition Model](https://huggingface.co/deprem-ml/name_anonymization)
- [Named Entity Recognition Model](https://huggingface.co/akdeniz27/bert-base-turkish-cased-ner)
