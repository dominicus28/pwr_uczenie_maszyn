import re
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def clean_messages(input_file, output_file):
    encoding = detect_encoding(input_file)
    print(f"Detected encoding: {encoding}")

    with open(input_file, 'r', encoding=encoding) as infile:
        lines = infile.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r'^\s*\d+\s*[-."]*\s*', '', line)
        cleaned_line = cleaned_line.replace('â€™', "'")
        cleaned_line = cleaned_line.strip('"').strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(line + '\n' for line in cleaned_lines)

# input_file = 'raw_synthetic_spam.txt'
input_file = 'synthetic_spam.txt'
output_file = 'synthetic_spam_tmp.txt'
clean_messages(input_file, output_file)
