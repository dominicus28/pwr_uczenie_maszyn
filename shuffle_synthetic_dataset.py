import random

input_file = "synthetic_spam_tmp.txt"
output_file = "synthetic_spam.txt"

with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

random.shuffle(lines)

with open(output_file, "w", encoding="utf-8") as file:
    file.writelines(lines)

