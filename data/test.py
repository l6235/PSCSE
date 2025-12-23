from transformers import AutoTokenizer, AutoModel
import numpy

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fileObject = open('text.txt', 'w', encoding='UTF-8')

with open("wiki1m_for_simcse.txt",encoding="UTF-8") as f:
    lines = f.readlines()
    for line in lines:
        tokens = tokenizer.encode_plus(line)
        input_ids = tokens["input_ids"]
        input_ids = str(input_ids)
        # str = '/'.join(input_ids)
        fileObject.write(input_ids)
        fileObject.write('\n')

fileObject.close()