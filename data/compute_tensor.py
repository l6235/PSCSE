import queue
# from simhash import Simhash
from simcse import SimCSE
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def main():
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    #USE_CUDA = torch.cuda.is_available()
    USE_CUDA = False

    model = model.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
    batch_size = 144
    # out = open("wiki1m_add_sup_roberta144.txt", 'w', newline='', encoding='utf-8')
    with open("wiki1m_for_simcse.txt", "r", encoding='utf-8') as f:
        datas = f.readlines()[:10]
    embeddings = []
    for data in tqdm(datas):
        input = tokenizer(data, padding=True, truncation=True, return_tensors="pt",max_length=32)
        input = input.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        embedding = model(**input,output_hidden_states=True, return_dict=True).pooler_output
        embeddings.append(embedding[0].tolist())
        # time.sleep(0.001)
    print("embedding计算完毕")
    embeddings = torch.tensor(embeddings)
    torch.save(embeddings,"embedding-file")
if __name__ == '__main__':
    main()