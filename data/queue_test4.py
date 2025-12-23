import queue
# from simhash import Simhash
from simcse import SimCSE
import torch
from transformers import AutoModel, AutoTokenizer
import time
from scipy.spatial.distance import cosine
# def cal_simhash(text):
#     hash = Simhash(text.strip('\n'))
#     return hash




def sim_1vsN(embedding, multi_embedding):
    max = 0
    for e in multi_embedding:
        similarity = torch.cosine_similarity(embedding, e, dim=-1)
        if similarity > max:
            max = similarity
    return max


def set_queue(q, datas, data_num):
    for i in datas:
        q.put(i)


def main():
    # model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    # tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    USE_CUDA = torch.cuda.is_available()

    model = model.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
    batch_size = 144
    out = open("wiki1m_add_sup_roberta144.txt", 'w', newline='', encoding='utf-8')
    with open("wiki1m_for_simcse.txt", "r", encoding='utf-8') as f:
        datas = f.readlines()
    embeddings = []
    for data in datas:
        input = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        input = input.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
        embedding = model(**input,output_hidden_states=True, return_dict=True).pooler_output
        embeddings.append(embedding[0].tolist())
        # time.sleep(0.001)
    print("embedding计算完毕")
    # embeddings = torch.tensor(embeddings)

    assert len(embeddings) == len(datas)
    # embeddings = embeddings.to(device=torch.device('cuda' if USE_CUDA else 'cpu'))
    data_num = len(datas)
    # 空队列
    q_sent = queue.Queue(data_num)
    q_hash = queue.Queue(data_num)
    # 给队列赋值
    set_queue(q_sent, datas, data_num)
    set_queue(q_hash, embeddings, data_num)
    # 取句子，用simhash判断相似性，
    sentence_first = q_sent.get()
    hash_first = q_hash.get()
    # 给队列添加一个元素
    sentences = [sentence_first]
    hashs = [hash_first]
    while True:
        h = q_hash.get()
        s = q_sent.get()

        sim = sim_1vsN(h, hashs)

        # 若相似，插入队列尾部
        if sim > 0.7:
            q_hash.put(h)
            q_sent.put(s)
        # 若不相似，写入新的dataset中
        if sim <= 0.7:
            hashs.append(h)
            sentences.append(s)
        # 若已经够了一个batch，则取出保存，并使得列表再次为空
        if len(sentences) % (batch_size) == 0:
            out.writelines(sentences)
            q_size = q_sent.qsize()
            print(q_size)
            sentences = [q_sent.get()]
            hashs = [q_hash.get()]

            # 循环停止策略
            #if q_size <= 247000:
            #    out.write(sentences[0])
            #    while True:
            #        out.write(q_sent.get())
            #        if q_sent.empty():
            #            print("队列空了")
            #            return


if __name__ == '__main__':
    main()
