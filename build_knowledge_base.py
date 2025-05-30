"""根据source构建知识库"""
# 遍历指定目录下的源文件
# 对文本进行分块（简单规则
# 用bce embedding进行向量化
# 将向量和文本保存到faiss索引
# 保存索引和元数据用于后续rag

import os
import fitz
import faiss
import json
from tqdm import tqdm


from BCEmbedding import EmbeddingModel

# doc_folder="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/book"
doc_folder="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/test"
index_save_path="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_index"
meta_save_path="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_index/meta.json"
chunk_size=200
chunk_overlap=50

# os.makedirs(index_save_path, exist_ok=True)
# os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)

model=EmbeddingModel(model_name_or_path="maidalun1020/bce-embedding-base_v1")

def read_txt(file_path):
    """读取文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
    
def read_pdf(file_path):
        """读取PDF文件"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

def chunk_text(text,chunksize=chunk_size,chunkoverlap=chunk_overlap):
    chunks=[]
    start=0
    while start < len(text):
        end=min(start+chunksize,len(text))
        chunk=text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start+=chunksize-chunkoverlap
    return chunks

def build_knowledge_base():
    """构建知识库"""
    all_chunks=[]
    metadata=[]

    print("🔍 正在加载文档并分块...")
    for filename in os.listdir(doc_folder):
         file_path=os.path.join(doc_folder, filename)
         if filename.endswith('.txt'):
             text=read_txt(file_path)
         elif filename.endswith('.pdf'):
             text=read_pdf(file_path)
         else:
             continue
         
         chunks =chunk_text(text)
         for idx, chunk in enumerate(chunks):
             all_chunks.append(chunk)
             metadata.append({
                 "source": filename, 
                 "chunk_id": idx,
                 "text":chunk
                 })
    print(f"✅ 共分块 {len(all_chunks)} 段，开始生成向量...")

    embeddings=model.encode(all_chunks,batch_size=16)

    print("✅ 正在构建 faiss 索引...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(index_save_path), exist_ok=True)
    faiss.write_index(index, index_save_path)

    with open(meta_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ 索引已保存至 {index_save_path}")
    print(f"✅ 元数据已保存至 {meta_save_path}")