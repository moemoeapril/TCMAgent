
""""""
"""该脚本对单个文本文件进行向量化处理，并存储为FAISS索引，然后进行rerank。"""
""""""
import os
import getpass
from BCEmbedding import EmbeddingModel,RerankerModel

import chardet
import torch
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

class TextLoader:
    def __init__(self,file_path,save_folder=None):
        self.file_path=file_path

        # self.save_folder=save_folder if save_folder is not None else "./cleaned_text"
        self.save_folder=save_folder or "./cleaned_text"  # 默认保存路径
        os.makedirs(self.save_folder, exist_ok=True)
        self.file_name=os.path.splitext(os.path.basename(file_path))[0]
        self.raw_lines=[]
        self.cleaned_lines=[]
        
    def read_file(self):
        encodings=['utf-8', 'gbk', 'latin1', 'gb18030', 'utf-16']   
        
        for encoding in encodings:
            try:
                with open(self.file_path, 'r', encoding=encoding) as f:
                    self.raw_lines=f.read().splitlines()
                    print(f"[✅Success] 使用编码 '{encoding}' 成功读取文件: {self.file_path}")
                    return 
            except UnicodeDecodeError:
                print(f"[❌Fail] 编码 '{encoding}' 无法读取文件: {self.file_path}")
                continue
            
        # 如果预定义编码都失败，则使用 chardet 自动检测编码
        print("[🔍Detecting] 开始自动检测文件编码...")

        try:
            with open(self.file_path, 'rb') as f:
                raw_data = f.read()
            result=chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']

            if detected_encoding and confidence > 0.5:
                with open(self.file_path, 'r', encoding=detected_encoding) as f:
                    self.raw_lines=f.read().splitlines()
                    print(f"[✅Success] 使用编码 '{encoding}' 成功读取文件: {detected_encoding},置信度{confidence:.2f}")
                    return
            else:
                print(f"[❌Fail] 检测编码失败，无法读取文件: {self.file_path}")

        except Exception as e:
            print(f"[❌Error] 自动检测编码时发生异常: {e}")

        raise ValueError("无法读取文件！")

    def clean_text(self):
        self.cleaned_lines = []

        for line in self.raw_lines:
            line=line.strip()
            if line:
                cleaned=line.replace('\u3000', ' ')
                self.cleaned_lines.append(cleaned)

        if not self.cleaned_lines:
            raise ValueError("清洗后的文本为空！")
        
        #保存清洗后的文本
        cleaned_file_path=os.path.join(self.save_folder,f"{self.file_name}.txt")
        with open(cleaned_file_path, 'w', encoding='utf-8') as f:
            for line in self.cleaned_lines:
                f.write(line + '\n')

        print(f"[💾] 清洗后的文本已保存至: {cleaned_file_path}")
        
    def get_cleaned_lines(self):
        return self.cleaned_lines
    
class Embedder:
    """
    检测是否有GPU，然后对文本进行向量嵌入
    """
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[🔋] 使用设备: {self.device}")
        self.model = EmbeddingModel(model_name_or_path=model_path, device=self.device)

    def encode(self, texts):
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts=[] #文本存储

    def add_embeddings(self, embeddings,texts):
        self.index.add(embeddings)
        self.texts.extend(texts) #保存每个向量对应的文本

    def search(self, query_embedding, top_k=5): #topk默认为5，选择前5个最相关的条目
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k) #构建faiss索引
        
        results=[self.texts[i] for i in indices[0]] #获取最相关的条目
        return results
    
    def save(self, save_path):
        """
        保存faiss索引至本地"""
        save_dir=os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
        # print(f"[💾] 正在保存FAISS索引至: {save_path}")
        faiss.write_index(self.index, save_path)
        print(f"✅ FAISS索引已保存至: {save_path}")

    def load(self, load_path):
        """
        从本地加载faiss索引
        """
        self.index = faiss.read_index(load_path)


class App:
    def __init__(self, txt_path, model_path, faiss_save_dir,embedding_dir,reranker_path=None,top_k=5):
        self.txt_path = txt_path
        self.model_path = model_path
        self.faiss_save_path = faiss_save_dir
        self.embedding_save_path=embedding_dir
        self.save_folder=reranker_path
        self.reranker_path=reranker_path
        self.top_k = top_k

        #提取文件名作为目录保存的key
        file_system= os.path.splitext(os.path.basename(txt_path))[0]
        self.embedding_save_path = os.path.join(self.embedding_save_path, f"{file_system}_embeddings.npy")
        self.faiss_save_path = os.path.join(self.faiss_save_path, f"{file_system}_faiss.index")
        self.cleaned_save_folder = "/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/cleaned"

    def run(self):
        # 读取并清洗文本
        loader = TextLoader(self.txt_path,save_folder="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/cleaned")
        loader.read_file()
        loader.clean_text()
        docs = loader.get_cleaned_lines()

        # #打印文档前5行示例
        # for idx, line in enumerate(docs[:5]):
        #     print(f"第{idx}行：{line}")
        #     print("=" * 30)

        #查询本地是否已有Embedding
        # 如有直  接访问，没有则向量化并保存
        print(f"[🔍] 正在检查本地是否已有Embedding文件: {self.embedding_save_path}")
        print("=================================")
        if os.path.exists(self.embedding_save_path):
            print(f"[📂] 已发现本地 embedding 文件: {self.embedding_save_path}，直接加载...")
            embeddings=np.load(self.embedding_save_path)
 
        else:
        # 向量化
        
            embedder = Embedder(model_path=self.model_path)
            embeddings = embedder.encode(docs)
            print(f"[✅] Embedding向量 shape: {embeddings.shape}已保存至{self.embedding_save_path}")

            #确保目录存在
            embedding_dir=os.path.dirname(self.embedding_save_path)
            os.makedirs(embedding_dir, exist_ok=True)
            np.save(self.embedding_save_path, embeddings)  # 保存向量到本地
            print(f"[💾]向量已保存至{self.embedding_save_path}")

        # 构建向量索引
        store = VectorStore(embedding_dim=embeddings.shape[1])
        store.add_embeddings(embeddings,docs)

        # index_file=os.path.join(self.save_folder,"test_faiss.index")
        # print(f"[🔍] 正在搜索相似数据，查询内容: '天麻的功效是什么？'")
        # store.save(index_file)
        # 查询相似数据

        #查询示例并执行初步搜索
        embedder=Embedder(model_path=self.model_path)
        query = "天麻的功效是什么？"
        query_embedding = embedder.encode(query)
        # sim_indices = store.search(query_embedding, top_k=5)
        initial_results=store.search(query_embedding, top_k=self.top_k)

        print("\\n[🔍] 初步相似结果：")
        for idx,text in enumerate(initial_results):
            print(f"TOP{idx+1}:{text}")

        # 保存索引
        # index_file = os.path.join(self.faiss_save_path, "test_faiss.index")
        # store.save(index_file)
        # print(f"[💾] FAISS 索引已保存至: {index_file}")

        #执行reranke对结果进行重排
        if self.reranker_path:
            reranker_model=RerankerModel(model_name_or_path=self.reranker_path)
            rerank_results = reranker_model.rerank(query, initial_results)

            print("\n[🔄] Reranker重排结果：")
            for rank,(text,score,idx) in enumerate(
                zip(rerank_results['rerank_passages'],
                    rerank_results['rerank_scores'],
                    rerank_results['rerank_ids']
                ),start=1
            ):
                print(f"TOP{rank}:(score:{score:.4f},原索引={idx}\n {text}\n)")
        else:
            print("[⚠️] 未提供Reranker模型路径，跳过重排步骤。")

        #保存索引
        os.makedirs(os.path.dirname(self.faiss_save_path),exist_ok=True)
        store.save(self.faiss_save_path)
        print(f"[💾] FAISS索引已保存至: {self.faiss_save_path}")


# #计算句子对的语义相关score，对候选检索结果进行排序
# query=getpass("请输入查询内容：")
# passages=TextLoader.clean_text()
# sentence_pair=[[query,passage] for passage in passages]

# #初始化reranker模型
# model=RerankerModel(model_name_or_path="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/models/bce-reranker-base_v1")

#计算socres of sentence_pairs  
              

if __name__ == "__main__":
    txt_path = "/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/test/000-神农本草经-清-孙星衍.txt"
    model_path = "/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/models/bce-embedding-base_v1"
    faiss_save_path = "/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_index/faiss/"
    reranker_path="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/models/bce-reranker-base_v1"

    embedding_save_path = "/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_index/emb/"

   

    app = App(
        txt_path=txt_path,
        model_path=model_path,
        faiss_save_dir=faiss_save_path,
        embedding_dir=embedding_save_path,
        reranker_path=reranker_path,
        top_k=5)
    app.run()
     

# """textLoader 散装====begin"""
# def read_txt(file_path):
#     """读取多种编码的文本文件"""
#     # 可用编码列表
#     encodings = ['utf-8', 'gbk', 'latin1', 'gb18030', 'utf-16']

#     for encoding in encodings:
#         try:
#             with open(file_path, 'r', encoding=encoding) as f:
#                 content = f.read().splitlines()
#                 print(f"[✅Success] 使用编码 '{encoding}' 成功读取文件: {file_path}")
#                 return content
#         except UnicodeDecodeError:
#             print(f"[❌Fail] 编码 '{encoding}' 无法读取文件: {file_path}")
#             continue

#     # 如果预定义编码都失败，则使用 chardet 自动检测编码
#     print("[🔍Detecting] 开始自动检测文件编码...")

#     try:
#         with open(file_path, 'rb') as f:
#             raw_data = f.read()
#         result = chardet.detect(raw_data)
#         detected_encoding = result['encoding']
#         confidence = result['confidence']

#         if detected_encoding and confidence > 0.5:
#             with open(file_path, 'r', encoding=detected_encoding) as f:
#                 content = f.read().splitlines()
#                 print(f"[✅Success] 使用自动检测编码 '{detected_encoding}' (置信度: {confidence:.2f}) 读取文件")
#                 return content
#         else:
#             print(f"[⚠️Warning] 自动检测编码失败，置信度不足: {confidence:.2f}, 检测到编码: {detected_encoding}")
#     except Exception as e:
#         print(f"[❌Error] 自动检测编码时发生异常: {e}")

#     print("[🚨Final] 无法读取该文件，请检查文件编码或完整性。")
#     return []


# txt_path=r"/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_raw/test/000-神农本草经-清-孙星衍.txt"
# save_faiss_index_path="/home/ubuntu/moe/PaddleOCR_m/TCMAgent/data/knowledge_index/"
# doc=read_txt(txt_path)

# for idx,line in enumerate(doc[:5]):
#     print(f"第{idx}行：{line}")
#     print("===================================")

# # 数据清洗 后期封装为一个函数
# cleaned_doc = []
# for line in doc:
#     stripped_line = line.strip() #去除空白字符
#     if stripped_line:
#         cleaned_line=stripped_line.replace("\u3000",' ')#将全角空格替换为普通空格
#         cleaned_doc.append(cleaned_line)

# # 打印清洗后的内容
# for i,line in enumerate(cleaned_doc[:5]):
#     print("===================================")
#     print(f"第{i}行：{line}")

# # check if GPU is available
# device="cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# model=EmbeddingModel(model_name_or_path=r"./data/models/bce-embedding-base_v1", device=device)
# print(f"Using device: {device}")
# embeddings=model.encode(cleaned_doc)
# print(embeddings.shape)

# #用faiss存储库将向量进行本地化
# #构建索引
# def create_faiss_index(embeddings):
#     if len(cleaned_doc)==0:
#         print("清洗后的文档为空，终止流程")
#         exit()

#     #采用暴力检索FlatL2
#     index=faiss.IndexFlatL2(embeddings.shape[1]) #创建空索引
#     index.add(embeddings)

#     return index

# #数据检索
# def data_recall(faiss_index,query, top_k):
#     query_embedding=model.encode(query)
#     if len(query_embedding)==1:
#         query_embedding
#         Distance, Index=faiss_index.search(query_embedding,top_k)
#     return Index



# #索引的保存
# def faiss_index_save(faiss_index,save_path):
#     faiss.write_index(faiss_index,save_path)

# # 索引的加载
# def index_data_load(faiss_index_save_file_location):
#     index=faiss.read_index(faiss_index_save_file_location)
#     return index

# query="天麻的功效是什么？"
# faiss_embeddings=create_faiss_index(embeddings)
# sim_data_index=data_recall(faiss_embeddings,query,top_k=5)
# print(f"相似的数据为：")
# for index in sim_data_index[0]:
#     print(cleaned_doc[int(index)] + "\n")

# #保存至指定路径
# faiss_index_save(faiss_embeddings,os.path.join(save_faiss_index_path,"test_faiss.index"))
# print(f"✅ FAISS 索引已保存至 {save_faiss_index_path}")

# # # 大模型助手构建
# # from zhipuai import ZhipuAI
# # client=ZhipuAI(api_key=os.environ["GLM_API_KEY"])
# # sql_prompt = f"""
# # 你是一个中医药处方助手，这是从向量数据库中检索到的内容，请进行
