# 本脚本实现了从 SQLite 数据库中查询中药处方的功能
# 通过 GLM 模型生成 SQL 查询语句，并执行查询

# api_key:89df81a34eb1453b8b6936452851e1f9.YQbXHHYyipntG7E9
#通过命令行传入中药名称
# 将输出：GLM 自动生成的 SQL + 查询结果 + 自然语言回答。
# python query_herb_prescriptions.py herb_name 

import os
import sys
import sqlite3
import getpass
from zhipuai import ZhipuAI

def query_prescriptions_by_herb(herb_name: str, db_path: str = "./data/tcm_b.db") -> str:
    # 连接 SQLite 数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 设置 API Key（如果环境变量未设置）
    if not os.environ.get("GLM_API_KEY"):
        os.environ["GLM_API_KEY"] = getpass.getpass("Enter your GLM API key: ")

    # 初始化 GLM 客户端
    client = ZhipuAI(api_key=os.environ["GLM_API_KEY"])

    # -------- Step 1: 让 GLM 生成 SQL --------
    sql_prompt = f"""
你是一个中医药数据库助手，数据库中有一张名为 herbs 的表，字段包括：
- id
- prescriptions_id
- herb_name
- dose
- note

请生成一条 SQLite 查询语句，用于查询哪些处方中使用了“{herb_name}”。仅返回 SQL 本身。
    """

    sql_response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": "你是一个擅长 SQL 查询的助手。"},
            {"role": "user", "content": sql_prompt}
        ],
        stream=False
    )

    sql_query = sql_response.choices[0].message.content.strip().strip("```sql").strip("```")
    print(f"\n✅ 生成的 SQL：\n{sql_query}")

    # -------- Step 2: 执行 SQL 查询 --------
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        print("\n📊 查询结果：", rows)
    except Exception as e:
        conn.close()
        return f"❌ SQL 执行失败：{e}"

    # -------- Step 3: 让 GLM 生成自然语言回答 --------
    result_prompt = f"""
你是一个中医药专家。以下是查询“哪些处方中使用了 {herb_name}”的结果：
{rows}

请用简洁、专业的语言告诉用户结果。
    """

    result_response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {"role": "system", "content": "你是一个中医药问答专家。"},
            {"role": "user", "content": result_prompt}
        ],
        stream=False
    )

    conn.close()
    return result_response.choices[0].message.content.strip()


# ========= 命令行入口 =========
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python query_herb_prescriptions.py <中药名称>")
        sys.exit(1)

    herb = sys.argv[1]
    answer = query_prescriptions_by_herb(herb)
    print("\n💬 回答：", answer)
