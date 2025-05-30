import sqlite3
import csv
import re
import os

# 连接数据库
db_path = os.path.join(os.path.dirname(__file__), "tcm_b.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

"""--------------建表------------------"""
# 病人表
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE,
    gender INTEGER
)
""")

# 处方主表
cursor.execute("""
CREATE TABLE IF NOT EXISTS prescriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    diagonosis TEXT,
    note TEXT,
    quantity INTEGER,
    date TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(id)
)
""")

# 药材子表
cursor.execute("""
CREATE TABLE IF NOT EXISTS herbs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prescriptions_id INTEGER,
    herb_name TEXT,
    dose INTEGER,
    note TEXT,
    FOREIGN KEY(prescriptions_id) REFERENCES prescriptions(id)
)
""")

"""--------------批量导入数据------------------"""
prep_dir = os.path.join(os.path.dirname(__file__), "prep")

for patient_folder in os.listdir(prep_dir):
    patient_path = os.path.join(prep_dir, patient_folder)
    
    # 只处理目录，并且目录名是纯数字（病人ID）
    if os.path.isdir(patient_path) and patient_folder.isdigit():
        patient_code = patient_folder
        print(f"\n🚑 正在处理病人 {patient_code}...")

        # 插入病人信息（如果已经存在就忽略）
        cursor.execute("INSERT OR IGNORE INTO patients (code) VALUES (?)", (patient_code,))
        cursor.execute("SELECT id FROM patients WHERE code=?", (patient_code,))
        patient_id = cursor.fetchone()[0]

        # 遍历该病人文件夹下所有CSV文件
        for csv_file in os.listdir(patient_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(patient_path, csv_file)
                print(f"📄 正在处理文件：{file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                if len(rows) < 4:
                    print(f"⚠️ 文件内容不足，跳过：{file_path}")
                    continue

                # 读取CSV基础信息
                date_value = rows[0][1].strip() if len(rows[0]) > 1 else ''
                patient_code_in_file = rows[1][1].strip() if len(rows[1]) > 1 else ''
                gender_value = int(rows[2][1].strip()) if len(rows[2]) > 1 and rows[2][1].isdigit() else None
                diagonosis = rows[3][1].strip() if len(rows[3]) > 1 else ''

                # 校验病人编号
                if patient_code_in_file != patient_code:
                    print(f"❗警告：病人ID不一致！目录是 {patient_code}，文件中是 {patient_code_in_file}")

                # 更新性别信息（如果有）
                if gender_value is not None:
                    cursor.execute("UPDATE patients SET gender=? WHERE id=?", (gender_value, patient_id))

                # 插入处方主表
                cursor.execute("""
                    INSERT INTO prescriptions (patient_id, diagonosis, quantity, date, note)
                    VALUES (?, ?, ?, ?, ?)
                """, (patient_id, diagonosis, 1, date_value, ""))

                prescription_id = cursor.lastrowid

                # 插入药材子表
                herb_count = 0
                for row in rows[4:]:
                    if len(row) >= 2:
                        herb_name = row[0].strip().strip("，, ")
                        dose_str = row[1].strip().strip("，, ")
                        note = row[2].strip().strip("，, ") if len(row) >= 3 else ""

                        if re.match(r'^\d+$', dose_str):
                            dose = int(dose_str)
                            cursor.execute("""
                                INSERT INTO herbs (prescriptions_id, herb_name, dose, note)
                                VALUES (?, ?, ?, ?)
                            """, (prescription_id, herb_name, dose, note))
                            herb_count += 1
                            print(f"✅ 药材添加成功: {herb_name} {dose}g {note if note else ''}")
                        else:
                            print(f"⚠️ 跳过无效剂量行: {row}")

                print(f"✨ 该文件共添加 {herb_count} 种药材")

"""--------------收尾------------------"""
conn.commit()
conn.close()
print("\n🎉 批量数据导入完成！")
