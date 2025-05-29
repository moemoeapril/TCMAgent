import sqlite3
import csv
import os

conn = sqlite3.connect("tcm.db")
cursor = conn.cursor()

# 建表
cursor.execute("""
create table if not exists patients(
   id integer primary key autoincrement,
   code text unique,
   gender int
)
""")

cursor.execute("""
create table if not exists prescriptions(
   id integer primary key autoincrement,
   patient_id int,
   diagonosis text,
   quantity int,
   date text,
   foreign key(patient_id) references patients(id)
)
""")

cursor.execute("""
create table if not exists herbs(
   id integer primary key autoincrement,
   prescriptions_id int,
   herb_name text,
   dose int,
   note text,
   foreign key(prescriptions_id) references prescriptions(id)
)
""")

prep_dir = os.path.join(os.path.dirname(__file__), "prep")

for patient_dir in os.listdir(prep_dir):
    patient_path = os.path.join(prep_dir, patient_dir)
    if os.path.isdir(patient_path):
        for csv_file in os.listdir(patient_path):
            if csv_file.endswith('.csv'):
                file_path = os.path.join(patient_path, csv_file)
                print(f"\n📄处理文件: {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)

                if len(rows) < 5:
                    print(f"⚠️ 文件内容不足，跳过: {file_path}")
                    continue

                # 解析基本信息
                date = rows[0][1].strip()
                patient_code = rows[1][1].strip()
                gender = int(rows[2][1].strip())
                diagonosis = rows[3][1].strip()

                # 插入病人表
                cursor.execute("insert or ignore into patients (code, gender) values(?,?)", (patient_code, gender))
                cursor.execute("select id from patients where code=?", (patient_code,))
                patient_id = cursor.fetchone()[0]

                # 插入处方表
                quantity = 1  # 默认1份
                cursor.execute("""
                    insert into prescriptions(patient_id, diagonosis, quantity, date)
                    values (?, ?, ?, ?)
                """, (patient_id, diagonosis, quantity, date))
                prescription_id = cursor.lastrowid

                # 插入药材表
                herb_count = 0
                for row in rows[4:]:
                    if len(row) >= 2:
                        herb_name = row[0].strip().strip("，, ")
                        dose_str = row[1].strip().strip("，, ")
                        herb_note = row[2].strip() if len(row) > 2 else None

                        if dose_str.isdigit():
                            dose = int(dose_str)
                            cursor.execute("""
                                insert into herbs(prescriptions_id, herb_name, dose, note)
                                values (?, ?, ?, ?)
                            """, (prescription_id, herb_name, dose, herb_note))
                            herb_count += 1
                            print(f"✅ {herb_name} {dose} 添加成功" + (f"，备注: {herb_note}" if herb_note else ""))
                        else:
                            print(f"⚠️ 跳过无效剂量行: {row}")
                
                print(f"🌿 当前处方共添加 {herb_count} 条药材数据")

conn.commit()
conn.close()
print("\n✅ 数据导入完成")
