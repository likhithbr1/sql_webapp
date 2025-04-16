# Step 1: Install packages
%pip install 'vanna[chromadb]' transformers accelerate

# Step 2: Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore

# Step 3: Configuration
MODEL_NAME = "defog/sqlcoder-7b-2"
DB_PATH = "chatbot.db"  # 👈 Ensure you've uploaded this file in Colab

# Step 4: Custom LLM wrapper
class MyCustomLLM(VannaBase):
    def __init__(self, config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        vram = torch.cuda.get_device_properties(0).total_memory

        if vram > 15e9:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto"
            )

        print(f"✅ Model loaded (VRAM: {round(vram / 1e9, 2)} GB)")

    def submit_prompt(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 5: Combine LLM + Vector Store
class MyVanna(ChromaDB_VectorStore, MyCustomLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        MyCustomLLM.__init__(self, config=config)

# Step 6: FK-aware schema trainer
def train_from_db_schema_with_fks(vn, include_fk=True, table_types=('table',)):
    """
    Train Vanna using all DDLs from the SQLite DB, with optional FK extraction.
    """
    placeholders = ','.join(['?'] * len(table_types))
    ddl_query = f"SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND type IN ({placeholders})"
    df_ddl = vn.run_sql(ddl_query, params=table_types)

    trained = 0
    for _, row in df_ddl.iterrows():
        table_name = row['name']
        ddl = row['sql']

        if include_fk:
            try:
                fk_info = vn.run_sql(f"PRAGMA foreign_key_list('{table_name}')")
                fk_clauses = []
                for _, fk_row in fk_info.iterrows():
                    fk_clauses.append(
                        f"FOREIGN KEY ({fk_row['from']}) REFERENCES {fk_row['table']}({fk_row['to']})"
                    )
                if fk_clauses:
                    ddl = ddl.rstrip(');') + ',\n  ' + ',\n  '.join(fk_clauses) + "\n);"
            except Exception as e:
                print(f"⚠️ FK fetch failed for {table_name}: {e}")

        try:
            vn.train(ddl=ddl)
            trained += 1
        except Exception as e:
            print(f"❌ Training failed on table {table_name}: {e}")

    print(f"✅ Trained on {trained} tables (FKs included: {include_fk})")

# Step 7: Instantiate and train
vn = MyVanna()
vn.connect_to_sqlite(DB_PATH)
train_from_db_schema_with_fks(vn)

# Step 8: Ask a question
response = vn.ask("How many completed orders are there?")
print("\n📊 SQL Response & Answer:")
print(response)
