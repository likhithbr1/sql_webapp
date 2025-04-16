# Step 1: Install required packages
%pip install 'vanna[chromadb]' transformers accelerate

# Step 2: Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore

# === Configuration ===
MODEL_NAME = "defog/sqlcoder-7b-2"
DB_PATH = "chatbot.db"  # 👈 Upload your SQLite DB to Colab first

# Step 3: Custom LLM wrapper that satisfies VannaBase requirements
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

    # ✅ Required abstract methods
    def user_message(self, message: str) -> str:
        return message

    def system_message(self, message: str) -> str:
        return message

    def assistant_message(self, message: str) -> str:
        return message

# Step 4: Combine LLM and vector store
class MyVanna(ChromaDB_VectorStore, MyCustomLLM):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        MyCustomLLM.__init__(self, config=config)

# Step 5: Function to train on DDLs + foreign key info
def train_from_db_schema_with_fks(vn, include_fk=True, table_types=('table',)):
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
            print(f"❌ Training failed on {table_name}: {e}")

    print(f"✅ Trained on {trained} tables (FKs included: {include_fk})")

# Step 6: Instantiate Vanna + train
vn = MyVanna()
vn.connect_to_sqlite(DB_PATH)
train_from_db_schema_with_fks(vn)

# Step 7: Ask a natural language question
response = vn.ask("How many completed orders are there?")
print("\n📊 Generated SQL & Answer:")
print(response)



def train_from_db_schema_with_fks(vn, include_fk=True):
    ddl_query = "SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL AND type IN ('table')"
    df_ddl = vn.run_sql(ddl_query)

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
            print(f"❌ Training failed on {table_name}: {e}")

    print(f"✅ Trained on {trained} tables (FKs included: {include_fk})")





def submit_prompt(self, prompt: str, **kwargs) -> str:
    # Handle case where Vanna sends a list of strings
    if isinstance(prompt, list):
        prompt = "\n\n".join(prompt)  # Combine prompt parts

    # Tokenize with padding/truncation enabled
    inputs = self.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to("cuda")

    # Generate using model
    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=False
        )

    # Decode the output and log it
    result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🧾 Model Output:\n", result)
    return result




# Step 3: Define custom LLM wrapper
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
        # Join prompt list if necessary
        if isinstance(prompt, list):
            prompt = "\n\n".join(prompt)

        # Tokenize with padding and truncation
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to("cuda")

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=False
            )

        # Decode and print model output
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n🧾 Model Output:\n", result)
        return result

    def user_message(self, message: str) -> str:
        return message

    def system_message(self, message: str) -> str:
        return message

    def assistant_message(self, message: str) -> str:
        return message

