#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -qU faiss-cpu==1.7.2


# In[19]:


import numpy as np
import pandas as pd
import requests
import json
import re
import faiss
import networkx as nx

from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option("display.max_colwidth", None)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[4]:


get_ipython().run_line_magic('pip', 'install -U azureml-fsspec')

uri = (
  "azureml://subscriptions/ID/resourcegroups/Central_Tools_Prod"
  "/workspaces/marketingamlw/datastores/entity_matching"
  "/paths/Raw/Account_Master_Delta_20251117.csv"
)

df = pd.read_csv(uri)
df.head()


# #### **READ INPUT DATA**

# In[3]:


##df = pd.read_csv('Users/lc5705033/BNY_Sample.csv')
df = pd.read_csv('Users/lc5705033/AMFile.csv')
df = df[:10]
len(df)


# In[7]:


##df["Embedding_input"] = df["CleanName"].fillna("") + " "+df["N_City"].fillna("") + " "+df["N_State"].fillna("") + " "+df["N_Country"].fillna("")
df["Embedding_input"] = df["CleanAccountName"].fillna("")


# In[8]:


emb_inp = (df["Embedding_input"].dropna().astype(str).map(str.strip).loc[lambda s: s.str.len() > 0].unique().tolist())


# In[9]:


print(len(emb_inp))
print(len(df))


# In[85]:


'''
response1 = requests.post(
  "https://embed-v-4-0-qvhcs.eastus2.models.ai.azure.com/v2/embed",
  headers={
    "Authorization": "Bearer token"
  },
  json={
    "model": "embed-v-4.0",
    "input_type": "clustering",
    "texts": company_names[:95],
    "embedding_types": [
      "float"
    ]
  },
)
embed-v-4-0

if response1.status_code == 200:
    data = response1.json()
    print(data)

else:
    print(f"Request failed with status code {response1.status_code}")


response_json = response1.json()
embeddings1 = response_json["embeddings"]["float"]
embeddings1 = np.array(embeddings1)
embedding_dict = {name:emb for name, emb in zip(company_names, embeddings1)}
df["Embedding"] = df["CleanName"].apply(lambda x:embedding_dict.get(str(x)) if pd.notna(x) else None) '''


# #### **EMBEDDING MULTIPLE BATCHES**

# In[16]:


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL =   "https://embed-v-4-0-qvhcs.eastus2.models.ai.azure.com/v2/embed"

def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def make_session():
    s = requests.Session()
    retry = Retry(
        total=6,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    return s

def embed_company_names(
    company_names,
    batch_size=50,              
    model="embed-v-4-0",
    input_type="clustering",
    embedding_type="float",
    timeout = (10,60),
    verify_echo=True,            
):
    if batch_size > 96:
        raise ValueError("batch_size must be <= 96 for this endpoint.")

    headers={"Authorization": "Bearer token"}
    session = make_session()
    all_vectors = []
    for batch_idx, batch in enumerate(chunked(company_names, batch_size), start=1):
        payload = {
            "model": model,
            "input_type": input_type,
            "texts": batch,
            "embedding_types": [embedding_type], 
        }

        resp = session.post(URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Batch {batch_idx} failed: HTTP {resp.status_code}\n{resp.text[:2000]}"
            )
        rj = resp.json()
        if verify_echo:
            returned_texts = rj.get("texts", [])
            if returned_texts != batch:
                raise ValueError(
                    f"Batch {batch_idx}: returned texts do not match request batch.\n"
                    f"Request[0:3]={batch[:3]}\nReturned[0:3]={returned_texts[:3]}"
                )

        batch_vectors = rj["embeddings"][embedding_type]  # <-- your schema
        all_vectors.extend(batch_vectors)

    return all_vectors


# #### **ONE TIME RUN**

# In[17]:


## REPLACE emb_inp with full list instead of first 1000
embeddings = embed_company_names(emb_inp, batch_size=50)
print(len(embeddings), len(emb_inp))
emb_df = pd.DataFrame({"CleanNameEmbedding": emb_inp,"Embedding": embeddings})
clean_to_emb = dict(zip(emb_df["CleanNameEmbedding"], emb_df["Embedding"]))
df["embedding"] = df["Embedding_input"].map(clean_to_emb)


# In[ ]:


##ASSIGN SNO FOR PROMPT OUTPUT ??


# In[12]:


df.head(5)


# In[23]:


X = np.vstack(df["embedding"].values).astype("float32")
dim = X.shape[1]
print(dim)


# In[24]:


index = faiss.IndexFlatIP(dim)
index.add(X)


# In[25]:


print("ntotal:", index.ntotal) ##17716


# In[26]:


K = 10          
SIM_MIN = 0.88  
sims, ids = index.search(X, K)
pairs = []
company_ids = df["Embedding_input"].values

for i in range(len(company_ids)):
    src_id = company_ids[i]

    for j in range(1,K):

        neigh_ids = ids[i][j]
        neigh_sims = sims[i][j]

        n_id =  company_ids[neigh_ids]
        pairs.append({"embedding_input":src_id,
        "neighbor_id": n_id,
        "sim":float(neigh_sims)})
pairs_df = pd.DataFrame(pairs)


# In[44]:


pairs_filtered = pairs_df[pairs_df["sim"] >= 0.87]
pairs_filtered


# In[43]:


pairs_filtered[pairs_filtered["embedding_input"] == 'bny mellon capital markets']


# In[45]:


G = nx.Graph()
G.add_nodes_from(company_ids)
for _, row in pairs_filtered.iterrows():
    G.add_edge(row["embedding_input"],  row["neighbor_id"])


# In[46]:


clusters = list(nx.connected_components(G))
cluster_map = {}
for cluster_id, comp_set in enumerate(clusters):
    for comp in comp_set:
        cluster_map[comp] = cluster_id
df["ClusterID"] = df["Embedding_input"].map(cluster_map)


# In[31]:


df.to_csv("Users/lc5705033/clusters0215.csv", index=False)


# In[36]:


counts_df = (
    df
    .groupby('ClusterID', as_index=False)
    .size()
    .rename(columns={'size':'count'})
)
counts_df[counts_df['count'] > 1]


# In[47]:


counts_df = (df.groupby('ClusterID', as_index=False).size().rename(columns={'size':'count'}))
counts_df[counts_df['count'] > 1]


# #### **INCREMENTAL**

# In[19]:


import os
CSV_PATH = "Users/lc5705033/companyembeddings.csv"
MODEL_VERSION = "Cohere embed V4"

if os.path.exists(CSV_PATH):
    db = pd.read_csv(CSV_PATH)
else:
    db = pd.DataFrame(columns=[
        "FaissId", "CleanName", "CleanAccountName","Embedding_input","Embedding", "EmbeddingDim", "ModelVersion", "ClusterID"])


# In[22]:


incremental_rec = df.drop_duplicates("Embedding_input").copy()
existing_names = set(db["Embedding_input"])
incremental_rec["Exists"] = incremental_rec["Embedding_input"].isin(existing_names)
print(len(incremental_rec), len(existing_names))


# In[23]:


mask_new = incremental_rec["Exists"] == False
n_new = int(mask_new.sum())
idx_new = incremental_rec.index[mask_new]
idx_existing = incremental_rec.index[~mask_new]

texts_new = incremental_rec.loc[idx_new, "Embedding_input"].astype(str).tolist()


# In[24]:


print(len(incremental_rec), len(mask_new), mask_new.sum(), len(idx_new), len(idx_existing))


# In[25]:


## Calling Embedding Model (batch wise)
emb_list = embed_company_names(texts_new, batch_size=50)


# In[26]:


print(len(emb_list), len(idx_new))


# In[ ]:


#### map back to original df ?????


# In[27]:


assert len(emb_list) == len(idx_new)

if len(db) == 0 or db["FaissId"].isna().all():
    start_id = 0
else:
    start_id = int(db["FaissId"].max()) + 1

new_ids = np.arange(start_id, start_id + len(idx_new), dtype=np.int64)
incremental_rec.loc[idx_new, "EmbeddingList"] = pd.Series(emb_list, index=idx_new, dtype="object").values
incremental_rec.loc[idx_new, "FaissId"] = pd.array(new_ids, dtype="Int64")
incremental_rec.loc[idx_new, "ModelVersion"] = "Cohere embed v4"
incremental_rec.loc[idx_new, "EmbeddingDim"] = "1536"
incremental_rec.loc[idx_new, "Embedding"] = pd.Series(emb_list, index=idx_new, dtype="object").values


# In[55]:


if "ClusterID" not in incremental_rec.columns:
    incremental_rec["ClusterID"] = pd.array([None] * len(incremental_rec), dtype="Int64")
else:
    # make sure it's nullable Int64 for safe NA handling
    incremental_rec["ClusterID"] = incremental_rec["ClusterID"].astype("Int64")


# In[56]:


incremental_rec.head(3)


# In[37]:


db.head(5)


# In[29]:


## Create FAISS

INDEX_PATH = "vector_index1.faiss"
dim = 1536  # embedding dimension

if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    base = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap2(base)


# In[30]:


vecs = np.vstack([np.asarray(v, dtype="float32") for v in emb_list])
faiss.normalize_L2(vecs)
index.add_with_ids(vecs, new_ids)
faiss.write_index(index, INDEX_PATH)


# In[31]:


left = incremental_rec.loc[idx_existing, ["Embedding_input"]].merge(
            db[["Embedding_input", "FaissId", "Embedding", "EmbeddingDim", "ModelVersion", "ClusterID"]],
            on="Embedding_input",
            how="left",
            suffixes=("", "_db"),
        )

# Assign FaissId & Embedding from db for existing records
incremental_rec.loc[idx_existing, "FaissId"] = pd.array(left["FaissId"], dtype="Int64")
incremental_rec.loc[idx_existing, "EmbeddingList"] = left["Embedding"].values
incremental_rec.loc[idx_existing, "ModelVersion"] = left["ModelVersion"].values
incremental_rec.loc[idx_existing, "EmbeddingDim"] = left["EmbeddingDim"].values
incremental_rec.loc[idx_existing, "Embedding"] = left["Embedding"].values


# In[32]:


index = faiss.read_index("vector_index1.faiss")

print("Index class:", type(index))
print("is_trained:", index.is_trained)
print("ntotal:", index.ntotal)
print("d (dim):", index.d)


# In[112]:


output_db = pd.read_csv(CSV_PATH)

print("CSV rows:", len(output_db))
print("FAISS ntotal:", index.ntotal, "dim:", index.d)

# show a few entries sorted by id
sample = output_db.sort_values("FaissId").head(10)[["FaissId", "CleanName", "CleanAccountName", "ModelVersion"]]
print(sample.to_string(index=False))


# In[246]:


db.to_csv(CSV_PATH, index=False)


# In[34]:


db[db["ClusterID"] == 9992]


# In[35]:


K = 10       
threshold = 0.88

index_path = faiss.read_index("vector_index1.faiss")
db["Embedding"] = db['Embedding'].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else np.array(x))
faissid_to_cluster = dict(zip(db["FaissId"], db["ClusterID"]))
if db["ClusterID"].notnull().any():
    current_max_cluster = int(db["ClusterID"].max())
else:
    current_max_cluster = 0


# In[37]:


db[db["ClusterID"] == 9992]


# In[57]:


#new_cluster_df = incremental_rec["ClusterID"].isna()
#idx_new_cluster = incremental_rec.index[mask_new]
#print(len(idx_new_cluster))
#V = np.vstack(incremental_rec.loc[idx_new_cluster, "Embedding"].apply(lambda x: np.asarray(json.loads(x) if isinstance(x, str) else x, dtype=np.float32)).to_list())
V = np.vstack(incremental_rec["Embedding"].apply(lambda x: np.asarray(json.loads(x) if isinstance(x, str) else x, dtype=np.float32)).to_list())
V = normalize(V, axis = 1)
D,I = index.search(V, K)  


# In[61]:


new_cluster_assignments = {}
db_faiss = db["FaissId"].values

for i, row_idx in enumerate(incremental_rec.index):
    best_cluster = None
    best_score = 0
    for j in range(K):
        neighbor_pos = I[i][j]
        sim_score = D[i][j]

        if neighbor_pos >= len(db_faiss):
            continue
        neighbor_faiss_id = db_faiss[neighbor_pos]
        neighbor_cluster = faissid_to_cluster[neighbor_faiss_id]
        if sim_score > best_score and sim_score >= threshold:
            best_score = sim_score
            best_cluster = neighbor_cluster

    if best_cluster is not None:
        new_cluster_assignments[row_idx] = best_cluster
    else:
        current_max_cluster += 1
        new_cluster_assignments[row_idx] = current_max_cluster


# In[62]:


for idx, cid in new_cluster_assignments.items():
    incremental_rec.loc[idx, "ClusterID"] = cid


# In[63]:


final_df = df.merge(incremental_rec[["Embedding_input", "" "ClusterID"]], on = 'Embedding_input', how = 'left')


# In[64]:


final_df


# In[65]:


final_df["ClusterID"].isnull().any()


# In[40]:


db[db["CleanName"] == "alterna capital partners"]


# In[41]:


db[db["CleanName"] == "alterna equity partners"]


# In[ ]:


updated_rec = pd.concat([db,final_df], ignore_index = True)


# In[69]:


new_rows = incremental_rec.loc[idx_new,["FaissId", "CleanName", "CleanAccountName","Embedding_input",
    "Embedding", "EmbeddingDim", "ModelVersion", "ClusterID"]]

# upsert by FaissId
db = (pd.concat([db, new_rows], ignore_index=True)
      .drop_duplicates(subset=["FaissId"], keep="last")
      .reset_index(drop=True))


# In[77]:


db[db["Embedding_input"] == 'user']


# In[76]:


db[db["Embedding_input"] == 'abbott labotories']


# 

# In[90]:


db.to_csv(CSV_PATH, index = False)  ## DB


# #### **ONE TIME RUN CONTINUTION - FILTER CLUSTERS - INPUT TO PROMPT**

# In[53]:


summary = (
    df.assign(is_known=df["Comments"].str.lower().eq("known"))
      .groupby("ClusterID")
      .agg(
          total_count=("ClusterID", "size"),
          known_count=("is_known", "sum"),
          unknown_count=("is_known", lambda s: (~s).sum())
      )
      .reset_index())

filtered = summary[(summary["total_count"] > 1) & (summary["known_count"] > 0) & (summary["unknown_count"] > 0)]
filtered.sort_values(["known_count", "total_count"], ascending=False)


# In[54]:


filtered


# In[284]:


history = pd.read_csv('Users/lc5705033/cluster_0212.csv') ## READ HISTORY DATA IF INCREMENTAL


# In[68]:


cluster_ids = filtered["ClusterID"].unique()
related_history = df[df["ClusterID"].isin(cluster_ids)].copy()


# In[292]:


related_incremental = incremental_rec[incremental_rec["ClusterID"].isin(cluster_ids)].copy()


# In[296]:


combined = pd.concat([related_incremental, related_history], ignore_index=True)


# In[69]:


combined = related_history


# #### **PROMPT MODEL**

# In[59]:


def query_cohere_command_r(prompt):
    response = requests.post(
    "https://Cohere-command-r-08-2024-rceum.eastus2.models.ai.azure.com/v2/chat",
    headers={
        "Authorization": "Bearer 40qwxm3KGNAzgHLIIafT7yq8nfXdhoVi",
        "Content-Type": "application/json"
    },

    json={
            "model": "Cohere-command-r-08-2024",
            "temperature": 0,
            "seed": 42,
            "messages": [{"role": "user", "content": prompt}]
    }

)
    return response.json()


# In[178]:


##df = df.drop(columns=["newid"])


# In[76]:


combined["newid"] = None

BATCH_SIZE = 50   # adjust

for cluster in combined["ClusterID"].unique():
    subset   = combined[combined["ClusterID"] == cluster]
    knowns   = subset[subset["Comments"] == "known"].copy()
    unknowns = subset[subset["Comments"] == "unknown"].copy()

    default_new = f"NEW_{cluster}"

    # If no knowns then mark all unknowns NEW_cluster
    if knowns.empty or unknowns.empty:
        for idx in unknowns.index:
            combined.at[idx, "newid"] = default_new
        continue

    for df_part in (knowns, unknowns):
        for col in ["CleanAddressLine1","CleanAddressLine2"]:
            df_part[col] = df_part[col].apply(lambda x: None if x is None else (str(x).strip() or None))

    known_list = knowns[["C360AccountID","CleanName","SNO","CleanAddressLine1","CleanAddressLine2","N_City","N_State","N_Country","CleanPostalCode","CleanWebsite"]].replace({np.nan: None}).to_dict(orient="records")
    unknown_list = unknowns[["SNO","CleanName","CleanAddressLine1","CleanAddressLine2","N_City","N_State","N_Country","CleanPostalCode","CleanWebsite"]].replace({np.nan: None}).to_dict(orient="records")

    # keep df index for write‑back
    for rec, idx in zip(unknown_list, unknowns.index.tolist()):
        rec["_df_index"] = idx


    for i in range(0, len(unknown_list), BATCH_SIZE):
        batch = unknown_list[i:i+BATCH_SIZE]
        prompt = f"""
You are an expert in company entity resolution.

TASK
----
For each UNKNOWN record, assign exactly one KNOWN C360AccountId OR assign NEW_{cluster}.

IMPORTANT: PRE-NORMALIZED FIELDS
--------------------------------
The following fields are ALREADY normalized in Python preprocessing and MUST be used as-is:
- N_Country (canonical; e.g., 'united states')
- N_State   (canonical for US only; e.g., 'new york')
- N_City    (canonical for US only; e.g., 'new york')

Do NOT attempt to reinterpret, expand, or remap these three fields.

INPUTS
------
KNOWN records contain:
- C360AccountId
- CleanName
- CleanAddressLine1
- CleanAddressLine2
- N_City
- N_State
- N_Country
- CleanPostalCode
- CleanWebsite
{known_list}

UNKNOWN records contain:
- SNO
- CleanName
- CleanAddressLine1
- CleanAddressLine2
- N_City
- N_State
- N_Country
- CleanPostalCode
- CleanWebsite
{batch}

HARD LOCATION GATES (STOP RULES)
--------------------------------

Gate 1 (CITY + COUNTRY STOP GATE):
- Gate 1 is APPLICABLE only when UNKNOWN.N_Country is present AND UNKNOWN.N_City is present.
- If applicable, candidate KNOWN records are those where:
    KNOWN.N_Country == UNKNOWN.N_Country AND KNOWN.N_City == UNKNOWN.N_City
- If Gate 1 is applicable and the candidate set is empty:
    FAILED GATE = "1", Scoring = 0, AssignedID = NEW_{cluster}

If Gate 1 is NOT applicable (missing city or country), do NOT fail Gate 1.
Continue to other gates using available fields.

Gate 2 (STATE STOP GATE when BOTH present):
- If N_State is present on BOTH UNKNOWN and KNOWN: must match exactly
  else FAILED GATE = "2" and NEW_{cluster}.

Gate 3 (POSTAL STOP GATE when BOTH present):
- If CleanPostalCode is present on BOTH: must match exactly
  else FAILED GATE = "3" and NEW_{cluster}.

Gate 4 (ADDRESS MISSING on either side):
- Address Lines are present and should exactly MATCH.
- If UNKNOWN has BOTH address lines missing OR KNOWN has BOTH address lines missing:
  require ALL:
    - (N_City + N_Country match) when both present and should exctly MATCH
    - (N_State match) when both present and should exctly MATCH
    - (CleanPostalCode match) when both present and should exctly MATCH
  else FAILED GATE = "4" and NEW_{cluster}.

Gate 5 (ONE-SIDED missing address lines):
- If one side is missing BOTH address lines (line1+line2 missing) but Gate 4 did not apply:
  require:
    - (N_City + N_Country match) when both present
  else FAILED GATE = "5" and NEW_{cluster}.

SCORING (ONLY after all applicable gates pass)
----------------------------------------------
Address Override (score = 85) is allowed ONLY if:
- N_Country and N_City equal (when both present)
- N_State equal (when both present)
- If both have CleanPostalCode -> exact match
- Street number AND normalized street name match exactly
If any condition fails -> Override NOT allowed.

Otherwise compute score:
- Address alignment (0–75):
  - exact street number + exact normalized street name: +55
  - unit/suite neutral: +15 (only if base street already matches)
  - city/state/country aligned: +5 (only if Gate 1 applicable and passed)
- Website/domain (0–15):
  - exact apex-domain equality: +15
- Name similarity (0–10):
  - high: +10, moderate: +5, weak/generic: +0

Minimum threshold:
- Assign KNOWN if score >= 65
- Else NEW_{cluster}

TIE-BREAKERS:
1) higher Address alignment
2) domain equality
3) higher name similarity

OUTPUT FORMAT (STRICTLY FOLLOW FORMAT)
----------------------
Return a dictionary with one entry per UNKNOWN SNO:

"{{SNO}}": {{
  "CleanAccountName": "<from UNKNOWN CleanName>",
  "AssignedID": "<C360AccountId or NEW_{cluster}>",
  "Reasoning": "<ONE short sentence>"
}}

ANTI-HALLUCINATION (ENFORCE)
----------------------------
- If Gate 1 applicable fails -> Scoring MUST be 0 and FAILED GATE MUST be "1".
- NEVER say "Address matches" unless street number AND normalized street name match exactly.
- NEVER match across different cities/countries.
- NEVER match across different states when both present.
- If uncertain or score < 65 -> assign NEW_{cluster}.
"""


        resp = query_cohere_command_r(prompt)
        ##print(resp)
        if isinstance(resp, str):
            text = resp
        elif hasattr(resp, "text"):
            text = resp.text
        elif isinstance(resp, dict):
            try:
                text = resp["message"]["content"][0]["text"]
            except:
                text = resp.get("text", "{}")
        else:
            text = "{}"

        text = text or "{}"
        start, end = text.find("{"), text.rfind("}")
        payload = {}
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start:end+1])
            except Exception as ex:
                payload = {}

        mapping = {}


        if isinstance(payload, dict) and not payload.get("matches"):
            for sno_key, rec in payload.items():
                assigned = None
                if isinstance(rec, dict):
                    assigned = rec.get("AssignedID") or rec.get("newid")
                if assigned:
                    mapping[str(sno_key)] = str(assigned)

        elif isinstance(payload, dict) and isinstance(payload.get("matches"), list):
            for m in payload["matches"]:
                sno = str(m.get("unknown_sno") or m.get("SNO") or m.get("sno") or "")
                assigned = m.get("AssignedID") or m.get("newid")
                if sno and assigned:
                    mapping[sno] = str(assigned)

        elif isinstance(payload, list):
            for m in payload:
                if not isinstance(m, dict):
                    continue
                sno = str(m.get("unknown_sno") or m.get("SNO") or m.get("sno") or "")
                assigned = m.get("AssignedID") or m.get("newid")
                if sno and assigned:
                    mapping[sno] = str(assigned)

        for rec in batch:
          sno = str(rec["SNO"])
          idx = rec["_df_index"]
          combined.at[idx, "newid"] = mapping.get(sno, default_new)

'''   # parse the JSON
        try:
            s = text.find("{")
            e = text.rfind("}")
            parsed = json.loads(text[s:e+1])
            matches = parsed.get("matches", [])
        except:
            matches = []

        # build mapping
        mapping = {}
        for m in matches:
            sno = str(m.get("unknown_sno"))
            val = m.get("newid") or default_new
            mapping[sno] = val

        # write back
        for rec in batch:
            sno = str(rec["SNO"])
            idx = rec["_df_index"]
            combined.at[idx, "newid"] = mapping.get(sno, default_new)'''


# In[77]:


combined[["ClusterID", "SNO", "newid"]]


# In[78]:


assigned_df = combined[combined["newid"].notna() & ~combined["newid"].astype(str).str.startswith("NEW_")].copy()
assigned_df[["ClusterID", "SNO", "newid"]].head()


# In[79]:


total_unknowns  = (combined["Comments"].str.lower() == "unknown").sum()
assigned_count  = assigned_df.shape[0]
new_count       = total_unknowns - assigned_count
print(f"Assigned to existing: {assigned_count}")
print(f"Marked as NEW_*:     {new_count}")


# #### **EVALUATION**

# In[86]:


from difflib import SequenceMatcher

combined["Comments"] = combined["Comments"].astype(str).str.lower().str.strip()
df["Comments"] = df["Comments"].astype(str).str.lower().str.strip()

assigned_unknowns = combined[
    (combined["Comments"] == "unknown") &
    combined["newid"].notna() &
    ~combined["newid"].astype(str).str.startswith("NEW_")].copy()

print("Assigned unknowns:", assigned_unknowns.shape[0])

assigned_merged = assigned_unknowns.merge(
    df.add_prefix("k_"),
    left_on="newid",
    right_on="k_C360AccountID",
    how="left"
)

##print(assigned_merged)
strip_upper = lambda x: "" if pd.isna(x) else re.sub(r"[^A-Z0-9 ]+", "", str(x).upper().strip())
norm_space  = lambda s: re.sub(r"\s+", " ", s).strip()

assigned_merged["name_u"] = assigned_merged["CleanName"].map(strip_upper).map(norm_space)
assigned_merged["name_k"] = assigned_merged["CleanName"].map(strip_upper).map(norm_space)

assigned_merged["addr_u"] = (
    assigned_merged["CleanAddressLine1"].fillna("") + " " +
    assigned_merged["CleanAddressLine2"].fillna("") + " " +
    assigned_merged["N_City"].fillna("") + " " +
    assigned_merged["N_State"].fillna("") + " " +
    assigned_merged["N_Country"].fillna("") + " " +
    assigned_merged["CleanPostalCode"].fillna("")
).map(strip_upper).map(norm_space)

assigned_merged["addr_k"] = (
    assigned_merged["CleanAddressLine1"].fillna("") + " " +
    assigned_merged["CleanAddressLine2"].fillna("") + " " +
    assigned_merged["N_City"].fillna("") + " " +
    assigned_merged["N_State"].fillna("") + " " +
    assigned_merged["N_Country"].fillna("") + " " +
    assigned_merged["CleanPostalCode"].fillna("")
).map(strip_upper).map(norm_space)

to_domain = lambda v: "" if pd.isna(v) else re.sub(r"^www\.", "", re.sub(r"^https?://", "", str(v).strip().lower())).split("/")[0]
assigned_merged["web_u"] = assigned_merged["CleanWebsite"].map(to_domain)
assigned_merged["web_k"] = assigned_merged["CleanWebsite"].map(to_domain)

assigned_merged["name_sim"] = [
    round(100 * SequenceMatcher(None, u, k).ratio(), 1)
    for u, k in zip(assigned_merged["name_u"], assigned_merged["name_k"])
]

assigned_merged["postal_match"] = (
    assigned_merged["CleanPostalCode"].fillna("").astype(str).str.upper().str.strip()
    ==
    assigned_merged["CleanPostalCode"].fillna("").astype(str).str.upper().str.strip()
)

assigned_merged["geo_match"] = (
    (assigned_merged["N_City"].fillna("").str.upper().str.strip()
     == assigned_merged["N_City"].fillna("").str.upper().str.strip()) &
    (assigned_merged["N_State"].fillna("").str.upper().str.strip()
     == assigned_merged["N_State"].fillna("").str.upper().str.strip()) &
    (assigned_merged["N_Country"].fillna("").str.upper().str.strip()
     == assigned_merged["N_Country"].fillna("").str.upper().str.strip())
)

assigned_merged["addr_exact"] = (assigned_merged["addr_u"] == assigned_merged["addr_k"])

tok = lambda s: set([t for t in s.split() if t])
addr_jaccard = []
for u, k in zip(assigned_merged["addr_u"], assigned_merged["addr_k"]):
    su, sk = tok(u), tok(k)
    j = (len(su & sk) / len(su | sk)) if (su | sk) else 0.0
    addr_jaccard.append(round(100 * j, 1))
assigned_merged["addr_jaccard"] = addr_jaccard

assigned_merged["web_match"] = (assigned_merged["web_u"] == assigned_merged["web_k"])

NAME_SIM_THR = 90.0
ADDR_JAC_THR = 70.0

assigned_merged["is_same"] = (
    assigned_merged["web_match"] |
    (
        (assigned_merged["name_sim"] >= NAME_SIM_THR) &
        (
            assigned_merged["addr_exact"] |
            (assigned_merged["addr_jaccard"] >= ADDR_JAC_THR) |
            (assigned_merged["postal_match"] & assigned_merged["geo_match"])
        )
    )
)

agreement_rate = float(assigned_merged["is_same"].mean()) if len(assigned_merged) else np.nan
print(f"Rule-based agreement rate on assigned matches: {agreement_rate:.2%}  (n={len(assigned_merged)})")

assigned_merged["unknown_address"] = (
    assigned_merged["CleanAddressLine1"].fillna("") + " " +
    assigned_merged["CleanAddressLine2"].fillna("") + ", " +
    assigned_merged["N_City"].fillna("") + ", " +
    assigned_merged["N_State"].fillna("") + ", " +
    assigned_merged["N_Country"].fillna("") + " " +
    assigned_merged["CleanPostalCode"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.replace(r"\s+,", ",", regex=True).str.strip(" ,")

assigned_merged["assigned_address"] = (
    assigned_merged["CleanAddressLine1"].fillna("") + " " +
    assigned_merged["CleanAddressLine2"].fillna("") + ", " +
    assigned_merged["N_City"].fillna("") + ", " +
    assigned_merged["N_State"].fillna("") + ", " +
    assigned_merged["N_Country"].fillna("") + " " +
    assigned_merged["CleanPostalCode"].fillna("")
).str.replace(r"\s+", " ", regex=True).str.replace(r"\s+,", ",", regex=True).str.strip(" ,")

assigned_eval = assigned_merged[[
    "ClusterID","SNO","newid",                              # resolvable identifiers
    "CleanName","unknown_address","CleanWebsite",           # unknown (left) side
    "C360AccountID","CleanName","assigned_address","CleanWebsite",  # assigned account (right) side
    "name_sim","addr_exact","addr_jaccard","postal_match","geo_match","web_match","is_same"  # metrics
]].copy()

print(assigned_eval.head(10))
#assigned_eval.to_csv("assigned_matches_evaluation.csv", index=False)


# In[85]:


pd.set_option("display.max_colwidth", None)


# In[87]:


from collections import Counter
combined["Comments"] = combined["Comments"].astype(str).str.lower().str.strip()
df["Comments"] = df["Comments"].astype(str).str.lower().str.strip()

assigned_unknowns = combined[
    (combined["Comments"] == "unknown") &
    combined["newid"].notna() &
    ~combined["newid"].astype(str).str.startswith("NEW_")].copy()

lookup_cols = [
    "C360AccountID","CleanName","CleanAddressLine1","CleanAddressLine2",
    "N_City","N_State","N_Country","CleanPostalCode","CleanWebsite","ClusterID"
]
assigned_merged = assigned_unknowns.merge(
    df[lookup_cols].add_prefix("k_"),
    left_on="newid",
    right_on="k_C360AccountID",
    how="left"
)

assigned_merged = assigned_merged[assigned_merged["k_C360AccountID"].notna()].copy()

def _norm_text(x):
    return "" if pd.isna(x) else str(x).strip()

def _to_domain(u):
    if pd.isna(u): 
        return ""
    s = str(u).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s)
    return s.split("/")[0]

def _record_json(side, row, prefix=""):
    return {
        "Side": side,
        "CleanName": _norm_text(row[prefix+"CleanName"]),
        "CleanAddressLine1": _norm_text(row[prefix+"CleanAddressLine1"]),
        "CleanAddressLine2": _norm_text(row[prefix+"CleanAddressLine2"]),
        "N_City": _norm_text(row[prefix+"N_City"]),
        "N_State": _norm_text(row[prefix+"N_State"]),
        "N_Country": _norm_text(row[prefix+"N_Country"]),
        "CleanPostalCode": _norm_text(row[prefix+"CleanPostalCode"]),
        "CleanWebsite": _norm_text(row[prefix+"CleanWebsite"]),
        "WebsiteDomain": _to_domain(row[prefix+"CleanWebsite"]),
    }

def _judge_prompt(u_json, a_json):
    return f"""
You are an entity-resolution evaluator. Decide if the UNKNOWN record and the ASSIGNED account refer to the same organization.
Use the rubric and return STRICT JSON only.

Rubric:
- Strong signals: CleanName similarity, website domain match, address line(s) + city/state/country + postal code.
- Consider punctuation/spacing/abbreviation variants.
- If evidence conflicts, be cautious.

Output JSON:
{{
  "decision": "same" | "not_same" | "unsure",
  "scores": {{
    "name_sim_0_100": <number>,
    "website_match": true|false,
    "address_confidence_0_100": <number>
  }},
  "rationale": "<2-4 sentences citing fields>"
}}

UNKNOWN:
{json.dumps(u_json, ensure_ascii=False)}

ASSIGNED_ACCOUNT:
{json.dumps(a_json, ensure_ascii=False)}
""".strip()

def _extract_text(resp):
    if isinstance(resp, str): 
        return resp
    if hasattr(resp, "text"): 
        return resp.text
    if isinstance(resp, dict):
        try:
            return resp["message"]["content"][0]["text"]
        except Exception:
            return resp.get("text", "{}")
    return "{}"

def _parse_judge_json(txt):
    try:
        s, e = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[s:e+1])
        decision = str(obj.get("decision","unsure")).strip()
        scores = obj.get("scores", {}) or {}
        name_sim = float(scores.get("name_sim_0_100", 0))
        website_match = bool(scores.get("website_match", False))
        addr_conf = float(scores.get("address_confidence_0_100", 0))
        rationale = str(obj.get("rationale","")).strip()
        return decision, name_sim, website_match, addr_conf, rationale
    except Exception:
        return "unsure", 0.0, False, 0.0, ""

K = 3 

judge_rows = []
for idx, row in assigned_merged.iterrows():
    unknown_json  = _record_json("UNKNOWN", row, prefix="")
    assigned_json = _record_json("ASSIGNED", row, prefix="k_")

    prompts = [
        _judge_prompt(unknown_json, assigned_json),
        _judge_prompt(assigned_json, unknown_json)
    ]

    all_votes = []
    all_scores = [] 
    rationales = []

    for p in prompts:
        for _ in range(K):
            resp = query_cohere_command_r(p)
            txt = _extract_text(resp)
            d, name_sim, web_match, addr_conf, rat = _parse_judge_json(txt)

            all_votes.append(d)
            all_scores.append({
                "name_sim": name_sim,
                "website_match": 1.0 if web_match else 0.0,
                "addr_conf": addr_conf
            })
            if rat:
                rationales.append(rat)

    vote_counts = Counter(all_votes)
    majority_decision, maj_ct = vote_counts.most_common(1)[0]

    if all_scores:
        name_sim_mean = float(np.median([s["name_sim"] for s in all_scores]))
        web_match_rate = float(np.mean([s["website_match"] for s in all_scores]))
        addr_conf_mean = float(np.median([s["addr_conf"] for s in all_scores]))
    else:
        name_sim_mean, web_match_rate, addr_conf_mean = 0.0, 0.0, 0.0
    if vote_counts.get("same",0) >= K and vote_counts.get("not_same",0) >= K:
        final_decision = "unsure"
    else:
        final_decision = majority_decision

    rationale_out = ""
    if rationales:
        uniq = []
        for r in rationales:
            r = r.strip()
            if r and r not in uniq:
                uniq.append(r)
            if len(uniq) >= 2:
                break
        rationale_out = " | ".join(uniq)

    judge_rows.append({
        "index": row.name,
        "SNO": row["SNO"],
        "ClusterID": row["ClusterID"],
        "newid": row["newid"],
        "judge_decision": final_decision,
        "judge_name_sim": round(name_sim_mean, 1),
        "judge_website_agree_rate": round(100*web_match_rate, 1),   # %
        "judge_addr_conf": round(addr_conf_mean, 1),
        "judge_votes_same": vote_counts.get("same",0),
        "judge_votes_not_same": vote_counts.get("not_same",0),
        "judge_votes_unsure": vote_counts.get("unsure",0),
        "judge_rationale": rationale_out
    })
    
judge_df = pd.DataFrame(judge_rows).set_index("index")
for c in judge_df.columns:
    df.loc[judge_df.index, c] = judge_df[c]

# Quick view
df.loc[judge_df.index, ["ClusterID","SNO","newid","judge_decision","judge_name_sim","judge_website_agree_rate","judge_addr_conf"]].head()


# In[ ]:





# In[23]:


##combined_df.to_csv('Users/lc5705033/id.csv', index=False)
