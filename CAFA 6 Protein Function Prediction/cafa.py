import pandas as pd
import numpy as np
from Bio import SeqIO
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import tempfile
import re
import sys

# 确保sklearn导入正确
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.model_selection import cross_val_score
except ImportError as e_import:
    raise ImportError("❌ 未找到sklearn库！请先执行：pip install scikit-learn") from e_import

from typing import List, Dict, Set, Tuple, Any, Union

# -------------------------- 1. 配置参数（降低过滤门槛便于调试）--------------------------
TEST_FASTA = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//Test//testsuperset.fasta"
GO_OBO = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//Train//go-basic.obo"
IA_TSV = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//IA.tsv"
TRAIN_TERMS = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//Train//train_terms.tsv"
TRAIN_TAXON = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//Train//train_taxonomy.tsv"
TRAIN_FASTA = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//Train//train_sequences.fasta"
OUTPUT_SUBMISSION = "/home/aaa/Kaggle-Series-Competition/CAFA 6 Protein Function Prediction//cafa6_data//high_score_submission.tsv"

# 调整参数：降低过滤门槛，便于排查匹配问题
MAX_TERMS_PER_PROT: int = 400
PROPAGATION_DEPTH: int = 2
MIN_TRAIN_COUNT: int = 3  # 临时调低，确保有数据
BATCH_SIZE: int = 300
PROCESS_NUM: int = 3
PROB_THRESHOLD: float = 0.15
IA_DEFAULT_VALUE: float = 0.0
FORCE_OVERLAP_TERMS: List[str] = ["GO:0003674", "GO:0008150", "GO:0005575"]

# 高分参数
HIGH_IA_THRESHOLD: float = 1.3
SEQ_FEATURE_DIM: int = 578
MODEL_N_ESTIMATORS: int = 100
VALIDATION_FOLDS: int = 3
STANDARD_AAS: List[str] = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                           'Y', 'V']


# -------------------------- 2. 工具函数 --------------------------
def normalize_go_term(term: str) -> str:
    term = str(term).strip().upper()
    match = re.match(r'^GO:\d{7}', term)
    return match.group(0) if match else ""


def extract_seq_features(seq: str) -> np.ndarray:
    seq = seq.upper().replace('X', '').replace('*', '').replace('-', '')
    if len(seq) < 10:
        return np.zeros(SEQ_FEATURE_DIM, dtype=np.float32)

    # 1. 氨基酸组成
    aac_count = Counter(seq)
    aac_features: np.ndarray = np.array(
        [aac_count.get(aa, 0) / len(seq) for aa in STANDARD_AAS],
        dtype=np.float32
    )

    # 2. 二肽组成
    dpc_pairs = [a + b for a in STANDARD_AAS for b in STANDARD_AAS]
    dpc_count = Counter()
    for i in range(len(seq) - 1):
        dpc = seq[i] + seq[i + 1]
        if dpc in dpc_pairs:
            dpc_count[dpc] += 1
    total_dpc = len(seq) - 1 if len(seq) > 1 else 1
    dpc_features: np.ndarray = np.array(
        [dpc_count.get(pair, 0) / total_dpc for pair in dpc_pairs],
        dtype=np.float32
    )

    # 3. 理化性质
    physchem: Dict[str, List[float]] = {
        'A': [1.8, -0.5, 0.48, 0.23, 0.31, 0.25],
        'R': [-4.5, 3.0, 1.81, 0.95, 0.82, 0.96],
        'N': [-3.5, 0.2, 0.92, 0.42, 0.48, 0.39],
        'D': [-3.5, 3.0, 0.82, 0.40, 0.46, 0.38],
        'C': [2.5, -1.0, 0.55, 0.29, 0.34, 0.28],
        'Q': [-3.5, 0.2, 1.11, 0.58, 0.61, 0.54],
        'E': [-3.5, 3.0, 0.96, 0.47, 0.52, 0.46],
        'G': [-0.4, -0.3, 0.00, 0.00, 0.00, 0.00],
        'H': [-3.2, -0.5, 1.00, 0.59, 0.52, 0.56],
        'I': [4.5, -1.8, 0.73, 0.31, 0.34, 0.30],
        'L': [3.8, -1.8, 0.73, 0.31, 0.34, 0.30],
        'K': [-3.9, 3.0, 1.19, 0.63, 0.67, 0.60],
        'M': [1.9, -1.3, 0.74, 0.38, 0.40, 0.38],
        'F': [2.8, -2.5, 1.00, 0.54, 0.54, 0.53],
        'P': [-1.6, 0.0, 0.59, 0.34, 0.34, 0.34],
        'S': [-0.8, 0.3, 0.39, 0.20, 0.23, 0.18],
        'T': [-0.7, 0.3, 0.58, 0.31, 0.32, 0.29],
        'W': [-0.9, -3.4, 1.35, 0.77, 0.75, 0.76],
        'Y': [-1.3, -2.3, 1.04, 0.58, 0.56, 0.58],
        'V': [4.2, -1.5, 0.61, 0.27, 0.29, 0.26]
    }
    seq_physchem: List[List[float]] = [physchem.get(aa, [0] * 6) for aa in seq]
    seq_physchem_np: np.ndarray = np.array(seq_physchem, dtype=np.float32)
    physchem_stats: np.ndarray = np.hstack([
        np.mean(seq_physchem_np, axis=0),
        np.var(seq_physchem_np, axis=0),
        np.max(seq_physchem_np, axis=0),
        np.min(seq_physchem_np, axis=0)
    ])
    extra_stats: np.ndarray = np.array([
        len(seq), len(seq) / 100,
        sum(1 for aa in seq if aa in ['R', 'K', 'D', 'E']) / len(seq) if len(seq) > 0 else 0,
        sum(1 for aa in seq if aa in ['A', 'V', 'I', 'L', 'M', 'F', 'W']) / len(seq) if len(seq) > 0 else 0
    ], dtype=np.float32)
    physchem_features: np.ndarray = np.zeros(158, dtype=np.float32)
    physchem_features[:len(physchem_stats)] = physchem_stats
    physchem_features[len(physchem_stats):len(physchem_stats) + len(extra_stats)] = extra_stats

    return np.hstack([aac_features, dpc_features, physchem_features])


# -------------------------- 3. 数据预处理（添加调试与ID匹配优化）--------------------------
def load_and_preprocess() -> Dict[str, Any]:
    print("=" * 50)
    print("=== 数据预处理 ===")

    # 1. 解析GO本体
    valid_go: Dict[str, str] = {}
    go_parents: Dict[str, List[str]] = defaultdict(list)
    current_id: str = ""
    try:
        with open(GO_OBO, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("id: "):
                    id_part: str = line.split("id: ")[1].strip()
                    current_id = normalize_go_term(id_part)
                    if current_id:
                        valid_go[current_id] = ""
                elif line.startswith("is_obsolete: true") and current_id in valid_go:
                    valid_go[current_id] = "obsolete"
                elif line.startswith("namespace: ") and current_id in valid_go and valid_go[current_id] != "obsolete":
                    ns: str = line.split("namespace: ")[1].strip()
                    valid_go[current_id] = {"cellular_component": "C", "biological_process": "P",
                                            "molecular_function": "F"}.get(ns, "")
                elif line.startswith("is_a: ") and current_id in valid_go and valid_go[current_id] != "obsolete":
                    parent_part: str = line.split("is_a: ")[1].split()[0].strip()
                    parent_id: str = normalize_go_term(parent_part)
                    if parent_id and parent_id in valid_go:
                        go_parents[current_id].append(parent_id)
    except Exception as e1:
        raise RuntimeError(f"❌ GO本体解析失败：{str(e1)}") from e1
    valid_go = {t: a for t, a in valid_go.items() if a in ["C", "P", "F"]}
    valid_go_terms: Set[str] = set(valid_go.keys())
    print(f"✅ 有效GO术语数：{len(valid_go_terms)}")

    # 2. 加载IA权重
    try:
        ia_df: pd.DataFrame = pd.read_csv(
            IA_TSV, sep="\t", header=None, names=["go_term", "ia_score"],
            on_bad_lines="skip", quoting=3
        )
    except Exception as e2:
        raise RuntimeError(f"❌ IA文件读取失败：{str(e2)}") from e2
    ia_dict: Dict[str, float] = {}
    for _, row in ia_df.iterrows():
        if pd.isna(row["go_term"]) or pd.isna(row["ia_score"]):
            continue
        term: str = normalize_go_term(str(row["go_term"]))
        if term in valid_go_terms:
            score_str: str = str(row["ia_score"])
            ia_dict[term] = float(score_str) if score_str.replace('.', '').isdigit() else IA_DEFAULT_VALUE
    for term in valid_go_terms:
        if term not in ia_dict:
            ia_dict[term] = IA_DEFAULT_VALUE
    high_ia_terms: Set[str] = {t for t, ia in ia_dict.items() if ia > HIGH_IA_THRESHOLD}
    print(f"✅ 高IA术语数：{len(high_ia_terms)}")

    # 3. 加载训练集序列（优化ID解析，添加调试）
    train_seq: Dict[str, str] = {}
    try:
        with open(TRAIN_FASTA, "r", encoding="utf-8-sig") as f:
            for record in SeqIO.parse(f, "fasta"):
                id_str: str = record.id.strip()
                # 优化ID解析：尝试多种格式（常见于UniProt、NCBI的ID格式）
                if "|" in id_str:
                    # 支持UniProt格式（如sp|P12345|... 取中间部分）
                    parts = id_str.split("|")
                    if len(parts) >= 2:
                        prot_id: str = parts[1].strip()
                    else:
                        prot_id = id_str.split()[0].strip()
                elif "." in id_str:
                    # 支持NCBI格式（如XP_0012345.1 去掉版本号）
                    prot_id = id_str.split(".")[0].strip()
                else:
                    prot_id = id_str.split()[0].strip()
                train_seq[prot_id] = str(record.seq)
    except Exception as e3:
        raise RuntimeError(f"❌ 训练集序列读取失败：{str(e3)}") from e3
    # 调试：打印训练序列的ID示例
    print(f"✅ 训练集序列数：{len(train_seq)}")
    if train_seq:
        sample_seq_ids = list(train_seq.keys())[:3]
        print(f"📌 训练序列ID示例：{sample_seq_ids}")
    else:
        raise RuntimeError("❌ 训练集序列为空！检查TRAIN_FASTA路径是否正确")

    # 4. 加载训练集标签（添加调试，检查ID匹配）
    try:
        # 尝试不同分隔符（有些文件是空格，有些是制表符）
        train_terms_df: pd.DataFrame = pd.read_csv(
            TRAIN_TERMS,
            sep=r'\t|\s+',  # 支持制表符或空格分隔
            header=0,
            names=["EntryID", "term", "aspect"],
            engine="python"
        )
    except Exception as e4:
        raise RuntimeError(f"❌ 训练标签读取失败：{str(e4)}") from e4
    # 调试：打印标签数据的基本信息
    print(f"✅ 原始训练标签数：{len(train_terms_df)}")
    if len(train_terms_df) == 0:
        raise RuntimeError("❌ 训练标签为空！检查TRAIN_TERMS路径是否正确")
    sample_entry_ids = train_terms_df["EntryID"].unique()[:3]
    print(f"📌 训练标签EntryID示例：{sample_entry_ids}")

    # 处理GO术语
    train_terms_df["term"] = train_terms_df["term"].apply(normalize_go_term)
    train_terms_df = train_terms_df[train_terms_df["term"].isin(valid_go_terms)]
    print(f"✅ 过滤后有效GO术语的标签数：{len(train_terms_df)}")
    if len(train_terms_df) == 0:
        raise RuntimeError("❌ 所有标签的GO术语无效！检查GO_OBO是否正确")

    # 过滤低频次术语
    term_counts: pd.Series = train_terms_df["term"].value_counts()
    valid_train_terms: Set[str] = set(term_counts[term_counts >= MIN_TRAIN_COUNT].index)
    train_terms_df = train_terms_df[train_terms_df["term"].isin(valid_train_terms)]
    print(f"✅ 过滤低频次术语后剩余标签数：{len(train_terms_df)}")
    if len(train_terms_df) == 0:
        raise RuntimeError(f"❌ 所有术语的出现次数均低于MIN_TRAIN_COUNT（{MIN_TRAIN_COUNT}）！请降低该参数")

    # 匹配蛋白ID（核心步骤，添加调试）
    prot_to_terms: Dict[str, List[str]] = defaultdict(list)
    # 提取标签中的所有蛋白ID
    all_entry_ids = set(train_terms_df["EntryID"].unique())
    # 计算匹配率
    matched_ids = all_entry_ids & set(train_seq.keys())
    match_rate = len(matched_ids) / len(all_entry_ids) if all_entry_ids else 0
    print(f"📌 训练标签与序列的ID匹配率：{match_rate:.2%}（{len(matched_ids)}/{len(all_entry_ids)}）")
    if match_rate < 0.1:
        print("⚠️ 匹配率过低！可能是ID格式不兼容，请检查以下示例：")
        print(f"  序列ID示例：{list(train_seq.keys())[:2]}")
        print(f"  标签EntryID示例：{list(all_entry_ids)[:2]}")

    # 构建蛋白-术语映射
    for _, row in train_terms_df.iterrows():
        prot_id: str = row["EntryID"].strip()
        if prot_id in train_seq:  # 只保留有序列的蛋白
            prot_to_terms[prot_id].append(row["term"])
    print(f"✅ 有效训练蛋白数（ID匹配且有标签）：{len(prot_to_terms)}")
    if not prot_to_terms:
        raise RuntimeError("❌ 无匹配的训练蛋白！请检查ID格式是否一致")

    # 5. 构建训练集特征矩阵
    print("\n===== 提取训练集特征 =====")
    x_train: List[np.ndarray] = []
    y_train: List[List[str]] = []
    for prot_id, terms in tqdm(prot_to_terms.items(), desc="特征提取"):
        seq: str = train_seq[prot_id]
        seq_feat: np.ndarray = extract_seq_features(seq)
        term_ia: List[float] = [ia_dict[t] for t in terms] if terms else [0.0]
        go_stats: np.ndarray = np.array([np.mean(term_ia), np.max(term_ia), len(terms)], dtype=np.float32)
        prot_feat: np.ndarray = np.hstack([seq_feat, go_stats])
        x_train.append(prot_feat)
        y_train.append(terms)
    if not x_train:
        raise ValueError("❌ 无有效训练特征！这通常是因为prot_to_terms为空")

    # 6. 多标签二值化
    mlb: MultiLabelBinarizer = MultiLabelBinarizer(classes=sorted(valid_train_terms))
    y_train_bin: np.ndarray = mlb.fit_transform(y_train)
    print(f"✅ 训练集特征维度：{len(x_train)} × {len(x_train[0])}")

    # 7. 训练模型
    print("\n===== 训练模型 =====")
    model: RandomForestClassifier = RandomForestClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        max_depth=30,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    try:
        cv_scores: np.ndarray = cross_val_score(model, x_train, y_train_bin, cv=VALIDATION_FOLDS, scoring="f1_macro")
    except Exception as e5:
        raise RuntimeError(f"❌ 模型交叉验证失败：{str(e5)}") from e5
    print(f"✅ 交叉验证F1：{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    model.fit(x_train, y_train_bin)

    # 8. 加载测试超集
    print("\n===== 解析测试超集 =====")
    test_prot_info: Dict[str, Dict[str, Union[int, str, np.ndarray]]] = {}
    try:
        with open(TEST_FASTA, "r", encoding="utf-8-sig") as f:
            for record in SeqIO.parse(f, "fasta"):
                id_str: str = record.id.strip()
                id_parts: List[str] = re.split(r'\s+', id_str)
                prot_id: str = id_parts[0] if id_parts else "unknown_prot"

                # 解析taxon_id
                taxon_id: int = 9606
                if len(id_parts) >= 2:
                    taxon_str: str = id_parts[1]
                    taxon_id = int(taxon_str) if taxon_str.isdigit() else 9606

                # 提取特征
                seq: str = str(record.seq)
                seq_feat: np.ndarray = extract_seq_features(seq)
                go_stats: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                prot_feat: np.ndarray = np.hstack([seq_feat, go_stats])

                test_prot_info[prot_id] = {
                    "taxon": taxon_id,
                    "seq": seq,
                    "feat": prot_feat
                }
    except Exception as e6:
        raise RuntimeError(f"❌ 测试集读取失败：{str(e6)}") from e6
    print(f"✅ 测试超集蛋白数：{len(test_prot_info)}")

    return {
        "valid_go": valid_go,
        "go_parents": go_parents,
        "ia_dict": ia_dict,
        "high_ia_terms": high_ia_terms,
        "model": model,
        "mlb": mlb,
        "test_prot_info": test_prot_info,
        "valid_go_terms": valid_go_terms
    }


# -------------------------- 4. 核心预测 --------------------------
def predict_batch_prots(args: Tuple[List[str], Dict[str, Any]]) -> List[List[str]]:
    prot_ids, shared_data = args
    batch_predictions: List[List[str]] = []

    model: RandomForestClassifier = shared_data["model"]
    mlb: MultiLabelBinarizer = shared_data["mlb"]
    go_parents: Dict[str, List[str]] = dict(shared_data["go_parents"])
    ia_dict: Dict[str, float] = dict(shared_data["ia_dict"])
    high_ia_terms: Set[str] = set(shared_data["high_ia_terms"])
    test_prot_info: Dict[str, Dict[str, Union[int, str, np.ndarray]]] = dict(shared_data["test_prot_info"])
    valid_go_terms: Set[str] = set(shared_data["valid_go_terms"])

    for prot_id in prot_ids:
        try:
            prot_data: Dict[str, Union[int, str, np.ndarray]] = test_prot_info[prot_id]

            # 处理特征
            prot_feat_raw: Union[np.ndarray, List[float]] = prot_data["feat"]
            prot_feat: np.ndarray = np.array(prot_feat_raw, dtype=np.float32) if isinstance(prot_feat_raw,
                                                                                            list) else prot_feat_raw
            prot_feat_reshaped: np.ndarray = prot_feat.reshape(1, -1)

            # 处理taxon_id
            taxon_raw: Union[int, str] = prot_data["taxon"]
            taxon_id: int = int(taxon_raw) if isinstance(taxon_raw, str) and taxon_raw.isdigit() else taxon_raw
            assert isinstance(taxon_id, int), f"taxon_id必须是int，实际是{type(taxon_id)}"

            # 模型预测
            y_pred_prob: np.ndarray = model.predict_proba(prot_feat_reshaped)[0]
            pred_terms: np.ndarray = mlb.classes_
            base_prob: Dict[str, float] = {}
            for term, prob in zip(pred_terms, y_pred_prob):
                term_str: str = str(term)
                if term_str in valid_go_terms and prob > PROB_THRESHOLD:
                    if term_str in high_ia_terms:
                        prob = min(0.999, prob * 1.15)
                    base_prob[term_str] = round(prob, 3)

            # 补充高IA术语
            missing_high_ia: Set[str] = high_ia_terms - set(base_prob.keys())
            if missing_high_ia:
                if taxon_id == 9606:
                    for term in missing_high_ia:
                        base_prob[term] = round(max(PROB_THRESHOLD + 0.1, ia_dict[term] / 4), 3)
                else:
                    for term in missing_high_ia:
                        base_prob[term] = round(max(PROB_THRESHOLD + 0.05, ia_dict[term] / 5), 3)

            # 术语传播
            propagated_prob: Dict[str, float] = base_prob.copy()
            stack: List[str] = list(base_prob.keys())
            while stack:
                child_term: str = stack.pop()
                child_prob: float = base_prob[child_term]
                for parent_term in go_parents.get(child_term, []):
                    if parent_term not in propagated_prob or child_prob > propagated_prob[parent_term]:
                        parent_prob: float = round(child_prob * 0.85, 3)
                        if parent_prob > PROB_THRESHOLD:
                            propagated_prob[parent_term] = parent_prob
                            stack.append(parent_term)

            # 筛选最终术语
            weighted_terms: List[Tuple[str, float]] = sorted(
                propagated_prob.items(),
                key=lambda x: x[1] * ia_dict.get(x[0], 0.0),
                reverse=True
            )
            final_terms: List[Tuple[str, float]] = weighted_terms[:MAX_TERMS_PER_PROT]

            # 收集结果
            for term, prob in final_terms:
                batch_predictions.append([prot_id, term, f"{prob:.3f}"])

        except Exception as e7:
            print(f"⚠️ 处理蛋白{prot_id}出错：{str(e7)}，跳过")
            continue

    return batch_predictions


# -------------------------- 5. 生成提交文件 --------------------------
def generate_high_score_submission(prep_data: Dict[str, Any]) -> None:
    print("\n=== 生成提交文件 ===")
    test_prot_info: Dict[str, Dict[str, Union[int, str, np.ndarray]]] = prep_data["test_prot_info"]
    prot_ids: List[str] = list(test_prot_info.keys())
    print(f"📌 待预测测试蛋白数：{len(prot_ids)}")

    from multiprocessing import Pool, Manager
    manager = Manager()
    shared_data: Dict[str, Any] = {
        "model": prep_data["model"],
        "mlb": prep_data["mlb"],
        "go_parents": manager.dict({k: v for k, v in prep_data["go_parents"].items()}),
        "ia_dict": manager.dict({k: v for k, v in prep_data["ia_dict"].items()}),
        "high_ia_terms": manager.list(prep_data["high_ia_terms"]),
        "test_prot_info": manager.dict({
            k: {
                "taxon": v["taxon"],
                "seq": v["seq"],
                "feat": v["feat"].tolist()
            } for k, v in test_prot_info.items()
        }),
        "valid_go_terms": manager.list(prep_data["valid_go_terms"])
    }

    args_list: List[Tuple[List[str], Dict[str, Any]]] = [
        (prot_ids[i:i + BATCH_SIZE], shared_data)
        for i in range(0, len(prot_ids), BATCH_SIZE)
    ]
    temp_files: List[str] = []
    with Pool(processes=PROCESS_NUM) as pool:
        for batch_result in tqdm(
                pool.imap(predict_batch_prots, args_list),
                total=len(args_list),
                desc="预测进度"
        ):
            if not batch_result:
                continue
            with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
                for pred in batch_result:
                    f.write(f"{pred[0]}\t{pred[1]}\t{pred[2]}\n")
                temp_files.append(f.name)

    os.makedirs(os.path.dirname(OUTPUT_SUBMISSION), exist_ok=True)
    with open(OUTPUT_SUBMISSION, "w", encoding="utf-8") as out_f:
        for temp_path in temp_files:
            with open(temp_path, "r", encoding="utf-8") as temp_f:
                out_f.write(temp_f.read())
            os.remove(temp_path)

    sub_df: pd.DataFrame = pd.read_csv(
        OUTPUT_SUBMISSION, sep="\t", header=None, names=["prot_id", "go_term", "prob"]
    )
    high_ia_count: int = sum(sub_df["go_term"].isin(prep_data["high_ia_terms"]))
    print(f"✅ 提交记录总数：{len(sub_df)}")
    print(f"✅ 高IA术语占比：{high_ia_count / len(sub_df):.2%}")


# -------------------------- 6. 主函数 --------------------------
def main() -> None:
    print("=" * 50)
    print("📌 CAFA竞赛修复版代码（解决训练集匹配问题）")
    print("=" * 50)
    print("⚠️ 依赖检查：请确保已安装：")
    print("   pip install biopython pandas numpy scikit-learn tqdm")
    print("=" * 50)

    try:
        prep_data: Dict[str, Any] = load_and_preprocess()
        generate_high_score_submission(prep_data)
        print("\n🎉 全流程完成！")
    except Exception as e_main:
        print(f"\n❌ 流程失败：{str(e_main)}")
        sys.exit(1)


if __name__ == "__main__":
    main()