# -*- coding: utf-8 -*-
"""
utils/db_helper.py
==================
本地化合物知识库（SQLite）。

Day 9-10 任务：
    - 初始化 data/compounds.db，建立 compounds 表。
    - query_property(name) 按名称查询性质。
    - populate_database() 通过 PubChem 批量抓取 20 个常见化合物入库。

数据来源：
    - 核心字段（SMILES/分子量/分子式/XLogP）：PubChem PUG REST property 端点（可靠）。
    - 实验字段（熔点/沸点/密度/CAS）：PubChem PUG-View 全记录（best-effort，
      PUG-View 不稳定或字段缺失时填 NULL，符合 AGENT.md「忽略缺失非关键字段」）。
"""

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# 项目根目录 / data 目录
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 允许 `python utils/db_helper.py` 直接运行时导入 utils 包内兄弟模块
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.curated_data import get_experimental

_DATA_DIR = _PROJECT_ROOT / "data"
_DEFAULT_DB = _DATA_DIR / "compounds.db"

# AGENT.md 指定的 20 个常见化合物：中文名 -> PubChem 查询用英文名
COMPOUNDS = {
    "甲烷": "methane",
    "乙醇": "ethanol",
    "苯": "benzene",
    "甲苯": "toluene",
    "苯酚": "phenol",
    "环己烷": "cyclohexane",
    "乙酸": "acetic acid",
    "丙酮": "acetone",
    "乙醚": "diethyl ether",
    "氯仿": "chloroform",
    "四氯化碳": "carbon tetrachloride",
    "乙二醇": "ethylene glycol",
    "甘油": "glycerol",
    "苯胺": "aniline",
    "硝基苯": "nitrobenzene",
    "苯甲酸": "benzoic acid",
    "水杨酸": "salicylic acid",
    "阿司匹林": "aspirin",
    "对乙酰氨基酚": "acetaminophen",
    "咖啡因": "caffeine",
}

# PubChem PUG REST：按名称取计算属性（可靠）
_PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/CanonicalSMILES,MolecularWeight,MolecularFormula,IUPACName,XLogP/JSON"
)
# PubChem PUG-View：按 CID 取实验数据（best-effort，不稳定时整段跳过）
_PUGVIEW_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/cid/{cid}/JSON"

_PUBCHEM_TIMEOUT = 20
_RATE_LIMIT_SEC = 0.35  # PubChem 礼貌限速（请求/秒 < 5）


# --------------------------------------------------------------------------- #
# 数据库初始化
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS compounds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT UNIQUE NOT NULL,
    en_name          TEXT,
    iupac_name       TEXT,
    smiles           TEXT,
    molecular_formula TEXT,
    mol_weight       REAL,
    xlogp            REAL,
    melting_point    TEXT,
    boiling_point    TEXT,
    density          TEXT,
    cas              TEXT,
    cid              INTEGER,
    updated_at       TEXT
);
"""


def init_db(db_path: Path = _DEFAULT_DB) -> Path:
    """创建 data 目录与 compounds 表（若已存在则跳过）。返回 db 路径。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()
    print(f"[init_db] 数据库就绪: {db_path}")
    return db_path


def _connect(db_path: Path = _DEFAULT_DB):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 查询结果按列名取值
    return conn


# --------------------------------------------------------------------------- #
# 查询接口
# --------------------------------------------------------------------------- #
def query_property(name: str, db_path: Path = _DEFAULT_DB):
    """按名称查询化合物性质（大小写、首尾空格不敏感）。

    返回:
        dict | None: 命中时返回各字段键值（含 smiles/mol_weight/melting_point 等）；
                     未命中或库不存在返回 None。
    """
    if not name:
        return None
    if not Path(db_path).exists():
        print(f"[query_property] 数据库不存在: {db_path}，请先运行 populate_database()。")
        return None

    with _connect(db_path) as conn:
        target = name.strip()
        row = conn.execute(
            "SELECT * FROM compounds "
            "WHERE LOWER(name) = LOWER(?) OR LOWER(en_name) = LOWER(?) "
            "OR LOWER(iupac_name) = LOWER(?)",
            (target, target, target),
        ).fetchone()
    if row is None:
        print(f"[query_property] 未查到化合物: {name!r}")
        return None
    print(f"[query_property] 命中: {name!r} -> {row['molecular_formula']} (MW={row['mol_weight']})")
    return dict(row)


# --------------------------------------------------------------------------- #
# PubChem 抓取
# --------------------------------------------------------------------------- #
def _fetch_core_properties(en_name: str):
    """PUG REST 取核心计算属性（可靠）。失败返回 None。"""
    url = _PUBCHEM_PROPERTY_URL.format(name=requests.utils.quote(en_name))
    try:
        resp = requests.get(url, timeout=_PUBCHEM_TIMEOUT)
    except requests.exceptions.RequestException as e:
        print(f"[_fetch_core_properties] {en_name!r} 网络异常: {e}")
        return None
    if resp.status_code == 404:
        print(f"[_fetch_core_properties] PubChem 未收录: {en_name!r}")
        return None
    if resp.status_code != 200:
        print(f"[_fetch_core_properties] {en_name!r} HTTP {resp.status_code}")
        return None
    try:
        props = resp.json()["PropertyTable"]["Properties"][0]
    except (ValueError, KeyError, IndexError):
        print(f"[_fetch_core_properties] {en_name!r} 响应解析失败")
        return None
    # SMILES 字段名存在 CanonicalSMILES / ConnectivitySMILES 两种返回，统一兜底
    smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
    return {
        "cid": props.get("CID"),
        "smiles": smiles,
        "molecular_formula": props.get("MolecularFormula"),
        "mol_weight": _to_float(props.get("MolecularWeight")),
        "iupac_name": props.get("IUPACName"),
        "xlogp": _to_float(props.get("XLogP")),
    }


def _fetch_experimental(cid):
    """PUG-View 按 CID 取实验数据（best-effort）。任何失败返回空 dict。"""
    if not cid:
        return {}
    url = _PUGVIEW_URL.format(cid=cid)
    try:
        resp = requests.get(url, timeout=_PUBCHEM_TIMEOUT)
    except requests.exceptions.RequestException:
        return {}
    if resp.status_code != 200:
        return {}
    try:
        sections = resp.json().get("Record", {}).get("Section", [])
    except ValueError:
        return {}
    return {
        "melting_point": _extract_section(sections, "Melting Point"),
        "boiling_point": _extract_section(sections, "Boiling Point"),
        "density": _extract_section(sections, "Density"),
        "cas": _extract_section(sections, "CAS"),
    }


def _extract_section(sections, heading):
    """递归在 PUG-View 的 Section 树中查找 TOCHeading 匹配 heading 的小节，
    提取首个 Information 的值。PUG-View 结构高度嵌套，未命中返回 ""。"""
    for sec in sections or []:
        if heading.lower() in str(sec.get("TOCHeading", "")).lower():
            extracted = _extract_first_value(sec.get("Information", []))
            if extracted:
                return extracted
        # 递归子节点（命中后即返回，避免被父节点的空值覆盖）
        found = _extract_section(sec.get("Section", []), heading)
        if found:
            return found
    return ""


def _extract_first_value(information_list):
    """从 PUG-View 的 Information 条目中提取首个值（兼容多种值格式）。"""
    for info in information_list or []:
        value = info.get("Value", {})
        if not isinstance(value, dict):
            continue
        # StringWithExp: [{"Value": "122 °C", ...}]
        swe = value.get("StringWithExp")
        if swe and isinstance(swe, list) and swe[0].get("Value"):
            return str(swe[0]["Value"]).strip()
        # Number: 1.234
        if "Number" in value and value["Number"] is not None:
            return str(value["Number"])
        # String: "..." 或 ["..."]
        s = value.get("String")
        if isinstance(s, list) and s:
            return str(s[0]).strip()
        if isinstance(s, str) and s:
            return s.strip()
    return ""


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# 批量填充
# --------------------------------------------------------------------------- #
def populate_database(db_path: Path = _DEFAULT_DB, compounds=None):
    """批量抓取化合物数据并写入数据库。

    参数:
        db_path: 数据库路径。
        compounds: dict {中文名: 英文查询名}，默认 COMPOUNDS（20 个常见化合物）。

    返回:
        int: 成功入库的化合物数量。
    """
    compounds = compounds or COMPOUNDS
    init_db(db_path)
    total = len(compounds)
    success = 0

    print(f"[populate_database] 开始抓取 {total} 个化合物 ...")
    for cn_name, en_name in compounds.items():
        print(f"\n[populate_database] ({en_name}) {cn_name} ...")
        core = _fetch_core_properties(en_name)
        if core is None:
            print(f"[populate_database] {cn_name} 核心数据获取失败，跳过。")
            time.sleep(_RATE_LIMIT_SEC)
            continue

        # 实验数据：curated 优先（可靠），PUG-View 作为库外化合物 best-effort 回退
        experimental = get_experimental(en_name) or _fetch_experimental(core.get("cid")) or {}
        if experimental:
            print(f"[populate_database] 实验数据: 熔点={experimental.get('melting_point') or 'NULL'}"
                  f" 密度={experimental.get('density') or 'NULL'}")

        _upsert(db_path, {
            "name": cn_name,
            "en_name": en_name,
            "iupac_name": core.get("iupac_name"),
            "smiles": core.get("smiles"),
            "molecular_formula": core.get("molecular_formula"),
            "mol_weight": core.get("mol_weight"),
            "xlogp": core.get("xlogp"),
            "melting_point": experimental.get("melting_point") or None,
            "boiling_point": experimental.get("boiling_point") or None,
            "density": experimental.get("density") or None,
            "cas": experimental.get("cas") or None,
            "cid": core.get("cid"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        success += 1
        time.sleep(_RATE_LIMIT_SEC)

    print(f"\n[populate_database] 完成: {success}/{total} 入库。")
    return success


def _upsert(db_path: Path, row: dict):
    """插入或替换（按 name 唯一约束）一条化合物记录。"""
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO compounds
                (name, en_name, iupac_name, smiles, molecular_formula, mol_weight,
                 xlogp, melting_point, boiling_point, density, cas, cid, updated_at)
            VALUES
                (:name, :en_name, :iupac_name, :smiles, :molecular_formula, :mol_weight,
                 :xlogp, :melting_point, :boiling_point, :density, :cas, :cid, :updated_at)
            """,
            row,
        )
        conn.commit()
    print(f"[populate_database] 已入库: {row['name']} ({row['molecular_formula']}, MW={row['mol_weight']})")


def update_experimental(db_path: Path = _DEFAULT_DB):
    """用 curated 数据更新现有 DB 记录的实验字段（不重新抓取核心数据）。

    用于已 populate 过的库补充实验数据，避免重打 PubChem。仅更新 curated 命中的化合物。
    返回更新的记录数。
    """
    init_db(db_path)
    updated = 0
    with _connect(db_path) as conn:
        for row in conn.execute("SELECT id, en_name, name FROM compounds"):
            exp = get_experimental(row["en_name"])
            if not exp:
                continue
            conn.execute(
                "UPDATE compounds SET melting_point=?, boiling_point=?, density=?, cas=?, updated_at=? "
                "WHERE id=?",
                (exp.get("melting_point"), exp.get("boiling_point"),
                 exp.get("density"), exp.get("cas"),
                 datetime.now().isoformat(timespec="seconds"), row["id"]),
            )
            updated += 1
            print(f"[update_experimental] {row['name']}: mp={exp.get('melting_point')} "
                  f"bp={exp.get('boiling_point')} d={exp.get('density')} CAS={exp.get('cas')}")
        conn.commit()
    print(f"[update_experimental] 共更新 {updated} 条记录的实验字段。")
    return updated


# --------------------------------------------------------------------------- #
# 测试入口
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("db_helper: 初始化 + 批量填充 + 查询测试")
    print("=" * 60)

    db = init_db()

    if "--refresh-exp" in sys.argv:
        # 刷新实验字段：python db_helper.py --refresh-exp
        print("\n[模式] 用 curated 数据刷新实验字段 ...")
        update_experimental(db)
        print("\n[查询测试] 阿司匹林")
        result = query_property("阿司匹林", db)
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
    elif "--query" in sys.argv:
        # 查询模式：python db_helper.py --query 阿司匹林
        idx = sys.argv.index("--query")
        target = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "阿司匹林"
        print(f"\n[查询测试] {target}")
        result = query_property(target, db)
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
    else:
        # 默认：批量填充
        populate_database(db)
        print("\n[查询测试] 阿司匹林")
        result = query_property("阿司匹林", db)
        if result:
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print("  （未查到，请检查填充是否成功）")

    print("=" * 60)
    print("测试结束。")
