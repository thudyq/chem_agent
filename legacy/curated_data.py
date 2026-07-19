# -*- coding: utf-8 -*-
"""
legacy/curated_data.py
      =====================
20 个常见化合物的实验数据（curated 参考值）。

用途（UPGRADE 1.1）：
    PUG-View 实验数据接口不稳定，20 个库内化合物改用 curated 数据作为主数据源，
    PUG-View 降级为库外化合物的 best-effort 回退。

数据说明：
    - CAS 号：稳定注册号，准确。
    - melting_point / boiling_point：标准大气压下的参考值（°C），取常用教科书数值。
    - density：液态/固态密度（g/cm³，25°C 附近）；气态化合物标注为气态密度。
    - 会分解 / 升华的化合物，boiling_point 标注为"分解"或"升华"，不填虚假数值。
    - 所有值为参考值，如需 NIST 级精度请核对权威数据库。
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

# 键 = en_name（与 db_helper.COMPOUNDS 的英文查询名一致）
CURATED_EXPERIMENTAL = {
    "methane": {
        "melting_point": "-182.5",
        "boiling_point": "-161.5",
        "density": "0.657 g/L（气态）",
        "cas": "74-82-8",
    },
    "ethanol": {
        "melting_point": "-114.0",
        "boiling_point": "78.4",
        "density": "0.789",
        "cas": "64-17-5",
    },
    "benzene": {
        "melting_point": "5.5",
        "boiling_point": "80.1",
        "density": "0.879",
        "cas": "71-43-2",
    },
    "toluene": {
        "melting_point": "-95.0",
        "boiling_point": "110.6",
        "density": "0.867",
        "cas": "108-88-3",
    },
    "phenol": {
        "melting_point": "40.5",
        "boiling_point": "181.7",
        "density": "1.07",
        "cas": "108-95-2",
    },
    "cyclohexane": {
        "melting_point": "6.5",
        "boiling_point": "80.7",
        "density": "0.778",
        "cas": "110-82-7",
    },
    "acetic acid": {
        "melting_point": "16.6",
        "boiling_point": "118.1",
        "density": "1.049",
        "cas": "64-19-7",
    },
    "acetone": {
        "melting_point": "-94.7",
        "boiling_point": "56.0",
        "density": "0.791",
        "cas": "67-64-1",
    },
    "diethyl ether": {
        "melting_point": "-116.3",
        "boiling_point": "34.6",
        "density": "0.708",
        "cas": "60-29-7",
    },
    "chloroform": {
        "melting_point": "-63.5",
        "boiling_point": "61.2",
        "density": "1.489",
        "cas": "67-66-3",
    },
    "carbon tetrachloride": {
        "melting_point": "-22.9",
        "boiling_point": "76.7",
        "density": "1.594",
        "cas": "56-23-5",
    },
    "ethylene glycol": {
        "melting_point": "-12.9",
        "boiling_point": "197.3",
        "density": "1.112",
        "cas": "107-21-1",
    },
    "glycerol": {
        "melting_point": "17.8",
        "boiling_point": "290",
        "density": "1.261",
        "cas": "56-81-5",
    },
    "aniline": {
        "melting_point": "-6.3",
        "boiling_point": "184.1",
        "density": "1.022",
        "cas": "62-53-3",
    },
    "nitrobenzene": {
        "melting_point": "5.7",
        "boiling_point": "210.9",
        "density": "1.204",
        "cas": "98-95-3",
    },
    "benzoic acid": {
        "melting_point": "122.1",
        "boiling_point": "249",
        "density": "1.266",
        "cas": "65-85-0",
    },
    "salicylic acid": {
        "melting_point": "159",
        "boiling_point": "211（分解）",
        "density": "1.443",
        "cas": "69-72-7",
    },
    "aspirin": {
        "melting_point": "135",
        "boiling_point": "分解",
        "density": "1.40",
        "cas": "50-78-2",
    },
    "acetaminophen": {
        "melting_point": "169",
        "boiling_point": "分解",
        "density": "1.293",
        "cas": "103-90-2",
    },
    "caffeine": {
        "melting_point": "235",
        "boiling_point": "升华",
        "density": "1.23",
        "cas": "58-08-2",
    },
}


def get_experimental(en_name: str):
    """按 en_name 查 curated 实验数据。

    返回:
        dict | None: {melting_point, boiling_point, density, cas}；未收录返回 None。
    """
    return CURATED_EXPERIMENTAL.get(en_name)
