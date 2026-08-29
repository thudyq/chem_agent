# -*- coding: utf-8 -*-
"""core/electron_sim.py — 电子推动模拟器（P1，Drawbacks §16.2/§16.3）。

给定一步反应的"反应物组件 + 机理弯箭头"，确定性模拟电子流（RWMol 图
操作），把推出的产物与声明的产物做图同构比对——验证机理图的**自洽性**
（箭头 ⇄ 产物一致），不评判机理选择的化学真理（超出确定性方法范围）。

关键设计（20260828）：
- 形式电荷**不做增量簿记**——改为跟踪孤对电子对数/自由基单电子数，
  末尾按"族价电子 − 非键电子 − 键级和"公式对**被触碰的原子**重算
  （增量簿记在"得键 + 失键"协同场景如 SN2 的碳上必错）；未触碰原子
  保持原电荷（规避罕见元素的价态模型边界）；
- 反应物先合并为一张不相交大图（CombineMols），箭头按全局序号操作，
  末尾按连通分量分裂（GetMolFrags）为产物分子；显式 H 是真实原子
  （管线解析口径），质子转移 = H 原子在组分间移动；
- 两遍式应用：先削键/断键（避免 H 转移等中间态超价），再成键；
- π 进攻（键 → 他分子原子）的新键端位（π 键两端碳选谁）有真歧义——
  两个变体都试，任一与声明产物一致即通过（≤2 根歧义箭头，超出跳过）；
- 错侧判定（§16.3）：模拟产物连价态/消毒都过不了 → 箭头必错（确定）；
  不一致 → 默认信产物修箭头（产物已过守恒/价态/label 独立检查），
  报错附模拟预期值；"修箭头不收敛/产物不可达 → 翻修产物"归 P1.5
  （需图 diff 分析），本版只在消息中提示。

跳过面（宁漏勿拦）：无箭头步、化学式文本组件、含 dummy 原子（R/Ph）
的组件参与的步、BLOCK 内外混合引用、不认识的箭头形态。
"""

import re

from rdkit import Chem

from renderers.mol_primitives import _VALENCE_ELECTRONS, lone_pair_count

# 箭头 spec 解析（与 tag_validator._MECH_ARROW_RE 同构，独立成一份避免
# 模块循环依赖）
_PT = r"(?:\d+|\d+-\d+)"
_ARROW_SPEC_RE = re.compile(
    rf"^\s*([A-Za-z0-9_]+)\s*:\s*({_PT})\s*(>>|>)\s*"
    rf"([A-Za-z0-9_]+)\s*:\s*({_PT})"
    r"(?:\s*\+\s*([A-Za-z0-9_]+)\s*:\s*(\d+))?\s*$")

_BOND_UP = {1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}
_BOND_DOWN = {2: Chem.BondType.SINGLE, 3: Chem.BondType.DOUBLE}

_MAX_VARIANTS = 4          # π 进攻歧义组合上限（2 根歧义箭头），超出跳过该步
_FREE_PROTONS = {"[H+]", "[H]", "[H-]"}   # 脱质子副产惯例（允许预告多出）


class _Sim:
    """一步反应的电子流模拟状态（不相交大图 + 每原子非键电子簿记）。"""

    def __init__(self, cids_mols, variant: dict = None):
        self.offsets = {}
        combo = None
        for cid, mol in cids_mols:
            self.offsets[cid] = combo.GetNumAtoms() if combo is not None else 0
            combo = mol if combo is None else Chem.CombineMols(combo, mol)
        self.rw = Chem.RWMol(combo) if combo is not None else Chem.RWMol()
        n = self.rw.GetNumAtoms()
        self.pairs = [0] * n       # 孤对电子对数
        self.singles = [0] * n     # 自由基单电子数
        # 隐含 H 数在操作前冻结（键级变化不改骨架氢数；RDKit 的 GetTotalNumHs
        # 依赖形式电荷，改电荷后再取会循环依赖）
        self.implicit_hs = [a.GetTotalNumHs() for a in self.rw.GetAtoms()]
        for cid, mol in cids_mols:
            for a in mol.GetAtoms():
                p, s = lone_pair_count(a)
                gi = self.offsets[cid] + a.GetIdx()
                self.pairs[gi], self.singles[gi] = p, s
        self.touched = set()       # 被任一操作触及的原子（末尾重算电荷）
        self.trace = []            # 操作日志（trace 模式/replay 展示）
        self.impossible = None     # 第一个不可能的价态/电子操作原因
        self._variant = variant or {}

    # ---- 端点换算 ----
    def _gi(self, cid, pt):
        return self.offsets[cid] + int(pt)

    def _bond(self, cid, pt):
        a, _, b = pt.partition("-")
        return self.rw.GetBondBetweenAtoms(self._gi(cid, a), self._gi(cid, b))

    # ---- 图操作 ----
    def _reduce_bond(self, bd, what):
        order = int(round(bd.GetBondTypeAsDouble()))
        if order <= 1:
            self.rw.RemoveBond(bd.GetBeginAtomIdx(), bd.GetEndAtomIdx())
            self.trace.append(f"断键 {what}（σ 键断裂）")
        else:
            bd.SetBondType(_BOND_DOWN[order])
            self.trace.append(f"削弱 {what}（键级 {order}→{order - 1}）")

    def _raise_bond(self, bd, what):
        order = int(round(bd.GetBondTypeAsDouble()))
        if order >= 3:
            self.impossible = self.impossible or f"键 {what} 已是三键无法再加级"
            return
        bd.SetBondType(_BOND_UP[order])
        self.trace.append(f"增强 {what}（键级 {order}→{order + 1}）")

    def _form_bond(self, gi, gj, what):
        if self.rw.GetBondBetweenAtoms(gi, gj) is not None:
            self._raise_bond(self.rw.GetBondBetweenAtoms(gi, gj), what)
            return
        self.rw.AddBond(gi, gj, Chem.BondType.SINGLE)
        self.trace.append(f"成键 {what}")

    # ---- 双电子箭头 ----
    def apply_pair_reduce(self, spec):
        """第一遍：双电子箭头的键削弱/断键侧。"""
        m = _ARROW_SPEC_RE.match(spec)
        if not m or m.group(3) != ">":
            return
        src_id, src_pt, dst_id, dst_pt = m.group(1), m.group(2), \
            m.group(4), m.group(5)
        if "-" not in src_pt:
            return   # 孤对起点无削弱侧
        bd = self._bond(src_id, src_pt)
        if bd is None:
            self.impossible = self.impossible or \
                f"{src_id}:{src_pt} 引用的键不存在"
            return
        self.touched.add(bd.GetBeginAtomIdx())
        self.touched.add(bd.GetEndAtomIdx())
        if "-" in dst_pt:
            # 键 → 键（π 移位/落回闭环键）：起点键削弱，终点键增强在第二遍
            self._reduce_bond(bd, f"{src_id}:{src_pt}")
            return
        dst_gi = self._gi(dst_id, dst_pt)
        if dst_gi in (bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()):
            # 键 → 键端原子之一（断键，电子对归该原子）：σ 断键时另一端的
            # 形式电荷由末尾公式重算（失键）；π 削弱对端不动 σ 骨架
            self.pairs[dst_gi] += 1
            self.touched.add(dst_gi)
            self._reduce_bond(bd, f"{src_id}:{src_pt}→{dst_id}:{dst_pt}")
            self.trace.append(f"电子对落在 {dst_id}:{dst_pt}")
        else:
            # 键 → 他分子原子（π/σ 进攻）：削弱源键，新键在第二遍形成
            self._reduce_bond(bd, f"{src_id}:{src_pt}（进攻 {dst_id}:{dst_pt}）")

    def apply_pair_form(self, spec):
        """第二遍：双电子箭头的成键/增级侧。"""
        m = _ARROW_SPEC_RE.match(spec)
        if not m or m.group(3) != ">":
            return
        src_id, src_pt = m.group(1), m.group(2)
        dst_id, dst_pt, dst2_id, dst2_pt = m.group(4), m.group(5), \
            m.group(6), m.group(7)
        if "-" not in src_pt:
            # 孤对起点：供体少一对孤对
            src_gi = self._gi(src_id, src_pt)
            if self.pairs[src_gi] <= 0:
                self.impossible = self.impossible or \
                    f"{src_id}:{src_pt} 没有孤对电子可供捐赠"
                return
            self.pairs[src_gi] -= 1
            self.touched.add(src_gi)
        if dst2_id is not None:
            # 成键空白位：双电子汇聚成新键
            self._form_bond(self._gi(dst_id, dst_pt),
                            self._gi(dst2_id, dst2_pt),
                            f"{dst_id}:{dst_pt}+{dst2_id}:{dst2_pt}")
            self.touched.add(self._gi(dst_id, dst_pt))
            self.touched.add(self._gi(dst2_id, dst2_pt))
            return
        if "-" in dst_pt:
            # 键 → 键：终点键增强（σ→π，恢复芳香性等）
            bd = self._bond(dst_id, dst_pt)
            if bd is None:
                self.impossible = self.impossible or \
                    f"{dst_id}:{dst_pt} 引用的键不存在"
                return
            self.touched.add(bd.GetBeginAtomIdx())
            self.touched.add(bd.GetEndAtomIdx())
            self._raise_bond(bd, f"{dst_id}:{dst_pt}")
            return
        dst_gi = self._gi(dst_id, dst_pt)
        if "-" in src_pt:
            # π/σ 进攻与氢迁移：源键电子对形成新键（端位歧义由变体机制处理）。
            # 锚点从 spec 取——第一遍若把单键整条断掉（氢迁移的 C—H），
            # 此时键对象已不存在，不能依赖 _bond 回查
            a_loc, _, b_loc = src_pt.partition("-")
            ga, gb = self._gi(src_id, a_loc), self._gi(src_id, b_loc)
            if dst_gi in (ga, gb):
                return   # 断键型（电子对归键端原子）第一遍已完成，不再成键
            pick = self._variant.get(spec, 0)
            anchor = ga if pick == 0 else gb
            self._form_bond(anchor, dst_gi,
                            f"{src_id}:{src_pt}→{dst_id}:{dst_pt}"
                            f"（锚定 {src_id}:{anchor - self.offsets[src_id]}）")
            # 未锚定的一端 σ 骨架上少了一根 π 键——电荷末尾重算
            self.touched.add(ga)
            self.touched.add(gb)
            self.touched.add(dst_gi)
            return
        # 孤对 → 原子：新键供体—受体
        self._form_bond(self._gi(src_id, src_pt), dst_gi,
                        f"{src_id}:{src_pt}→{dst_id}:{dst_pt}")
        self.touched.add(self._gi(src_id, src_pt))
        self.touched.add(dst_gi)

    # ---- 鱼钩（单电子）----
    def apply_hook_count(self, spec, bond_hooks, vacancy_hooks):
        """第一遍：鱼钩计数（源键收集电子数）。"""
        m = _ARROW_SPEC_RE.match(spec)
        if not m or m.group(3) != ">>":
            return
        src_id, src_pt = m.group(1), m.group(2)
        dst_id, dst_pt = m.group(4), m.group(5)
        dst2_id, dst2_pt = m.group(6), m.group(7)
        if "-" in src_pt:
            bd = self._bond(src_id, src_pt)
            if bd is None:
                self.impossible = self.impossible or \
                    f"{src_id}:{src_pt} 引用的键不存在"
                return
            bond_hooks.setdefault((src_id, src_pt), []).append(bd)
            self.touched.add(bd.GetBeginAtomIdx())
            self.touched.add(bd.GetEndAtomIdx())
        else:
            src_gi = self._gi(src_id, src_pt)
            if self.singles[src_gi] <= 0:
                self.impossible = self.impossible or \
                    f"{src_id}:{src_pt} 没有单电子（鱼钩起点须是自由基）"
                return
            self.singles[src_gi] -= 1
            self.touched.add(src_gi)
            self.trace.append(f"单电子自 {src_id}:{src_pt} 出发")
        if dst2_id is not None:
            vacancy_hooks.setdefault(
                (dst_id, dst_pt, dst2_id, dst2_pt), 0)
            vacancy_hooks[(dst_id, dst_pt, dst2_id, dst2_pt)] += 1
        else:
            dst_gi = self._gi(dst_id, dst_pt)
            self.singles[dst_gi] += 1
            self.touched.add(dst_gi)
            self.trace.append(f"单电子落在 {dst_id}:{dst_pt}")

    def apply_hook_form(self, bond_hooks, vacancy_hooks):
        """第二遍：均裂断键（收齐 2 个电子）与空白位汇聚成键。"""
        for (cid, pt), bds in bond_hooks.items():
            for bd in bds[:1]:    # 同一键只需断一次
                if self.rw.GetBondBetweenAtoms(bd.GetBeginAtomIdx(),
                                               bd.GetEndAtomIdx()) is None:
                    continue
                self._reduce_bond(bd, f"{cid}:{pt}（均裂）")
        for (d1, p1, d2, p2), cnt in vacancy_hooks.items():
            if cnt >= 2:
                self._form_bond(self._gi(d1, p1), self._gi(d2, p2),
                                f"{d1}:{p1}+{d2}:{p2}（单电子汇聚）")
                self.touched.add(self._gi(d1, p1))
                self.touched.add(self._gi(d2, p2))

    # ---- 收尾 ----
    def finalize(self):
        """被触碰原子按 族价电子−非键电子−（键级和+冻结的隐含 H 数） 重算
        形式电荷，并把簿记的单电子数写回原子。二周期原子（B/C/N/O/F）键级和
        + 孤对数 > 4 = 超价（缺补偿箭头，如进攻没画断键）——判箭头不可能。"""
        for gi in self.touched:
            a = self.rw.GetAtomWithIdx(gi)
            a.SetNumRadicalElectrons(self.singles[gi])
            ve = _VALENCE_ELECTRONS.get(a.GetAtomicNum())
            if ve is None:
                continue   # 金属等不在表——保持原电荷
            bonds = sum(int(round(b.GetBondTypeAsDouble()))
                        for b in a.GetBonds()) + self.implicit_hs[gi]
            a.SetFormalCharge(ve - 2 * self.pairs[gi]
                              - self.singles[gi] - bonds)
            if a.GetAtomicNum() in (5, 6, 7, 8, 9) \
                    and bonds + self.pairs[gi] > 4:
                self.impossible = self.impossible or (
                    f"原子 {gi}（{a.GetSymbol()}）超八隅体"
                    f"（{bonds} 键级 + {self.pairs[gi]} 孤对——"
                    f"进攻/成键后缺配套的断键箭头）")

    def fragment(self):
        """分裂为产物分子列表；任一碎片消毒不过 → 返回 (None, 原因)。"""
        mol = self.rw.GetMol()
        mol.UpdatePropertyCache(strict=False)
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        out = []
        for f in frags:
            f.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(f)
            except Exception as e:
                return None, f"产生不可能的价态结构（{e}）"
            out.append(f)
        return out, None


def _parse_specs(mech_children) -> list:
    """接受 RenderTag（MECHARROW 子标记）或已解析的 spec 字符串。"""
    specs = []
    for child in mech_children:
        if isinstance(child, str):
            specs.append(child)
            continue
        if not child.args or not child.args[0]:
            continue
        specs.extend(s.strip() for s in child.args[0].split(",") if s.strip())
    return specs


def _ambiguous_variants(specs, mol_by_cid) -> list:
    """π/σ 进攻箭头（源为键、终点为他分子原子）的锚定端位歧义：
    每个返回两个变体。断键型（终点是源键端原子之一）不在歧义之列。
    返回变体组合列表（{spec: pick}）；歧义过多返回 []（跳过该步）。"""
    amb = []
    for spec in specs:
        m = _ARROW_SPEC_RE.match(spec)
        if not m or m.group(3) != ">":
            continue
        src_id, src_pt, dst_id, dst_pt = (m.group(1), m.group(2),
                                          m.group(4), m.group(5))
        if "-" not in src_pt or "-" in dst_pt:
            continue
        mol = mol_by_cid.get(src_id)
        if mol is None:
            continue
        a, _, b = src_pt.partition("-")
        bd = mol.GetBondBetweenAtoms(int(a), int(b))
        if bd is None:
            continue
        if int(dst_pt) in (bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()):
            continue   # 断键型（电子对归键端原子），非进攻歧义
        amb.append(spec)
    if not amb:
        return [{}]
    if len(amb) > 2:
        return []        # 歧义过多——超出教学场景，跳过该步
    combos = []
    for mask in range(1 << len(amb)):
        combos.append({spec: (mask >> i) & 1 for i, spec in enumerate(amb)})
    return combos


def simulate(cids_mols, mech_children) -> list:
    """模拟一步：返回 [(frags 或 None, impossible 原因或 None, trace)]，
    每个变体一条（无歧义时仅一条）。"""
    specs = _parse_specs(mech_children)
    combos = _ambiguous_variants(specs, dict(cids_mols))
    if not combos:
        return [(None, "π 进攻歧义箭头过多（>2），本步不模拟", [])]
    results = []
    for variant in combos:
        sim = _Sim(cids_mols, variant=variant)
        bond_hooks, vacancy_hooks = {}, {}
        for spec in specs:
            sim.apply_pair_reduce(spec)
            sim.apply_hook_count(spec, bond_hooks, vacancy_hooks)
        for spec in specs:
            sim.apply_pair_form(spec)
        sim.apply_hook_form(bond_hooks, vacancy_hooks)
        if sim.impossible is None:
            sim.finalize()
            frags, err = sim.fragment()
            results.append((frags, err or sim.impossible, sim.trace))
        else:
            results.append((None, sim.impossible, sim.trace))
    return results


def _normalize(mol):
    """显式 H 折叠为隐含（同一分子无论 H 写不写实都是同一物种；显式 H
    只是机理箭头的引用需要）。孤立 H（游离质子/氢自由基，无邻居）保留。
    芳香体系统一芳香化，凯库勒/芳香写法差异归一。"""
    try:
        mol = Chem.RemoveHs(mol, sanitize=False)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass   # 无法消毒的按原样比对（调用方的价态检查在 fragment 已做）
    return mol


def _signature(mol):
    """图同构签名：规范原子序下的（原子属性多重集，键多重集）。
    原子属性含形式电荷/自由基电子数/总氢数（折叠后）。"""
    mol = _normalize(mol)
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    atoms = sorted((int(ranks[a.GetIdx()]), a.GetSymbol(),
                    a.GetFormalCharge(), a.GetNumRadicalElectrons(),
                    a.GetTotalNumHs()) for a in mol.GetAtoms())
    bonds = sorted((min(int(ranks[b.GetBeginAtomIdx()]),
                        int(ranks[b.GetEndAtomIdx()])),
                    max(int(ranks[b.GetBeginAtomIdx()]),
                        int(ranks[b.GetEndAtomIdx()])),
                    int(round(b.GetBondTypeAsDouble())))
                   for b in mol.GetBonds())
    return (tuple(atoms), tuple(bonds))


def _smiles_of(mols) -> str:
    return "、".join(sorted(Chem.MolToSmiles(m) for m in mols))


def _compare(predicted, declared, left_sigs):
    """声明物种是否都被推出（多重集包含）；预告允许多出：与左侧完全相同
    的带电旁观离子（HSO4⁻/Na⁺ 省略惯例）与游离质子（脱质子副产惯例）。
    返回多出的声明物种（=没被推出的）列表。"""
    pred_sigs = [_signature(m) for m in predicted]
    missing = []
    for cid, mol in declared:
        sig = _signature(mol)
        if sig in pred_sigs:
            pred_sigs.remove(sig)
        else:
            missing.append(cid)
    return missing


def verify_step(left, mech_children, right) -> tuple:
    """验证一步：left/right 为 [(cid, mol)]，mech_children 为箭头标记。

    返回 (原因串, trace)——原因为 "" 表示通过。错侧判定：
    - 全部变体连合法结构都产不出 → 箭头必错（确定）；
    - 有产物但推不出声明产物 → 默认信产物修箭头（产物已过守恒/价态
      独立检查），消息附模拟预期值；翻转修产物归 P1.5。
    """
    if not mech_children:
        return "", []
    results = simulate(left, mech_children)
    traces = []
    mismatch_info = None
    for frags, impossible, trace in results:
        traces.extend(trace)
        if impossible is not None:
            continue
        left_sigs = [_signature(m) for _, m in left]
        missing = _compare(frags, right, left_sigs)
        if not missing:
            return "", traces
        if mismatch_info is None:
            mismatch_info = (frags, missing)
    if mismatch_info is None:
        # 所有变体都不可能——箭头在化学上不成立（确定的错侧判定）
        reason = next((r for _, r, _ in results if r), "无法构成合法结构")
        return (f"MECHARROW 电子流模拟：按这些箭头推导{reason}——"
                f"箭头在化学上不成立，请重画机理箭头"), traces
    frags, missing = mismatch_info
    pred_desc = _smiles_of(frags)
    miss_desc = "、".join(missing)
    return (f"MECHARROW 电子流模拟：这些箭头从反应物推出的是 {pred_desc}，"
            f"推不出声明的产物组件 {miss_desc}——声明产物已通过守恒与价态"
            f"检查，大概率是箭头画错，请重画机理箭头使其能推出声明产物"
            f"（若确认箭头无误，产物应改为模拟推得的结构）"), traces


# ---------------------------------------------------------------------------
# COMPOSITE 级接入：reaction 布局逐步模拟 + BLOCK 共振块内模拟
# ---------------------------------------------------------------------------

def _segments(children):
    """按 ARROW 分段（与 tag_validator 的 comp_step 口径一致）：
    返回 (segments, comp_seg, block_inner)。
    segments = 每段主序列 cid 列表（sup 附件计入归属段）；comp_seg =
    {cid: 段序号}；block_inner = BLOCK 内组件 id 集（外层模拟遇其引用
    时跳过该步——块内/跨块混合引用由块内模拟单独处理）。"""
    segments, comp_seg, block_inner = [[]], {}, set()
    seg = 0
    n_reg = 0    # 组件注册计数（无 id 组件按 r{n} 自动编号，与校验器同口径）
    for child in children:
        if child.type == "ARROW":
            sup = child.args[1] if len(child.args) > 1 else []
            for s in (sup or []):
                s = s.strip()
                if s:
                    sign, sid = (s[0], s[1:]) if s[0] in "+-" else ("+", s)
                    tgt = seg if sign == "+" else seg + 1
                    comp_seg[sid] = tgt
                    while len(segments) <= tgt:
                        segments.append([])
                    segments[tgt].append(sid)
            seg += 1
            segments.append([])
        elif child.type == "STRUCT":
            cid = child.attrs.get("id")
            if child.attrs.get("arrow"):
                n_reg += 1
                continue   # 附件在 ARROW 的 sup 处归属
            cid = cid or f"r{n_reg}"
            n_reg += 1
            if cid not in comp_seg:
                comp_seg[cid] = seg
                segments[seg].append(cid)
        elif child.type == "BLOCK":
            for bc in (child.args[0] if child.args else []):
                if bc.type == "STRUCT":
                    bid = bc.attrs.get("id") or f"b{n_reg}r{n_reg}"
                    n_reg += 1
                    if bid not in comp_seg:
                        comp_seg[bid] = seg
                        block_inner.add(bid)
    return segments, comp_seg, block_inner


def _mechs_by_segment(mech_children, comp_seg, block_inner=frozenset()):
    """机理箭头按其引用组件所在段分配（跨段的已被校验器拦截）。
    引用 BLOCK 内组件的箭头不进入外层分段模拟（块内/跨块混合引用
    由块内模拟单独处理）。"""
    out = {}
    for child in mech_children:
        for spec in (child.args[0].split(",") if child.args
                     and child.args[0] else []):
            spec = spec.strip()
            if not spec:
                continue
            m = _ARROW_SPEC_RE.match(spec)
            if not m:
                continue
            ref_cids = [m.group(1), m.group(4), m.group(6)]
            if any(c in block_inner for c in ref_cids if c):
                continue
            refs = {comp_seg.get(c) for c in ref_cids if c}
            refs.discard(None)
            if len(refs) == 1:
                out.setdefault(refs.pop(), []).append(spec)
    return out


def _dummy_free(mol) -> bool:
    """不含 dummy 原子（R/Ph 缩写）且为真实 RDKit Mol（fake 环境跳过）。"""
    atoms_fn = getattr(mol, "GetAtoms", None)
    if atoms_fn is None:
        return False
    return all(a.GetAtomicNum() != 0 for a in atoms_fn())


def verify_composite_electron_flow(children, comps, comp_mols) -> tuple:
    """COMPOSITE 电子流自洽校验入口。返回 (原因串, trace)。

    reaction 布局逐步模拟（左段组件 + 左段机理箭头 → 推出右段声明产物）；
    BLOCK 共振块内共振步同理。跳过面见模块 docstring（宁漏勿拦）。
    """
    mech_children = []
    for child in children:
        if child.type == "MECHARROW":
            mech_children.append(child)
        elif child.type == "BLOCK":
            mech_children.extend(
                bc for bc in (child.args[0] if child.args else [])
                if bc.type == "MECHARROW")
    if not mech_children:
        return "", []
    segments, comp_seg, block_inner = _segments(children)
    mechs_seg = _mechs_by_segment(mech_children, comp_seg, block_inner)
    traces = []
    for i in range(len(segments) - 1):
        specs = mechs_seg.get(i)
        if not specs:
            continue   # 左段无箭头（纯反应式步）——守恒已查，不模拟
        left_ids, right_ids = segments[i], segments[i + 1]
        mols_l, mols_r = [], []
        skip = False
        for cid in left_ids + right_ids:
            info = comps.get(cid)
            mol = comp_mols.get(cid)
            if info is None or mol is None or info.get("formula") \
                    or not _dummy_free(mol):
                skip = True     # 化学式/无法解析/dummy 占位组件——跳过该步
                break
            (mols_l if cid in set(left_ids) else mols_r).append((cid, mol))
        if skip:
            continue
        reason, tr = verify_step(mols_l, mechs_seg[i], mols_r)
        traces.extend(tr)
        if reason:
            return reason, traces
    # BLOCK 共振块内模拟（块内引用组件已全局注册，comp_mols 可取）
    for child in children:
        if child.type != "BLOCK":
            continue
        inner = child.args[0] if child.args else []
        seg_in, comp_seg_in, _ = _segments(inner)
        mech_in = [c for c in inner if c.type == "MECHARROW"]
        if not mech_in:
            continue
        mechs_in = _mechs_by_segment(mech_in, comp_seg_in)
        for i in range(len(seg_in) - 1):
            specs = mechs_in.get(i)
            if not specs:
                continue
            left_ids, right_ids = seg_in[i], seg_in[i + 1]
            mols_l = [(cid, comp_mols[cid]) for cid in left_ids
                      if comp_mols.get(cid) is not None]
            mols_r = [(cid, comp_mols[cid]) for cid in right_ids
                      if comp_mols.get(cid) is not None]
            if len(mols_l) != len(left_ids) or len(mols_r) != len(right_ids):
                continue
            if not all(_dummy_free(m) for _, m in mols_l + mols_r):
                continue
            reason, tr = verify_step(mols_l, specs, mols_r)
            traces.extend(tr)
            if reason:
                return reason, traces
    return "", traces
