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

from rdkit import Chem

from core.tag_parser import MECH_ARROW_RE
from renderers.mol_primitives import _VALENCE_ELECTRONS, lone_pair_count

# 箭头 spec 解析：与校验/渲染共用 core.tag_parser.MECH_ARROW_RE
_ARROW_SPEC_RE = MECH_ARROW_RE

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

    def _loc(self, gi):
        """全局序号 → 组件引用形式（cid:局部序号）——错误消息给 LLM 用，
        全局序号无法定位。"""
        for cid, off in self.offsets.items():
            nxt = [o for c, o in self.offsets.items() if o > off]
            hi = min(nxt) if nxt else self.rw.GetNumAtoms()
            if off <= gi < hi:
                return f"{cid}:{gi - off}"
        return str(gi)

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
                    f"原子 {self._loc(gi)}（{a.GetSymbol()}）超八隅体"
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
        src_id, src_pt, _dst_id, dst_pt = (m.group(1), m.group(2),
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
            # 引擎自身的原因（含 cid:局部序号定位）优先于 RDKit 的消毒报错
            #（后者只有碎片内序号，LLM 无法定位到组件）
            results.append((frags, sim.impossible or err, sim.trace))
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


def simulate_step_products(left, mech_children):
    """模拟一步并返回首个可构成的预测产物集。
    返回 (frags 或 None, 原因或 None, trace)——frags 供错侧翻转（P1.5）
    替换声明产物用。"""
    results = simulate(left, mech_children)
    traces = []
    for frags, impossible, trace in results:
        traces.extend(trace)
        if impossible is None:
            return frags, None, traces
    reason = next((r for _, r, _ in results if r), "无法构成合法结构")
    return None, reason, traces


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
        return (f"MECHARROW 电子流模拟：按这些箭头推导会得到化学上不成立的"
                f"结构（{reason}）——箭头画错，请重画机理箭头"), traces
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
    child_of = {}   # cid → 子标记（含无 id 自动编号组件；反推/翻转的
                    # 文本手术用，与 comp_seg 同一 id 口径）
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
                    if sid in child_of and sid not in segments[tgt]:
                        segments[tgt].append(sid)
            seg += 1
            segments.append([])
        elif child.type == "STRUCT":
            cid = child.attrs.get("id")
            if child.attrs.get("arrow"):
                cid = cid or f"r{n_reg}"
                child_of[cid] = child   # sup 归属在 ARROW 处登记 comp_seg
                n_reg += 1
                continue
            cid = cid or f"r{n_reg}"
            n_reg += 1
            child_of[cid] = child
            if cid not in comp_seg:
                comp_seg[cid] = seg
                segments[seg].append(cid)
        elif child.type == "BLOCK":
            for bc in (child.args[0] if child.args else []):
                if bc.type == "STRUCT":
                    bid = bc.attrs.get("id") or f"b{n_reg}r{n_reg}"
                    n_reg += 1
                    child_of[bid] = bc
                    if bid not in comp_seg:
                        comp_seg[bid] = seg
                        block_inner.add(bid)
    return segments, comp_seg, block_inner, child_of


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
    segments, comp_seg, block_inner, _child_of = _segments(children)
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
        seg_in, comp_seg_in, _bi, _co_in = _segments(inner)
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


# ---------------------------------------------------------------------------
# 图 diff 反推箭头 + 错侧自动翻转（P1.5，20260828）
#
# 设计（Drawbacks §16.3）：反应物 + 声明产物都正确时，两侧图差异本身就是
# 电子流规范——断键/成键/键级变化/电荷转移/H 迁移全部可计算。分解唯一 →
# 确定性反推箭头（免 LLM，全量重校验把关）；不唯一 → None（归手术重写，
# diff 作提示）。错侧翻转：箭头收敛不了时把声明产物改为模拟推得（同样
# 重校验把关——翻转产物若与 label/守恒冲突会被拒，安全回退）。
# ---------------------------------------------------------------------------


def _fold_with_map(mol):
    """折叠显式 H；返回 (folded_mol, orig_of_folded, explicit_h_orig)。
    orig_of_folded: 折叠序号 → 原始序号（重原子保序，RemoveHs 实测保持
    顺序）；explicit_h_orig: {anchor 折叠序号: H 原始序号}。"""
    orig_of_folded, explicit_h_orig = {}, {}
    heavy = 0
    atoms = list(mol.GetAtoms())
    for a in atoms:
        if a.GetAtomicNum() == 1:
            nbrs = a.GetNeighbors()
            if nbrs:
                anchor_orig = nbrs[0].GetIdx()
                anchor_folded = sum(1 for x in atoms
                                    if x.GetAtomicNum() > 1
                                    and x.GetIdx() < anchor_orig)
                explicit_h_orig[anchor_folded] = a.GetIdx()
        else:
            orig_of_folded[heavy] = a.GetIdx()
            heavy += 1
    folded = Chem.RemoveHs(mol, sanitize=False)
    folded.UpdatePropertyCache(strict=False)   # 重算隐含 H（否则折叠后
    # GetTotalNumHs 返回折叠前的旧值，h_delta 永远为空——20260828 实测）
    return folded, orig_of_folded, explicit_h_orig


def _bond_map(mol):
    out = {}
    for b in mol.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        out[(i, j)] = int(round(b.GetBondTypeAsDouble()))
    return out


def _pair_diff(lm, rm):
    """同一物种两写法（重原子序列相同、序号一一对应）的结构差异。"""
    lb, rb = _bond_map(lm), _bond_map(rm)
    diff = {"bond_up": [], "bond_down": [], "bond_removed": [],
            "bond_formed": [], "charge": {}, "h_delta": {}}
    for k in sorted(set(lb) | set(rb)):
        lo, ro = lb.get(k, 0), rb.get(k, 0)
        if lo == ro:
            continue
        if ro == 0:
            diff["bond_removed"].append(k)
        elif lo == 0:
            diff["bond_formed"].append(k)
        elif ro > lo:
            diff["bond_up"].append(k)
        else:
            diff["bond_down"].append(k)
    for i in range(lm.GetNumAtoms()):
        la, ra = lm.GetAtomWithIdx(i), rm.GetAtomWithIdx(i)
        if la.GetFormalCharge() != ra.GetFormalCharge():
            diff["charge"][i] = (la.GetFormalCharge(), ra.GetFormalCharge())
        if la.GetTotalNumHs() != ra.GetTotalNumHs():
            diff["h_delta"][i] = (la.GetTotalNumHs(), ra.GetTotalNumHs())
    return diff


def _diff_size(d) -> int:
    return sum(len(d[k]) for k in ("bond_up", "bond_down", "bond_removed",
                                   "bond_formed", "charge", "h_delta"))


def _best_pair_diff(lm, rm, lhx=None, max_mappings: int = 128):
    """同重原子序列的左右物种，在同符号群内枚举原子映射，取差异最小者
    （LLM 左右两侧书写顺序常不同——如叔丁基把显式 H 写在 3 号甲基而
    产物把 CH2= 写在 0 号；恒等映射会错配）。差异数相同时，优先
    "脱氢位置有显式 H"（箭头可引用）的映射。返回 (diff, perm)，
    perm[j] = 右侧 j 号原子对应的左侧序号。"""
    from itertools import permutations
    n = lm.GetNumAtoms()
    groups = {}
    for i in range(n):
        groups.setdefault(lm.GetAtomWithIdx(i).GetSymbol(), []).append(i)
    n_perm = 1
    for g in groups.values():
        import math
        n_perm *= math.factorial(len(g))
    if n_perm > max_mappings:
        return _pair_diff(lm, rm), list(range(n))   # 太大——恒等兜底
    # 生成全部映射：每个同符号群内的置换之积
    group_syms = list(groups)
    group_idxs = [groups[sym] for sym in group_syms]
    group_rperms = [
        list(permutations([j for j in range(n)
                           if rm.GetAtomWithIdx(j).GetSymbol() == sym]))
        for sym in group_syms]
    lhx_keys = set((lhx or {}).keys())

    def _score(d):
        # 脱氢位置有显式 H（可引用）的映射优先——差异数相同之下的决胜项
        bonus = 0.5 * len(set(d["h_delta"]) & lhx_keys)
        return _diff_size(d) - bonus

    from itertools import product
    best = None
    for combo in product(*group_rperms):
        perm = list(range(n))
        for idxs, rp in zip(group_idxs, combo):
            for li, rj in zip(idxs, rp):
                perm[rj] = li
        d = _pair_diff_perm(lm, rm, perm)
        if best is None or _score(d) < _score(best[0]):
            best = (d, perm)
            if _diff_size(d) == 0:
                break
    return best if best else (_pair_diff(lm, rm), list(range(n)))


def _pair_diff_perm(lm, rm, perm):
    """perm[j] = 右侧 j 号原子对应的左侧序号 下的结构差异。"""
    lb, rb = _bond_map(lm), _bond_map(rm)
    # 右侧键/原子属性映射到左侧序号
    rb_mapped = {}
    for (i, j), o in rb.items():
        rb_mapped[tuple(sorted((perm[i], perm[j])))] = o
    diff = {"bond_up": [], "bond_down": [], "bond_removed": [],
            "bond_formed": [], "charge": {}, "h_delta": {}}
    for k in sorted(set(lb) | set(rb_mapped)):
        lo, ro = lb.get(k, 0), rb_mapped.get(k, 0)
        if lo == ro:
            continue
        if ro == 0:
            diff["bond_removed"].append(k)
        elif lo == 0:
            diff["bond_formed"].append(k)
        elif ro > lo:
            diff["bond_up"].append(k)
        else:
            diff["bond_down"].append(k)
    for j in range(rm.GetNumAtoms()):
        la = lm.GetAtomWithIdx(perm[j])
        ra = rm.GetAtomWithIdx(j)
        if la.GetFormalCharge() != ra.GetFormalCharge():
            diff["charge"][perm[j]] = (la.GetFormalCharge(),
                                       ra.GetFormalCharge())
        if la.GetTotalNumHs() != ra.GetTotalNumHs():
            diff["h_delta"][perm[j]] = (la.GetTotalNumHs(),
                                        ra.GetTotalNumHs())
    return diff


_LG_SYMBOLS = {"F", "Cl", "Br", "I"}


def compute_flow_diff(left, right):
    """左右物种集合的结构差异（供箭头反推）。返回 (pairs, extra_left,
    extra_right) 或 None（多物种合并/拆分等无法干净对应的情形归 LLM
    路径）。pairs: [(left_cid, right_cid, pair_diff, left_orig_map,
    left_explicit_h)]；extra_*：未匹配的小分子/离子 cid（≤2 重原子，
    旁观/副产惯例）。主物种匹配允许"左侧去掉一个末端离去基团（卤素/
    带正电 N/O/S）后与右侧同序"（异裂离去型）。"""
    folded_l = {cid: _fold_with_map(m) for cid, m in left}
    folded_r = {cid: _fold_with_map(m) for cid, m in right}
    pairs, used_l, extra_r = [], set(), []
    for cid_r, (rm, _, _) in folded_r.items():
        seq_r = [a.GetSymbol() for a in rm.GetAtoms()]
        hit = None
        for cid_l, (lm, _, _) in folded_l.items():
            if cid_l not in used_l and \
                    [a.GetSymbol() for a in lm.GetAtoms()] == seq_r:
                hit = cid_l
                break
        if hit is not None:
            used_l.add(hit)
            pairs.append((hit, cid_r, None))
        else:
            extra_r.append(cid_r)
    extra_l = [c for c in folded_l if c not in used_l]
    # 第二遍：右侧主物种与"左侧去末端离去基团"对齐（异裂离去型）
    still_r = []
    for cid_r in extra_r:
        rm = folded_r[cid_r][0]
        seq_r = [a.GetSymbol() for a in rm.GetAtoms()]
        hit = None
        for cid_l in extra_l:
            lm = folded_l[cid_l][0]
            atoms_l = list(lm.GetAtoms())
            for a in atoms_l:
                if a.GetSymbol() not in _LG_SYMBOLS and not (
                        a.GetAtomicNum() in (7, 8, 16)
                        and a.GetFormalCharge() > 0):
                    continue
                if len(a.GetNeighbors()) != 1:
                    continue   # 只去末端离去基团
                seq_drop = [x.GetSymbol() for x in atoms_l
                            if x.GetIdx() != a.GetIdx()]
                if seq_drop == seq_r:
                    hit = (cid_l, a.GetIdx())
                    break
            if hit:
                break
        if hit:
            extra_l.remove(hit[0])
            pairs.append((hit[0], cid_r, hit[1]))   # 第三元素=被去掉的 LG 序号
        else:
            still_r.append(cid_r)
    extra_r = still_r
    for cid in extra_l + extra_r:
        src = folded_l.get(cid) or folded_r.get(cid)
        if src[0].GetNumAtoms() > 2:
            return None
    out = []
    for cid_l, cid_r, drop in pairs:
        lm, l_map, l_hx = folded_l[cid_l]
        rm = folded_r[cid_r][0]
        if drop is None:
            out.append((cid_l, cid_r, _best_pair_diff(lm, rm, lhx=l_hx)[0],
                        l_map, l_hx, lm))
        else:
            # 去 LG 后的差异 = 断键 + LG 对应位
            nb = lm.GetAtomWithIdx(drop).GetNeighbors()[0].GetIdx()
            d = {"bond_up": [], "bond_down": [],
                 "bond_removed": [tuple(sorted((drop, nb)))],
                 "bond_formed": [], "charge": {}, "h_delta": {}}
            out.append((cid_l, cid_r, d, l_map, l_hx, lm))
    return out, extra_l, extra_r


def derive_arrows(diff_data):
    """图 diff → 弯箭头 spec 列表；分解不唯一/不认识 → None（归 LLM 路径）。

    支持模式（diff 项必须全部被消费，否则 None）：
    1a. 去质子/再芳构化/E1：某原子 h 数减 1 + 相邻原子电荷 +1→0
        → x:a-h > x:a-b（C—H 电子落向 a—b 键）；
    1b. 自由脱质子成碳负离子：某原子 h 减 1 + 自身电荷 0→-1
        → x:a-h > x:a（落回自身）；
    2. 带碱质子转移：主物种 h 减 + 碱物种 h 增（无键变化）
       → 碱:b > 主:a-h 和 主:a-h > 主:a；
    3. π 移位链（共振/电子重排，纯键级变化）：bond_down 与 bond_up
       邻接配对成链 → 逐对 a-b > b-c；
    4. 异裂离去：bond_removed[(a,b)] + 右侧游离 LG 离子 → x:a-b > x:b。
    原子序号为**折叠图局部序号**，写出箭头时经 lmap/lhx 映射回原始序号。
    """
    pairs, extra_l, extra_r = diff_data
    changed = [(cl, cr, d, lmap, lhx, lm) for cl, cr, d, lmap, lhx, lm
               in pairs if any(d.values())]
    if len(changed) == 1:
        cl, _cr, d, lmap, lhx, lm = changed[0]
        # 模式 4：异裂离去（键消失 + 无其它变化）
        if len(d["bond_removed"]) == 1 and not d["bond_up"]                 and not d["bond_down"] and not d["bond_formed"]                 and not d["h_delta"]:
            a, b = d["bond_removed"][0]
            return [f"{cl}:{lmap[a]}-{lmap[b]}>{cl}:{lmap[b]}"]
        # 模式 1a/1b：去质子（允许再芳构化的键级变化，不允许成/断键）
        if not d["bond_removed"] and not d["bond_formed"]                 and len(d["h_delta"]) == 1:
            a, (lo, hi) = next(iter(d["h_delta"].items()))
            if lo > hi:   # h 数减少 = 脱质子
                charge = d["charge"]
                h_orig = lhx.get(a)
                if h_orig is None:
                    return None   # H 是隐含的，箭头无法引用——归 LLM
                # 1a：相邻原子电荷 +1→0（正电荷被新 π 键填补）——
                # 必须相邻（电子落向二者之间的键；σ 络合物闭环键同理）
                lbonds = _bond_map(lm)
                for b, (co, cn) in charge.items():
                    if co == 1 and cn == 0 \
                            and tuple(sorted((a, b))) in lbonds:
                        return [f"{cl}:{lmap[a]}-{h_orig}>{cl}:"
                                f"{lmap[a]}-{lmap[b]}"]
                # 1b：自身电荷 0→-1（碳负离子）
                if charge.get(a) == (0, -1) and len(charge) == 1:
                    return [f"{cl}:{lmap[a]}-{h_orig}>{cl}:{lmap[a]}"]
                return None
        # 模式 3：π 移位链（纯键级变化，无电荷/H 变化）
        up, down = d["bond_up"], d["bond_down"]
        if (up or down) and not d["bond_removed"] and not d["bond_formed"] \
                and not d["h_delta"] and not d["charge"]:
            specs = []
            up_pool = list(up)
            for a, b in down:
                hit = None
                for k in up_pool:
                    shared = set(k) & {a, b}
                    if len(shared) == 1:
                        hit = k
                        break
                if hit is None:
                    return None   # 有断无配——分解不唯一
                up_pool.remove(hit)
                shared_atom = next(iter(set(hit) & {a, b}))
                c = next(iter(set(hit) - {shared_atom}))
                specs.append(f"{cl}:{lmap[a]}-{lmap[b]}>"
                             f"{cl}:{lmap[shared_atom]}-{lmap[c]}")
            if up_pool:
                return None
            return specs or None
    # 模式 2：带碱质子转移（两对纯 H 增减，无键变化）
    if len(changed) == 2:
        if all(not d["bond_up"] and not d["bond_down"]
               and not d["bond_removed"] and not d["bond_formed"]
               for _, _, d, _, _, _ in changed):
            drop_p = gain_p = None
            for cl, _cr, d, lmap, lhx, _lm in changed:
                for i, (lo, hi) in d["h_delta"].items():
                    if lo > hi:
                        drop_p = (cl, i, lmap, lhx)
                    else:
                        gain_p = (cl, i, lmap)
            if drop_p and gain_p:
                cl_d, a, lmap_d, lhx_d = drop_p
                cl_g, b, lmap_g = gain_p
                h_orig = lhx_d.get(a)
                if h_orig is None:
                    return None
                ao, bo = lmap_d[a], lmap_g[b]
                return [f"{cl_g}:{bo}>{cl_d}:{h_orig}",
                        f"{cl_d}:{ao}-{h_orig}>{cl_d}:{ao}"]
        # 模式 2b：带碱 E1 去质子（主物种 h 减 + 键升 + 相邻电荷 +1→0；
        # 碱物种 h 增）→ 碱孤对 > H 原子 + C—H 电子 > C—C 键
        drop_p = gain_p = None
        for cl, _cr, d, lmap, lhx, _lm in changed:
            if d["h_delta"] and any(lo > hi for _, (lo, hi)
                                    in d["h_delta"].items()) \
                    and (d["bond_up"] or d["bond_down"]):
                drop_p = (cl, d, lmap, lhx)
            elif d["h_delta"] and all(lo < hi for _, (lo, hi)
                                      in d["h_delta"].items()) \
                    and not any(d[k] for k in
                                ("bond_up", "bond_down", "bond_removed",
                                 "bond_formed")):
                gain_p = (cl, d, lmap)
        if drop_p and gain_p:
            cl_d, d, lmap_d, lhx_d = drop_p
            cl_g, dg, lmap_g = gain_p
            if len(d["bond_up"]) == 1 and len(d["h_delta"]) == 1 \
                    and not d["bond_down"] and not d["bond_removed"] \
                    and not d["bond_formed"] and len(dg["h_delta"]) == 1:
                a, b = d["bond_up"][0]
                ha = d["h_delta"].get(a)
                hb = d["h_delta"].get(b)
                drop_atom, other = (a, b) if ha else (b, a) if hb \
                    else (None, None)
                ok_charge = all(co == 1 and cn == 0 and i == other
                                for i, (co, cn) in d["charge"].items())
                base_atom = next(iter(dg["h_delta"]))   # 碱上得 H 的原子即供体
                if drop_atom is not None and ok_charge:
                    h_orig = lhx_d.get(drop_atom)
                    if h_orig is not None:
                        ao, bo = lmap_d[drop_atom], lmap_d[other]
                        return [f"{cl_g}:{lmap_g[base_atom]}>"
                                f"{cl_d}:{h_orig}",
                                f"{cl_d}:{ao}-{h_orig}>{cl_d}:{ao}-{bo}"]
    return None


def fix_arrows_by_diff(tag):
    """图 diff 反推箭头（确定性，免 LLM）：模拟不一致时，按反应物与
    声明产物的结构差异导出正确箭头并替换进标记原文。

    返回 (新 raw, 说明) 或 None（无法干净 diff / 分解不唯一）。
    调用方必须全量重校验后再采用（校验器 + 模拟器把关）。"""
    if tag.type != "COMPOSITE" or len(tag.args) < 2:
        return None
    children = tag.args[1]
    segments, comp_seg, block_inner, child_of = _segments(children)
    # 组件表（与校验器同口径的 coeff 处理；无 id 组件经 child_of 的自动
    # 编号定位——Q17 病例的产物侧就没有显式 id）
    from core.tag_validator import _parse_coeff, _parse_mol
    mols = {}
    for cid, child in child_of.items():
        if not child.args or not child.args[0]:
            continue
        parsed = _parse_coeff(child.args[0].strip())
        bare = parsed[1] if parsed else child.args[0].strip()
        try:
            m = _parse_mol(bare)
        except Exception:
            m = None
        if m is not None:
            mols[cid] = m

    def _fix_scope(scope_children, scope_segments, scope_comp_seg, raw,
                   close, skip_refs):
        """对一段子序列（外层 / BLOCK 内）逐步 diff 反推并替换箭头。"""
        mechs = _mechs_by_segment([c for c in scope_children
                                   if c.type == "MECHARROW"],
                                  scope_comp_seg, skip_refs)
        new_raw, changed = raw, False
        for i in range(len(scope_segments) - 1):
            specs = mechs.get(i)
            if not specs:
                continue
            left = [(cid, mols[cid]) for cid in scope_segments[i]
                    if cid in mols]
            right = [(cid, mols[cid]) for cid in scope_segments[i + 1]
                     if cid in mols]
            if not left or not right:
                return raw, False
            diff_data = compute_flow_diff(left, right)
            if diff_data is None:
                return raw, False
            derived = derive_arrows(diff_data)
            if not derived:
                return raw, False
            # 替换本步的全部 MECHARROW（旧箭头整组作废）
            step_mech_raws = [c.raw for c in scope_children
                              if c.type == "MECHARROW"
                              for s in _parse_specs([c]) if s in set(specs)]
            for r in step_mech_raws:
                new_raw = new_raw.replace(r, "", 1)
            insert = "".join(f"[MECHARROW:{s}]" for s in derived)
            new_raw = new_raw.replace(close, insert + close)
            changed = True
        return new_raw, changed

    new_raw, changed = _fix_scope(children, segments, comp_seg, tag.raw,
                                  "[/COMPOSITE]", block_inner)
    # BLOCK 共振块内逐步反推（块内组件已全局注册进 child_of/mols；
    # 内层调用的 skip_refs 为空——块内引用块内组件是合法主流）
    for child in children:
        if child.type != "BLOCK":
            continue
        inner = child.args[0] if child.args else []
        seg_in, cseg_in, _bi, _co = _segments(inner)
        fixed_block, ch = _fix_scope(inner, seg_in, cseg_in, child.raw,
                                     "[/BLOCK]", frozenset())
        if ch:
            new_raw = new_raw.replace(child.raw, fixed_block, 1)
            changed = True
    return (new_raw, "图 diff 反推箭头（免 LLM）") if changed else None


def flip_products_to_simulated(tag):
    """错侧自动翻转：把声明产物改为电子流模拟推得的结构（箭头不动）。

    仅当逐步右侧物种数与预测数对齐（或可通过"两声明物种合并为一预测
    物种"对齐，如 乙醚+H+ ← 质子化乙醚）时进行；替换保留原组件 id/label。
    返回 (新 raw, 说明) 或 None。调用方必须全量重校验后再采用。"""
    if tag.type != "COMPOSITE" or len(tag.args) < 2:
        return None
    children = tag.args[1]
    segments, comp_seg, block_inner, child_of = _segments(children)
    if block_inner:
        return None
    from core.tag_validator import _parse_coeff, _parse_mol
    mols = {}
    for cid, child in child_of.items():
        if not child.args or not child.args[0]:
            continue
        parsed = _parse_coeff(child.args[0].strip())
        bare = parsed[1] if parsed else child.args[0].strip()
        try:
            m = _parse_mol(bare)
        except Exception:
            m = None
        if m is not None:
            mols[cid] = m
    mechs = _mechs_by_segment([c for c in children if c.type == "MECHARROW"],
                              comp_seg)
    new_raw = tag.raw
    for i in range(len(segments) - 1):
        specs = mechs.get(i)
        if not specs:
            continue
        left = [(cid, mols[cid]) for cid in segments[i] if cid in mols]
        right_ids = [cid for cid in segments[i + 1] if cid in mols]
        frags, reason, _ = simulate_step_products(
            left, [s for s in specs])
        if frags is None:
            return None   # 箭头不可能——错在箭头不在产物，不翻转
        pred = [Chem.MolToSmiles(f) for f in frags]

        def _formula(m):
            # 元素计数（显式 H 原子计入 H 总数）+ 总电荷 + 总单电子数——
            # 不区分电荷/氢在哪个原子上（翻转的对齐是物种级，不做原子分配）
            cnt = {}
            for a in m.GetAtoms():
                cnt[a.GetSymbol()] = cnt.get(a.GetSymbol(), 0) + 1
            cnt["H"] = cnt.get("H", 0) + sum(
                a.GetTotalNumHs() for a in m.GetAtoms()
                if a.GetAtomicNum() != 1)
            q = sum(a.GetFormalCharge() for a in m.GetAtoms())
            r = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
            return (tuple(sorted(cnt.items())), q, r)

        pred_f = [_formula(f) for f in frags]
        decl_f = {cid: _formula(mols[cid]) for cid in right_ids}
        # 逐一匹配（同式替换，保留 id/label）
        used_pred = set()
        repl = {}
        unmatched_decl = []
        for cid in right_ids:
            hit = next((j for j, pf in enumerate(pred_f)
                        if j not in used_pred and pf == decl_f[cid]), None)
            if hit is not None:
                used_pred.add(hit)
                repl[cid] = pred[hit]
            else:
                unmatched_decl.append(cid)
        unmatched_pred = [j for j in range(len(pred)) if j not in used_pred]
        # 合并对齐：一个预测物种 = 两个未匹配声明物种之和（如 质子化乙醚
        # = 乙醚 + H+）
        def _fadd(f1, f2):
            c = {}
            for k, v in list(f1[0]) + list(f2[0]):
                c[k] = c.get(k, 0) + v
            return (tuple(sorted(c.items())), f1[1] + f2[1],
                    f1[2] + f2[2])
        for j in list(unmatched_pred):
            merged = False
            for x in unmatched_decl:
                for y in unmatched_decl:
                    if x >= y:
                        continue
                    if _fadd(decl_f[x], decl_f[y]) == pred_f[j]:
                        repl[x] = pred[j]
                        repl[y] = None   # 删除
                        unmatched_pred.remove(j)
                        unmatched_decl.remove(x)
                        unmatched_decl.remove(y)
                        merged = True
                        break
                if merged:
                    break
        if unmatched_decl or unmatched_pred:
            return None   # 物种数对不齐——不翻转
        for cid, new_smi in repl.items():
            child = child_of[cid]
            if new_smi is None:
                # 删除该 STRUCT 及相邻一个 PLUS
                new_raw = new_raw.replace(f"[PLUS]{child.raw}", "", 1)
                if child.raw in new_raw:
                    new_raw = new_raw.replace(child.raw + "[PLUS]", "", 1) \
                        if child.raw + "[PLUS]" in new_raw else new_raw
                new_raw = new_raw.replace(child.raw, "", 1)
                continue
            old_smi = (child.args[0] or "").strip()
            new_child = child.raw.replace(old_smi, new_smi, 1)
            new_raw = new_raw.replace(child.raw, new_child, 1)
    return new_raw, "错侧翻转：产物改为电子流模拟推得结构"
