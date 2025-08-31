import argparse, json, re, hashlib
from collections import defaultdict

def strip_outer_parens(s: str) -> str:
    if not s: return s
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        d=0; ok=True
        for i,ch in enumerate(s):
            if ch=="(": d+=1
            elif ch==")": d-=1
            if d==0 and i < len(s)-1:
                ok=False; break
        if ok: s = s[1:-1].strip()
        else: break
    return s

def split_top_level_or(s: str):
    out=[]; d=0; start=0
    for i,ch in enumerate(s or ""):
        if ch=="(": d+=1
        elif ch==")": d-=1
        elif ch=="|" and d==0:
            out.append((s or "")[start:i]); start=i+1
    out.append((s or "")[start:])
    return out

def remove_tcf_cnf_wrapper(text: str) -> str:
    t = (text or "").strip()
    m = re.match(r"^\s*([ct]nf)\s*\(", t, re.IGNORECASE)
    if not m: return strip_outer_parens(t)
    depth=0; start=None
    for i,ch in enumerate(t):
        if ch=="(" and depth==0 and start is None:
            start=i+1
        if ch=="(":
            depth+=1
        elif ch==")":
            depth-=1
            if depth==0 and start is not None:
                inside = t[start:i]
                args_t=[]; d=0; st=0
                for j,c in enumerate(inside):
                    if c=="(":
                        d+=1
                    elif c==")":
                        d-=1
                    elif c=="," and d==0:
                        args_t.append(inside[st:j].strip()); st=j+1
                args_t.append(inside[st:].strip())
                if len(args_t)>=3:
                    body = strip_outer_parens(args_t[2])
                    return body
                break
    return strip_outer_parens(t)

def drop_leading_quantifiers(body: str) -> str:
    s = (body or "").strip()
    while True:
        m = re.match(r"^\s*[!|?]\s*\[(.*?)\]\s*:\s*(.*)$", s, re.S)
        if not m: break
        s = m.group(2).strip()
    return s

def remove_type_annotations(s: str) -> str:
    return re.sub(r":\s*\$[A-Za-z0-9_]+", "", s)

def tokenize_terms(s: str):
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\!=|=|~|\(|\)|\||,|\[|\]|:|.", s)

VAR_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$")
SKOLEM_RE_STRICT = re.compile(r"^(?:sK|sk)\d+$")
SP_RE_STRICT = re.compile(r"^(?:sP|sp)\d+(?:_iProver_def)?$")

def is_variable(tok: str) -> bool:
    return bool(VAR_RE.match(tok))

def is_skolemish(tok: str) -> bool:
    if SKOLEM_RE_STRICT.match(tok): return True
    if SP_RE_STRICT.match(tok): return True
    if tok.endswith("_iProver_def"): return True
    return False

def rename_vars_and_skolem(s: str, scope_dict=None):
    if scope_dict is None:
        vmap = {}; smap = {}
    else:
        vmap = scope_dict.setdefault('var', {})
        smap = scope_dict.setdefault('sko', {})
    vcnt = len(vmap); scnt = len(smap)
    def map_var(tok):
        nonlocal vcnt
        if tok not in vmap:
            vmap[tok] = f"V{vcnt}"; vcnt+=1
        return vmap[tok]
    def map_sko(tok):
        nonlocal scnt
        if tok not in smap:
            smap[tok] = f"SK{scnt}"; scnt+=1
        return smap[tok]
    out=[]; toks = tokenize_terms(s)
    for tok in toks:
        if is_variable(tok): out.append(map_var(tok)); continue
        if is_skolemish(tok): out.append(map_sko(tok)); continue
        out.append(tok)
    return "".join(out)

def canonicalize_equalities_in_literal(lit: str) -> str:
    t = lit.strip()
    m = re.match(r"^\s*~\s*\(\s*(.+?)\s*([!=]=)\s*(.+?)\s*\)\s*$", t)
    if m:
        a,op,b = m.group(1).strip(), m.group(2), m.group(3).strip()
        op = "!=" if op=="=" else "="
        t = f"{a}{op}{b}"
    m2 = re.match(r"^\s*(.+?)\s*([!=]=)\s*(.+?)\s*$", t)
    if m2:
        a,op,b = m2.group(1).strip(), m2.group(2), m2.group(3).strip()
        if a > b: a,b = b,a
        t = f"{a}{op}{b}"
    m3 = re.match(r"^\s*~\s*\(\s*([a-z][A-Za-z0-9_]*)\s*\(", t)
    if m3:
        t = re.sub(r"^\s*~\s*\(\s*([a-z][A-Za-z0-9_]*)\s*\(", r"~\1(", t, count=1)
        if t.endswith(")"): t = t[:-1]
    return t

def predicate_name_of_literal(lit: str) -> str:
    t = lit.lstrip()
    if t.startswith("~"): t = t[1:].lstrip()
    if "!=" in t or re.match(r".+?=.+", t): return "~EQ"
    m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*\(", t)
    if not m: return t[:16]
    return m.group(1)

def canonicalize_clause(text: str, scope=None):
    body = remove_tcf_cnf_wrapper(text)
    body = drop_leading_quantifiers(body)
    body = remove_type_annotations(body)
    lits = [canonicalize_equalities_in_literal(strip_outer_parens(l))
            for l in split_top_level_or(body)]
    clause_raw = "|".join(l.strip() for l in lits if l.strip())
    clause_renamed = rename_vars_and_skolem(clause_raw, scope_dict=scope)  # <-- FIXED
    lits2 = [l.strip() for l in split_top_level_or(clause_renamed) if l.strip()]
    lits2.sort(key=lambda l: (predicate_name_of_literal(l), l))
    norm = "|".join(lits2)
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

ID_TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|!=|=")
V_RE   = re.compile(r"^V\d+$")

def pred_names_from_clause_body(body: str):
    names=set()
    for lit in split_top_level_or(body):
        t = lit.strip()
        if not t: continue
        if t.startswith("~"): t=t[1:].lstrip()
        if "!=" in t or re.match(r".+?=.+", t):
            names.add("~EQ"); continue
        m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*\(", t)
        names.add(m.group(1) if m else t[:16])
    return names

def token_set(s: str, exclude_vars=True):
    toks=set(ID_TOK.findall(s or ""))
    if exclude_vars:
        toks={t for t in toks if not V_RE.match(t)}
    return toks

def jaccard(a:set,b:set)->float:
    if not a and not b: return 0.0
    inter=len(a&b); uni=len(a|b)
    return inter/uni if uni>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="输入 JSONL（大表）")
    ap.add_argument("--out", dest="outp", required=True, help="输出 JSONL（改写且保留特征）")
    ap.add_argument("--scope", choices=["clause","problem"], default="clause",
                    help="Skolem/变量映射作用域（clause 独立 / problem 统一）")
    ap.add_argument("--keep-raw", action="store_true", help="保留 raw_text/raw_text_body 备份原文")
    ap.add_argument("--recompute-overlaps", action="store_true", help="基于归一化后的文本重算两个重合度特征")
    ap.add_argument("--overlap-exclude-vars", action="store_true", default=True, help="重算时排除变量 token (V\\d+)")
    ap.add_argument("--no-overlap-exclude-vars", dest="overlap_exclude_vars", action="store_false",
                    help="重算时不排除变量 token（与上一开关互斥，后者优先）")
    args = ap.parse_args()

    scope_maps = defaultdict(lambda: {"var":{}, "sko":{}}) if args.scope=="problem" else None

    n=0; updated_overlap=0
    with open(args.outp, "w", encoding="utf-8") as fo, open(args.inp, "r", encoding="utf-8") as fi:
        for line in fi:
            if not line.strip(): continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            prob = o.get("problem_name") or ""
            scope = scope_maps[prob] if scope_maps is not None else None

            src = o.get("text_body") or o.get("text") or ""
            if not src:
                fo.write(line); continue

            norm_body = canonicalize_clause(src, scope=scope)
            key = sha1(prob + "|" + norm_body)

            if args.keep_raw:
                if "text_body" in o: o["raw_text_body"] = o["text_body"]
                if "text" in o: o["raw_text"] = o["text"]

            o["text"] = norm_body
            o["text_body"] = norm_body
            o["text_cnf"] = f"cnf(c_{key[:8]},plain,({norm_body}))."
            o.setdefault("norm", {})
            o["norm"]["body"] = norm_body
            o["norm"]["key"] = key
            o["norm"]["cnf"] = o["text_cnf"]
            o["text_is_normalized"] = True

            if args.recompute-overlaps:
                conj = o.get("conjecture_text_body") or o.get("conjecture_text") or ""
                conj_norm = canonicalize_clause(conj, scope=None) if conj else ""
                p_a = pred_names_from_clause_body(norm_body)
                p_b = pred_names_from_clause_body(conj_norm) if conj_norm else set()
                t_a = token_set(norm_body, exclude_vars=args.overlap_exclude_vars)
                t_b = token_set(conj_norm,  exclude_vars=args.overlap_exclude_vars) if conj_norm else set()
                o.setdefault("features", {})
                o["features"]["conj_pred_overlap"]   = round(jaccard(p_a, p_b), 6)
                o["features"]["conj_token_jaccard"]  = round(jaccard(t_a, t_b), 6)
                updated_overlap += 1

            fo.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
            if n % 50000 == 0:
                print(f"[progress] {n} rows")

    print(f"[done] wrote {n} rows to {args.outp}; recomputed_overlap_for={updated_overlap}")

if __name__ == "__main__":
    main()