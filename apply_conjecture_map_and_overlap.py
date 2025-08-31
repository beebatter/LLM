import argparse, json, re, sys, hashlib

BUILTINS = {"$true","$false"}

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

def split_top_level_commas(s: str):
    items=[]; d=0; start=0
    for i,ch in enumerate(s):
        if ch=="(":
            d+=1
        elif ch==")":
            d-=1
        elif ch=="," and d==0:
            items.append(s[start:i].strip()); start=i+1
    items.append(s[start:].strip())
    return items

def parse_tstp_fun(s: str):
    s = s.strip()
    m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*\(", s)
    if not m: return None, []
    name = m.group(1)
    inner = s[s.find("(", m.end(1)-1)+1:]
    if inner.endswith(")."): inner = inner[:-2]
    if inner.endswith(")"): inner = inner[:-1]
    args = split_top_level_commas(inner)
    return name, args

def to_clause_body(text: str) -> str:
    if not text: return ""
    t = text.strip()
    m = re.match(r"^\s*([ct]nf)\s*\(", t, re.IGNORECASE)
    if m:
        _, args = parse_tstp_fun(t)
        if len(args) >= 3:
            body = args[2].strip()
            body = strip_outer_parens(body)
            return body
    return strip_outer_parens(t)

def to_cnf_text(body: str, problem: str, role: str="plain", seed: str="") -> str:
    body = body.strip()
    h = hashlib.sha1((problem + "|" + body + "|" + seed).encode("utf-8")).hexdigest()[:8]
    cid = f"c_{h}"
    return f"cnf({cid},{role},({body}))."

def split_top_level_or(s: str):
    out=[]; d=0; start=0
    for i,ch in enumerate(s or ""):
        if ch=="(":
            d+=1
        elif ch==")":
            d-=1
        elif ch=="|" and d==0:
            out.append((s or "")[start:i]); start=i+1
    out.append((s or "")[start:])
    return out

def top_predicates_from_clause_text(txt: str):
    preds=set()
    for lit in split_top_level_or(txt or ""):
        t = lit.strip()
        if not t: continue
        if t.startswith("~"): t = t[1:].strip()
        if "!=" in t or "=" in t:
            continue
        m = re.match(r"^([a-z][A-Za-z0-9_]*)\s*\(", t)
        if m:
            name = m.group(1)
            if name not in BUILTINS:
                preds.add(name)
    return preds

def anylevel_predicates(s: str):
    """
    FOF-friendly predicate scan: collect all names that look like name(...).
    This may include function symbols, but works well as a high-recall signal.
    """
    names = set()
    for m in re.finditer(r"\b([a-z][A-Za-z0-9_]*)\s*\(", s or ""):
        name = m.group(1)
        if name not in BUILTINS and name not in {"and","or","not","implies","iff"}:
            names.add(name)
    return names

def symbols_all_tokens(s: str):
    toks = set(re.findall(r"\b[a-z][A-Za-z0-9_]*\b", s or ""))
    return {t for t in toks if t not in BUILTINS}

def jaccard(a:set,b:set)->float:
    if not a and not b: return 0.0
    u = len(a | b); i = len(a & b)
    return (i / u) if u else 0.0

def load_conjecture_map(path: str):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    cmap = {}
    for k,v in m.items():
        if isinstance(v, str):
            cmap[k] = v
        elif isinstance(v, dict):
            ct = v.get("conjecture_text")
            if ct: cmap[k] = ct
    return cmap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--conj", required=True, help="Problem -> conjecture map (JSON)")
    ap.add_argument("--out", required=True, help="Output JSONL")
    ap.add_argument("--force", action="store_true", help="Overwrite existing conjecture_text if present")
    ap.add_argument("--emit", choices=["none","body","cnf","both"], default="none",
                    help="Emit normalized text fields (default: none)")
    ap.add_argument("--rewrite-text", choices=["none","body","cnf"], default="none",
                    help="Rewrite the 'text' field (default: none)")
    ap.add_argument("--cnf-role", default="plain", help="Role to use when wrapping CNF (default: plain)")
    args = ap.parse_args()

    cmap = load_conjecture_map(args.conj)

    wrote = 0; filled = 0
    with open(args.out, "w", encoding="utf-8") as out, \
         open(args.inp, "r", encoding="utf-8") as inp:
        for line in inp:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prob = obj.get("problem_name")
            if not prob:
                out.write(line); wrote += 1; continue

            # fill conjecture_text if missing or --force
            if args.force or not obj.get("conjecture_text"):
                ct = cmap.get(prob, "")
                if ct:
                    obj["conjecture_text"] = ct
                    filled += 1

            # compute normalized bodies (for emit/features)
            body = to_clause_body(obj.get("text",""))
            conj_text = obj.get("conjecture_text","")
            conj_body = to_clause_body(conj_text) if conj_text else ""

            # emit normalized text variants
            if args.emit in ("body","both"):
                obj["text_body"] = body
                if conj_body:
                    obj["conjecture_text_body"] = conj_body
            if args.emit in ("cnf","both"):
                obj["text_cnf"] = to_cnf_text(body, prob, role=args.cnf_role)

            # rewrite original text if requested
            if args.rewrite_text == "body":
                obj["text"] = body
            elif args.rewrite_text == "cnf":
                obj["text"] = to_cnf_text(body, prob, role=args.cnf_role)

            # add overlap features (only if we have a conjecture)
            if conj_text:
                feats = obj.setdefault("features", {})
                # For clauses (CNF bodies), use top-level; for conjectures, try top-level then fallback to any-level scan
                preds_clause = top_predicates_from_clause_text(body)
                preds_conj   = top_predicates_from_clause_text(conj_body) if conj_body else set()
                if not preds_conj:
                    preds_conj = anylevel_predicates(conj_text)  # FOF fallback
                toks_clause  = symbols_all_tokens(body)
                toks_conj    = symbols_all_tokens(conj_text)
                feats["conj_pred_overlap"]  = round(jaccard(preds_clause, preds_conj), 6)
                feats["conj_token_jaccard"] = round(jaccard(toks_clause, toks_conj), 6)

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"[done] wrote={wrote}, filled_conjecture={filled}, out={args.out}")

if __name__ == "__main__":
    main()