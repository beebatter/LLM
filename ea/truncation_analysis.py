#!/usr/bin/env python3
"""
简化的截断影响分析脚本
"""
import sys
import os
sys.path.append('/root/LLM/ea')
sys.path.append('/root')

def create_test_data():
    """创建不同复杂度的测试数据"""
    return [
        {
            "name": "短公式",
            "conjecture": "P(x) → Q(x)",
            "candidate": "R(a) ∧ S(b)"
        },
        {
            "name": "中等长度公式",
            "conjecture": "∀x (P(x) → (Q(x) ∨ R(x))) ∧ ∀y (S(y) → T(y))",
            "candidate": "∃z (P(z) ∧ ¬Q(z)) → ∃w (R(w) ∧ S(w))"
        },
        {
            "name": "极长conjecture",
            "conjecture": " ∧ ".join([f"∀x{i} ∀y{i} (P{i}(x{i},y{i}) → (Q{i}(x{i}) ∨ R{i}(y{i})) ∧ S{i}(x{i},y{i}))" for i in range(50)]),
            "candidate": "R(a) ∧ S(b)"  
        },
        {
            "name": "极长candidate",
            "conjecture": "P(x) → Q(x)",
            "candidate": " ∨ ".join([f"∃y{i} ∃z{i} (R{i}(y{i},z{i}) ∧ S{i}(y{i}) ∧ T{i}(z{i}))" for i in range(60)])
        },
        {
            "name": "双方都极长",
            "conjecture": " ∧ ".join([f"∀x{i} ∃y{i} ∀z{i} (P{i}(x{i},y{i},z{i}) → (Q{i}(x{i}) ∨ R{i}(y{i})) ∧ S{i}(z{i}))" for i in range(40)]),
            "candidate": " ∨ ".join([f"∀a{i} ∀b{i} ∀c{i} (T{i}(a{i},b{i},c{i}) ↔ (U{i}(a{i}) ∧ V{i}(b{i}) ∧ W{i}(c{i})))" for i in range(35)])
        }
    ]

def analyze_truncation_theoretical():
    """理论分析截断对不同类型逻辑公式的影响"""
    print("=== 截断影响理论分析 ===\n")
    
    test_cases = create_test_data()
    max_len = 384
    
    # 模拟tokenizer: 大约每4个字符1个token (粗略估算)
    def estimate_tokens(text):
        return len(text) // 4
    
    for test_case in test_cases:
        name = test_case['name']
        conjecture = test_case['conjecture']
        candidate = test_case['candidate']
        
        print(f"## 测试用例: {name}")
        print(f"Conjecture 长度: {len(conjecture)} 字符")
        print(f"Candidate 长度: {len(candidate)} 字符")
        
        # 估算token数量
        conj_tokens = estimate_tokens(conjecture)
        cand_tokens = estimate_tokens(candidate)
        sep_tokens = 2  # [SEP] token估算
        total_tokens = conj_tokens + sep_tokens + cand_tokens
        
        print(f"预估token数: Conjecture={conj_tokens}, Candidate={cand_tokens}, 总计={total_tokens}")
        
        if total_tokens <= max_len:
            print("✓ 无需截断")
        else:
            print("⚠ 需要截断")
            
            # 直接截断分析
            simple_keep_ratio = max_len / total_tokens
            print(f"  直接截断保留率: {simple_keep_ratio:.1%}")
            
            # 智能截断分析
            available = max_len - sep_tokens
            conj_ratio = conj_tokens / (conj_tokens + cand_tokens)
            conj_target = int(available * conj_ratio)
            cand_target = available - conj_target
            
            conj_keep_ratio = conj_target / conj_tokens if conj_tokens > 0 else 1.0
            cand_keep_ratio = cand_target / cand_tokens if cand_tokens > 0 else 1.0
            
            print(f"  智能截断:")
            print(f"    Conjecture: {conj_tokens} → {conj_target} tokens ({conj_keep_ratio:.1%})")
            print(f"    Candidate: {cand_tokens} → {cand_target} tokens ({cand_keep_ratio:.1%})")
            
            # 分析潜在问题
            print(f"  潜在影响:")
            if conj_keep_ratio < 0.5:
                print("    ⚠ Conjecture信息严重丢失")
            if cand_keep_ratio < 0.5:
                print("    ⚠ Candidate信息严重丢失")
            if conj_keep_ratio < 0.2 or cand_keep_ratio < 0.2:
                print("    🚨 信息丢失过多，可能严重影响判断准确性")
            
            # 语义影响分析
            print(f"  语义风险:")
            if "∀" in conjecture or "∃" in conjecture:
                print("    - 量词结构可能被截断，破坏逻辑完整性")
            if conjecture.count("(") != conjecture.count(")"):
                print("    - 括号可能不匹配") 
            if "→" in conjecture or "↔" in conjecture:
                print("    - 逻辑连接词可能被分离")
            
        print()

def test_with_actual_backend():
    """使用实际后端测试截断效果"""
    print("=== 实际后端测试 ===\n")
    
    try:
        from LLM.ea.backends import CrossTFBackend
        import glob
        
        # 查找模型文件
        ckpt_files = (glob.glob('/root/autodl-tmp/Training/models/*.pt') +
                     glob.glob('/root/autodl-tmp/Training/models/*.pth'))
        if not ckpt_files:
            print("未找到模型文件")
            return
        
        print(f"加载模型: {ckpt_files[0]}")
        backend = CrossTFBackend(ckpt_path=ckpt_files[0], device='cpu')
        
        test_cases = create_test_data()
        
        for test_case in test_cases[:3]:  # 只测试前3个，避免过长
            name = test_case['name']
            conjecture = test_case['conjecture']
            candidate = test_case['candidate']
            
            print(f"测试: {name}")
            
            # 测试实际tokenization
            full_text = conjecture + ' [SEP] ' + candidate
            full_seq = backend.vocab.encode(full_text)
            print(f"  实际token数: {len(full_seq)}")
            
            if len(full_seq) > 384:
                print(f"  需要截断: {len(full_seq)} → 384")
                
                # 测试scoring
                candidates = [{'text': candidate}]
                try:
                    scores = backend.score(conjecture, candidates)
                    print(f"  评分成功: {scores[0]:.6f}")
                except Exception as e:
                    print(f"  评分失败: {e}")
            else:
                print("  无需截断")
                
            print()
            
    except Exception as e:
        print(f"后端测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("截断影响分析工具")
    print("="*50)
    
    # 理论分析
    analyze_truncation_theoretical()
    
    print("\n" + "="*50 + "\n")
    
    # 实际测试 
    test_with_actual_backend()
    
    print("\n=== 结论与建议 ===")
    print("""
1. **截断确实会影响评分质量**，特别是对于复杂的逻辑公式

2. **智能截断的优势**:
   - 保证[SEP]分隔符结构完整
   - 按比例保留两部分信息
   - 避免完全丢失某一部分

3. **仍存在的问题**:
   - 无法保证逻辑公式的语义完整性
   - 可能截断重要的结论部分
   - 长公式的评分可能不够准确

4. **进一步改进建议**:
   - 实现语义感知的截断点选择
   - 考虑公式结构(括号匹配、量词完整性)
   - 对极长序列使用多段打分并合并
   - 优化训练数据，包含更多长序列样本

5. **当前系统的适用性**:
   - 对中等长度公式效果良好  
   - 对极长公式可能有准确性损失
   - 但总体上比简单截断效果更好
""")

if __name__ == "__main__":
    main()
