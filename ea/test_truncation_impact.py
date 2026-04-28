#!/usr/bin/env python3
"""
Test script to evaluate truncation impact on real data
"""
import sys
import os
sys.path.append('/root/LLM/ea')

import json
from backends import CrossTFBackend
from truncation_eval import TruncationEvaluator

def create_test_data():
    """Create test data with various complexity levels"""
    test_cases = [
        {
            "name": "short_formulas",
            "conjecture": "P(x) → Q(x)",
            "candidate": "R(a) ∧ S(b)"
        },
        {
            "name": "medium_formulas", 
            "conjecture": "∀x (P(x) → (Q(x) ∨ R(x))) ∧ ∀y (S(y) → T(y))",
            "candidate": "∃z (P(z) ∧ ¬Q(z)) → ∃w (R(w) ∧ S(w))"
        },
        {
            "name": "very_long_conjecture",
            "conjecture": " ∧ ".join([f"∀x{i} ∀y{i} (P{i}(x{i},y{i}) → (Q{i}(x{i}) ∨ R{i}(y{i})) ∧ S{i}(x{i},y{i}))" for i in range(50)]),
            "candidate": "R(a) ∧ S(b)"  
        },
        {
            "name": "very_long_candidate",
            "conjecture": "P(x) → Q(x)",
            "candidate": " ∨ ".join([f"∃y{i} ∃z{i} (R{i}(y{i},z{i}) ∧ S{i}(y{i}) ∧ T{i}(z{i}))" for i in range(60)])
        },
        {
            "name": "extremely_long_both",
            "conjecture": " ∧ ".join([f"∀x{i} ∃y{i} ∀z{i} (P{i}(x{i},y{i},z{i}) → (Q{i}(x{i}) ∨ R{i}(y{i})) ∧ S{i}(z{i}))" for i in range(40)]),
            "candidate": " ∨ ".join([f"∀a{i} ∀b{i} ∀c{i} (T{i}(a{i},b{i},c{i}) ↔ (U{i}(a{i}) ∧ V{i}(b{i}) ∧ W{i}(c{i})))" for i in range(35)])
        }
    ]
    return test_cases

def analyze_truncation_patterns(backend, test_cases):
    """Analyze what happens with different truncation approaches"""
    print("=== Truncation Pattern Analysis ===\n")
    
    # Get max_len - use the same value as in the backend
    max_len = 384
    
    for test_case in test_cases:
        name = test_case['name']
        conjecture = test_case['conjecture']  
        candidate = test_case['candidate']
        
        print(f"## Test Case: {name}")
        print(f"Conjecture length: {len(conjecture)} chars")
        print(f"Candidate length: {len(candidate)} chars")
        
        # Test tokenization
        full_text = conjecture + ' [SEP] ' + candidate
        full_seq = backend.vocab.encode(full_text)
        print(f"Full sequence tokens: {len(full_seq)}")
        
        if len(full_seq) <= max_len:
            print("✓ No truncation needed")
        else:
            print("⚠ Truncation required")
            
            # Simple truncation
            simple_truncated = full_seq[:max_len]
            simple_decoded = backend.vocab.decode(simple_truncated)
            
            # Our smart truncation (simulate)
            sep_seq = backend.vocab.encode(' [SEP] ')
            overhead = len(sep_seq)
            available = max_len - overhead
            
            conj_seq = backend.vocab.encode(conjecture)
            cand_seq = backend.vocab.encode(candidate)
            
            if len(conj_seq) + len(cand_seq) > available:
                total_original = len(conj_seq) + len(cand_seq)
                conj_ratio = len(conj_seq) / total_original
                conj_target = int(available * conj_ratio)
                cand_target = available - conj_target
                
                smart_conj = conj_seq[:conj_target]
                smart_cand = cand_seq[:cand_target] 
                smart_seq = smart_conj + sep_seq + smart_cand
                smart_decoded = backend.vocab.decode(smart_seq)
                
                print(f"  Simple truncation result length: {len(simple_decoded)}")
                print(f"  Smart truncation result length: {len(smart_decoded)}")
                
                # Check if [SEP] is preserved
                has_sep_simple = '[SEP]' in simple_decoded
                has_sep_smart = '[SEP]' in smart_decoded
                print(f"  Simple preserves [SEP]: {has_sep_simple}")
                print(f"  Smart preserves [SEP]: {has_sep_smart}")
                
                # Information retention
                simple_retention = len(simple_decoded) / len(full_text)
                smart_retention = len(smart_decoded) / len(full_text)
                print(f"  Simple retention: {simple_retention:.2%}")
                print(f"  Smart retention: {smart_retention:.2%}")
                
        print()

def test_current_backend():
    """Test the current backend with various inputs"""
    print("Loading CrossTFBackend...")
    
    try:
        # Look for a model checkpoint
        import glob
        ckpt_files = (glob.glob('/root/autodl-tmp/models/*.pt') + 
                     glob.glob('/root/autodl-tmp/models/*.pth') +
                     glob.glob('/root/autodl-tmp/Training/models/*.pt') +
                     glob.glob('/root/autodl-tmp/Training/models/*.pth'))
        if not ckpt_files:
            print("No checkpoint files found")
            print("Creating a mock backend for testing truncation logic...")
            return test_truncation_logic_only()
            
        ckpt_path = ckpt_files[0]
        print(f"Using checkpoint: {ckpt_path}")
        
        backend = CrossTFBackend(
            ckpt_path=ckpt_path,
            device='cpu',  # Use CPU for testing
            vocab_path=None  # Use default vocab
        )
        print("✓ Backend loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load backend: {e}")
        print("Creating a mock backend for testing truncation logic...")
        return test_truncation_logic_only()
    
    # Create test data
    test_cases = create_test_data()
    
    # Analyze patterns
    analyze_truncation_patterns(backend, test_cases)
    
    # Test actual scoring
    print("=== Actual Scoring Test ===\n")
    
    for test_case in test_cases:
        name = test_case['name']
        conjecture = test_case['conjecture']
        candidate = test_case['candidate']
        
        print(f"Testing {name}...")
        try:
            # Test the score method - need to format as required
            candidates = [{'text': candidate}]
            scores = backend.score(conjecture, candidates)
            if scores:
                print(f"  Score: {scores[0]:.6f}")
            
            # Check if truncation statistics were recorded
            if hasattr(backend, 'truncation_stats'):
                stats = backend.truncation_stats
                if stats['total_calls'] > 0:
                    print(f"  Truncation rate: {stats['truncated_calls']}/{stats['total_calls']} ({stats['truncated_calls']/stats['total_calls']:.1%})")
                    if stats['truncated_calls'] > 0:
                        avg_retention = stats['total_retention'] / stats['truncated_calls']
                        print(f"  Average retention: {avg_retention:.1%}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
        
        print()

def test_truncation_logic_only():
    """Test truncation logic without loading actual models"""
    print("=== Truncation Logic Test (No Model) ===\n")
    
    # Mock vocab for testing
    class MockVocab:
        def encode(self, text):
            # Simple mock: 1 token per 4 characters (rough approximation)
            return list(range(len(text) // 4))
        
        def decode(self, tokens):
            # Simple mock: approximate decoding
            return "decoded_text_" + str(len(tokens))
    
    # Create test data
    test_cases = create_test_data()
    vocab = MockVocab()
    max_len = 384
    
    print(f"Mock vocab encoding, max_len = {max_len}")
    print()
    
    for test_case in test_cases:
        name = test_case['name']
        conjecture = test_case['conjecture']  
        candidate = test_case['candidate']
        
        print(f"## Test Case: {name}")
        print(f"Conjecture: '{conjecture[:100]}{'...' if len(conjecture) > 100 else ''}'")
        print(f"Candidate: '{candidate[:100]}{'...' if len(candidate) > 100 else ''}'")
        print(f"Conjecture length: {len(conjecture)} chars")
        print(f"Candidate length: {len(candidate)} chars")
        
        # Test tokenization
        full_text = conjecture + ' [SEP] ' + candidate
        full_seq = vocab.encode(full_text)
        print(f"Full sequence tokens: {len(full_seq)}")
        
        if len(full_seq) <= max_len:
            print("✓ No truncation needed")
        else:
            print("⚠ Truncation required")
            
            # Simple truncation
            simple_truncated = full_seq[:max_len]
            
            # Smart truncation simulation
            sep_seq = vocab.encode(' [SEP] ')
            overhead = len(sep_seq)
            available = max_len - overhead
            
            conj_seq = vocab.encode(conjecture)
            cand_seq = vocab.encode(candidate)
            
            print(f"  Conjecture tokens: {len(conj_seq)}")
            print(f"  Candidate tokens: {len(cand_seq)}")
            print(f"  SEP tokens: {len(sep_seq)}")
            print(f"  Available tokens: {available}")
            
            if len(conj_seq) + len(cand_seq) > available:
                # Proportional allocation
                total_original = len(conj_seq) + len(cand_seq)
                conj_ratio = len(conj_seq) / total_original
                conj_target = int(available * conj_ratio)
                cand_target = available - conj_target
                
                print(f"  Proportional allocation:")
                print(f"    Conjecture: {len(conj_seq)} → {conj_target} tokens ({conj_target/len(conj_seq):.1%})")
                print(f"    Candidate: {len(cand_seq)} → {cand_target} tokens ({cand_target/len(cand_seq):.1%})")
                
                smart_conj = conj_seq[:conj_target]
                smart_cand = cand_seq[:cand_target] 
                smart_seq = smart_conj + sep_seq + smart_cand
                
                print(f"  Result lengths:")
                print(f"    Simple truncation: {len(simple_truncated)} tokens")
                print(f"    Smart truncation: {len(smart_seq)} tokens")
                
                # Information retention analysis
                simple_retention = len(simple_truncated) / len(full_seq)
                smart_retention = len(smart_seq) / len(full_seq) 
                print(f"  Information retention:")
                print(f"    Simple: {simple_retention:.2%}")
                print(f"    Smart: {smart_retention:.2%}")
                
                # SEP preservation check
                print(f"  SEP preservation:")
                print(f"    Simple: {'✓' if len(simple_truncated) > len(conj_seq) else '✗'}")
                print(f"    Smart: ✓ (guaranteed)")
                
        print()
    
    print("=== Summary ===")
    print("Smart truncation advantages:")
    print("1. Preserves [SEP] token structure")
    print("2. Proportional allocation maintains balance") 
    print("3. Semantic boundaries can be respected")
    print("4. Information from both parts is retained")
    print()
    print("Potential concerns:")
    print("1. May truncate mid-formula without semantic awareness")
    print("2. Simple proportion may not be optimal for all cases")
    print("3. No consideration of formula importance/structure")

if __name__ == "__main__":
    test_current_backend()
