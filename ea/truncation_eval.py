"""
Truncation impact evaluation tool.
Tests different truncation strategies on sample data to measure quality impact.
"""
import json
import numpy as np
from typing import Dict, List, Tuple
import time
from advanced_truncation import TruncationStrategy, LogicAwareTruncator, MultiSegmentScorer

class TruncationEvaluator:
    """Evaluate the impact of different truncation strategies"""
    
    def __init__(self, backend):
        self.backend = backend
        self.evaluations = []
        
    def evaluate_strategy(self, test_pairs: List[Tuple[str, str]], 
                         strategy: TruncationStrategy) -> Dict:
        """Evaluate a specific truncation strategy"""
        results = {
            'strategy': strategy.value,
            'total_pairs': len(test_pairs),
            'truncated_pairs': 0,
            'score_differences': [],
            'processing_times': [],
            'information_retention': []
        }
        
        for conjecture, candidate in test_pairs:
            # Test original (if possible) vs truncated
            original_text = conjecture + ' [SEP] ' + candidate
            original_seq = self.backend.vocab.encode(original_text)
            
            if len(original_seq) <= self.backend.max_len:
                # No truncation needed
                continue
                
            results['truncated_pairs'] += 1
            
            # Time the truncation process
            start_time = time.time()
            
            if strategy == TruncationStrategy.SIMPLE:
                truncated_seq = original_seq[:self.backend.max_len]
            elif strategy == TruncationStrategy.BALANCED:
                # Use our balanced approach
                truncated_seq = self._balanced_truncate(conjecture, candidate)
            elif strategy == TruncationStrategy.SEMANTIC_AWARE:
                truncator = LogicAwareTruncator(self.backend.max_len)
                truncated_seq = truncator.truncate_semantic(
                    conjecture, candidate, self.backend.vocab.encode)
            
            processing_time = time.time() - start_time
            results['processing_times'].append(processing_time)
            
            # Calculate information retention
            retention = len(truncated_seq) / len(original_seq)
            results['information_retention'].append(retention)
            
            # Compare scores (this requires having a way to score both)
            # For now, we'll simulate this
            original_score = self._simulate_score(original_seq[:384])  # Pretend full
            truncated_score = self._simulate_score(truncated_seq)
            
            score_diff = abs(original_score - truncated_score)
            results['score_differences'].append(score_diff)
        
        # Compute statistics
        if results['score_differences']:
            results['avg_score_diff'] = np.mean(results['score_differences'])
            results['max_score_diff'] = np.max(results['score_differences'])
            results['avg_retention'] = np.mean(results['information_retention'])
            results['avg_processing_time'] = np.mean(results['processing_times'])
        
        return results
    
    def _balanced_truncate(self, conjecture: str, candidate: str) -> List[int]:
        """Our current balanced truncation implementation"""
        sep_seq = self.backend.vocab.encode(' [SEP] ')
        overhead = len(sep_seq)
        available = self.backend.max_len - overhead
        
        conj_seq = self.backend.vocab.encode(conjecture)
        cand_seq = self.backend.vocab.encode(candidate)
        
        if len(conj_seq) + len(cand_seq) <= available:
            return conj_seq + sep_seq + cand_seq
        
        # Proportional allocation
        total_original = len(conj_seq) + len(cand_seq)
        conj_ratio = len(conj_seq) / total_original
        
        conj_target = int(available * conj_ratio)
        cand_target = available - conj_target
        
        conj_seq = conj_seq[:conj_target]
        cand_seq = cand_seq[:cand_target]
        
        return conj_seq + sep_seq + cand_seq
    
    def _simulate_score(self, seq: List[int]) -> float:
        """Simulate scoring - in real test this would use actual model"""
        # Simple simulation based on sequence length and token diversity
        if not seq:
            return 0.0
        return len(set(seq)) / len(seq) + np.random.normal(0, 0.1)
    
    def run_comprehensive_evaluation(self, test_file: str = None) -> Dict:
        """Run evaluation on all strategies"""
        # Load or generate test data
        test_pairs = self._load_test_data(test_file)
        
        results = {}
        strategies = [
            TruncationStrategy.SIMPLE,
            TruncationStrategy.BALANCED,
            TruncationStrategy.SEMANTIC_AWARE
        ]
        
        for strategy in strategies:
            print(f"Evaluating {strategy.value} strategy...")
            results[strategy.value] = self.evaluate_strategy(test_pairs, strategy)
            
        return results
    
    def _load_test_data(self, test_file: str = None) -> List[Tuple[str, str]]:
        """Load test data for evaluation"""
        if test_file and os.path.exists(test_file):
            with open(test_file, 'r') as f:
                data = json.load(f)
            return [(item['conjecture'], item['candidate']) for item in data]
        
        # Generate some synthetic long test cases
        return [
            ("∀x (P(x) → Q(x))" * 50, "R(a) ∧ S(b)" * 40),  # Long conjecture
            ("P(x)" * 10, "∀y ∃z (R(y,z) → (S(z) ∨ T(y)))" * 30),  # Long candidate  
            ("∀x∀y (P(x,y) → ∃z Q(x,z))" * 25, "∀a∀b∀c (R(a,b,c) ↔ S(a) ∧ T(b))" * 25),  # Both long
        ]
    
    def generate_report(self, results: Dict) -> str:
        """Generate a readable evaluation report"""
        report = "# Truncation Strategy Evaluation Report\n\n"
        
        for strategy, data in results.items():
            report += f"## {strategy.title()} Strategy\n"
            report += f"- Pairs requiring truncation: {data['truncated_pairs']}/{data['total_pairs']}\n"
            
            if data['truncated_pairs'] > 0:
                report += f"- Average score difference: {data.get('avg_score_diff', 'N/A'):.4f}\n"
                report += f"- Maximum score difference: {data.get('max_score_diff', 'N/A'):.4f}\n"
                report += f"- Average information retention: {data.get('avg_retention', 'N/A'):.2%}\n"
                report += f"- Average processing time: {data.get('avg_processing_time', 'N/A'):.4f}s\n"
            report += "\n"
            
        return report

if __name__ == "__main__":
    # Example usage
    print("Truncation evaluation tool created.")
    print("Use: evaluator = TruncationEvaluator(your_backend)")
    print("     results = evaluator.run_comprehensive_evaluation()")
    print("     print(evaluator.generate_report(results))")
