"""
Advanced truncation strategies for improved scoring quality.
Addresses semantic preservation in logical formulas during truncation.
"""

import re
from typing import List, Tuple, Dict, Any

class SemanticTruncator:
    """Logic-aware truncation that preserves semantic structure"""
    
    def __init__(self, max_len: int = 384):
        self.max_len = max_len
        self.quantifier_patterns = [
            r'∀[^.()]*\.',    # Universal quantifiers with scope
            r'∃[^.()]*\.',    # Existential quantifiers with scope
        ]
        self.logical_connectives = ['∧', '∨', '→', '↔', '¬']
        
    def find_safe_truncation_points(self, text: str) -> List[int]:
        """Find positions where truncation won't break logical structure"""
        safe_points = [0, len(text)]
        
        # Add points after complete logical statements
        for match in re.finditer(r'[.;]\s*', text):
            safe_points.append(match.end())
            
        # Add points at major logical breaks
        for conn in self.logical_connectives:
            for match in re.finditer(f'\\s+{re.escape(conn)}\\s+', text):
                safe_points.append(match.start())
                safe_points.append(match.end())
        
        return sorted(set(safe_points))
    
    def score_truncation_quality(self, original: str, truncated: str) -> float:
        """Score the quality of a truncation (0-1, higher is better)"""
        if not truncated.strip():
            return 0.0
            
        # Length preservation
        length_score = len(truncated) / len(original)
        
        # Structure preservation
        structure_score = 1.0
        
        # Check parentheses matching
        orig_parens = original.count('(') - original.count(')')
        trunc_parens = truncated.count('(') - truncated.count(')')
        if orig_parens == 0 and trunc_parens != 0:
            structure_score -= 0.3
        elif abs(trunc_parens) > abs(orig_parens):
            structure_score -= 0.2
            
        # Check if quantifiers are complete
        for pattern in self.quantifier_patterns:
            orig_quant = len(re.findall(pattern, original))
            trunc_quant = len(re.findall(pattern, truncated))
            if trunc_quant < orig_quant:
                structure_score -= 0.1 * (orig_quant - trunc_quant)
        
        # Combine scores
        return 0.7 * length_score + 0.3 * max(0, structure_score)
    
    def smart_truncate_pair(self, conjecture: str, candidate: str, 
                           vocab_encode_fn) -> List[int]:
        """Semantically-aware truncation for conjecture-candidate pairs"""
        
        sep_tokens = vocab_encode_fn(' [SEP] ')
        available = self.max_len - len(sep_tokens)
        
        conj_seq = vocab_encode_fn(conjecture)
        cand_seq = vocab_encode_fn(candidate)
        
        if len(conj_seq) + len(cand_seq) <= available:
            return conj_seq + sep_tokens + cand_seq
        
        # Find best truncation points for both parts
        conj_points = self.find_safe_truncation_points(conjecture)
        cand_points = self.find_safe_truncation_points(candidate)
        
        best_score = -1
        best_conj_end = len(conjecture) // 2  # fallback
        best_cand_end = len(candidate) // 2   # fallback
        
        # Try different allocation ratios
        for ratio in [0.3, 0.4, 0.5, 0.6, 0.7]:
            conj_budget = int(available * ratio)
            cand_budget = available - conj_budget
            
            # Find best truncation points within budget
            conj_end = self._find_best_truncation_in_budget(
                conjecture, conj_points, conj_budget, vocab_encode_fn)
            cand_end = self._find_best_truncation_in_budget(
                candidate, cand_points, cand_budget, vocab_encode_fn)
            
            # Score this combination
            conj_trunc = conjecture[:conj_end]
            cand_trunc = candidate[:cand_end]
            
            conj_quality = self.score_truncation_quality(conjecture, conj_trunc)
            cand_quality = self.score_truncation_quality(candidate, cand_trunc)
            combined_score = (conj_quality + cand_quality) / 2
            
            if combined_score > best_score:
                best_score = combined_score
                best_conj_end = conj_end
                best_cand_end = cand_end
        
        # Build final sequence
        best_conj = conjecture[:best_conj_end]
        best_cand = candidate[:best_cand_end]
        
        final_conj_seq = vocab_encode_fn(best_conj)
        final_cand_seq = vocab_encode_fn(best_cand)
        
        return final_conj_seq + sep_tokens + final_cand_seq
    
    def _find_best_truncation_in_budget(self, text: str, safe_points: List[int],
                                       budget: int, vocab_encode_fn) -> int:
        """Find the best truncation point within token budget"""
        best_end = len(text) // 2  # fallback
        
        for point in reversed(safe_points):
            if point <= len(text):
                test_seq = vocab_encode_fn(text[:point])
                if len(test_seq) <= budget:
                    # This point fits in budget, check quality
                    quality = self.score_truncation_quality(text, text[:point])
                    if quality > 0.3:  # Minimum quality threshold
                        return point
                    else:
                        best_end = point  # Keep as fallback
        
        return best_end

class AdaptiveTruncationBackend:
    """Wrapper that adds adaptive truncation to any CrossEncoder backend"""
    
    def __init__(self, base_backend, enable_semantic=True, enable_statistics=True):
        self.base = base_backend
        self.semantic_truncator = SemanticTruncator() if enable_semantic else None
        self.enable_stats = enable_statistics
        
        if self.enable_stats:
            self.stats = {
                'total_queries': 0,
                'truncated_queries': 0,
                'avg_retention_ratio': 0.0,
                'semantic_vs_simple_improvement': 0.0
            }
    
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """Enhanced scoring with adaptive truncation"""
        if not candidates:
            return []
        
        self.stats['total_queries'] += len(candidates)
        scores = []
        
        for candidate in candidates:
            cand_text = candidate.get('text', '')
            
            # Check if truncation needed
            full_text = conjecture + ' [SEP] ' + cand_text
            full_seq = self.base.vocab.encode(full_text)
            
            if len(full_seq) <= 384:  # No truncation needed
                result = self._score_single_pair(conjecture, cand_text)
            else:
                # Apply adaptive truncation
                self.stats['truncated_queries'] += 1
                
                if self.semantic_truncator:
                    # Use semantic-aware truncation
                    smart_seq = self.semantic_truncator.smart_truncate_pair(
                        conjecture, cand_text, self.base.vocab.encode)
                    result = self._score_sequence_directly(smart_seq)
                else:
                    # Fallback to simple proportional
                    result = self._score_with_simple_truncation(conjecture, cand_text)
                
                # Update statistics
                retention = len(smart_seq) / len(full_seq) if 'smart_seq' in locals() else 0.5
                self._update_retention_stats(retention)
            
            scores.append(result)
        
        return scores
    
    def _score_single_pair(self, conjecture: str, candidate: str) -> float:
        """Score a single pair without truncation"""
        # Delegate to base backend's actual scoring logic
        candidates = [{'text': candidate}]
        result = self.base.score(conjecture, candidates)
        return result[0] if result else 0.0
    
    def _score_sequence_directly(self, seq: List[int]) -> float:
        """Score a pre-tokenized sequence directly"""
        # This would need integration with the actual model scoring
        # For now, return a placeholder
        return 0.5
    
    def _score_with_simple_truncation(self, conjecture: str, candidate: str) -> float:
        """Fallback simple truncation scoring"""
        candidates = [{'text': candidate}]
        return self.base.score(conjecture, candidates)[0]
    
    def _update_retention_stats(self, retention: float):
        """Update retention statistics"""
        if self.enable_stats:
            # Running average update
            n = self.stats['truncated_queries']
            old_avg = self.stats['avg_retention_ratio']
            self.stats['avg_retention_ratio'] = old_avg + (retention - old_avg) / n
    
    def get_truncation_report(self) -> Dict[str, Any]:
        """Get detailed truncation statistics"""
        if not self.enable_stats:
            return {"message": "Statistics disabled"}
        
        total = self.stats['total_queries']
        truncated = self.stats['truncated_queries']
        
        return {
            "total_queries": total,
            "truncated_queries": truncated,
            "truncation_rate": truncated / total if total > 0 else 0,
            "avg_retention_ratio": self.stats['avg_retention_ratio'],
            "estimated_quality_impact": self._estimate_quality_impact()
        }
    
    def _estimate_quality_impact(self) -> str:
        """Estimate the impact on scoring quality"""
        rate = self.stats['truncated_queries'] / max(1, self.stats['total_queries'])
        retention = self.stats['avg_retention_ratio']
        
        if rate < 0.1:
            return "Minimal impact - most queries don't need truncation"
        elif rate < 0.3 and retention > 0.7:
            return "Low impact - good retention rate on truncated queries"
        elif rate < 0.5 and retention > 0.5:
            return "Moderate impact - consider optimizing for longer sequences"
        else:
            return "High impact - significant information loss, recommend model optimization"

# Usage example:
def create_enhanced_backend(base_backend):
    """Create an enhanced backend with adaptive truncation"""
    return AdaptiveTruncationBackend(
        base_backend=base_backend,
        enable_semantic=True,
        enable_statistics=True
    )
