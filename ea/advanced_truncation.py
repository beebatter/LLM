"""
Advanced truncation strategies for CrossEncoder input processing.
Addresses the problem of sequence length limits while preserving semantic information.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

class TruncationStrategy(Enum):
    """Different strategies for handling long sequences"""
    SIMPLE = "simple"           # Direct truncation from end
    BALANCED = "balanced"       # Proportional allocation
    SLIDING_WINDOW = "sliding"  # Beginning + end preservation
    SEMANTIC_AWARE = "semantic" # Logic-formula aware truncation
    MULTI_SEGMENT = "multi"     # Score multiple segments and combine

class LogicAwareTruncator:
    """Smart truncation for logical formulas"""
    
    def __init__(self, max_len: int = 384):
        self.max_len = max_len
        # Patterns for important logical structures
        self.important_patterns = [
            r'\([^)]*\)',      # Parentheses groups
            r'∀[^.]*\.',       # Universal quantifiers
            r'∃[^.]*\.',       # Existential quantifiers  
            r'\[[^\]]*\]',     # Brackets
            r'[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)',  # Function calls
        ]
    
    def find_logical_boundaries(self, text: str) -> List[int]:
        """Find positions where logical formulas can be safely truncated"""
        boundaries = [0]
        
        # Add boundaries at sentence/clause breaks
        for match in re.finditer(r'[.;]\s+', text):
            boundaries.append(match.end())
        
        # Add boundaries at major logical connectives
        for match in re.finditer(r'\s+(∧|∨|→|↔|&|\|)\s+', text):
            boundaries.append(match.start())
            boundaries.append(match.end())
            
        boundaries.append(len(text))
        return sorted(set(boundaries))
    
    def truncate_semantic(self, conjecture: str, candidate: str, 
                         vocab_encode_fn) -> List[int]:
        """Semantically-aware truncation that preserves logical structure"""
        
        # Try to find natural break points
        conj_boundaries = self.find_logical_boundaries(conjecture)
        cand_boundaries = self.find_logical_boundaries(candidate)
        
        sep_seq = vocab_encode_fn(' [SEP] ')
        overhead = len(sep_seq)
        available = self.max_len - overhead
        
        # Find the best truncation points that preserve most information
        best_score = -1
        best_conj_end = len(conjecture)
        best_cand_end = len(candidate)
        
        for conj_end in conj_boundaries:
            if conj_end == 0:
                continue
            conj_part = conjecture[:conj_end]
            conj_seq = vocab_encode_fn(conj_part)
            
            for cand_end in cand_boundaries:
                if cand_end == 0:
                    continue
                cand_part = candidate[:cand_end]  
                cand_seq = vocab_encode_fn(cand_part)
                
                total_len = len(conj_seq) + len(cand_seq)
                if total_len <= available:
                    # Score based on information retention
                    conj_retention = len(conj_part) / len(conjecture)
                    cand_retention = len(cand_part) / len(candidate)  
                    score = (conj_retention + cand_retention) / 2
                    
                    if score > best_score:
                        best_score = score
                        best_conj_end = conj_end
                        best_cand_end = cand_end
        
        # Build final sequence
        best_conj = conjecture[:best_conj_end]
        best_cand = candidate[:best_cand_end]
        
        conj_seq = vocab_encode_fn(best_conj)
        cand_seq = vocab_encode_fn(best_cand)
        
        return conj_seq + sep_seq + cand_seq

class MultiSegmentScorer:
    """Score multiple segments of long sequences and combine results"""
    
    def __init__(self, base_scorer, max_len: int = 384, num_segments: int = 3):
        self.base_scorer = base_scorer
        self.max_len = max_len
        self.num_segments = num_segments
    
    def score_multi_segment(self, conjecture: str, candidate: str) -> float:
        """Score multiple overlapping segments and combine"""
        full_text = conjecture + ' [SEP] ' + candidate
        full_seq = self.base_scorer.vocab.encode(full_text)
        
        if len(full_seq) <= self.max_len:
            # Single segment scoring
            return self.base_scorer._score_single_pair(conjecture, candidate)
        
        # Create overlapping segments
        segment_scores = []
        step = (len(full_seq) - self.max_len) // (self.num_segments - 1)
        
        for i in range(self.num_segments):
            start = i * step
            end = start + self.max_len
            segment = full_seq[start:end]
            
            # Decode back to text (approximate)
            # This is tricky - we'd need proper decode function
            # For now, score the segment as-is
            segment_score = self.base_scorer._score_sequence_directly(segment)
            segment_scores.append(segment_score)
        
        # Combine scores (could be max, mean, weighted average, etc.)
        return max(segment_scores)  # Take the best segment score

def create_advanced_truncation_config():
    """Configuration for different truncation scenarios"""
    return {
        'default_strategy': TruncationStrategy.BALANCED,
        'fallback_strategy': TruncationStrategy.SIMPLE,
        'enable_semantic_aware': True,
        'enable_multi_segment': False,  # Computationally expensive
        'max_len': 384,
        'preserve_ratio': 0.1,  # Minimum ratio to preserve from each part
        'sliding_window_size': 0.25,  # Fraction for sliding window
    }
