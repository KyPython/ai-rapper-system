"""
Evaluation System (LBCM - Lyric Battle Criteria Metrics)
Objective scoring for lyric quality
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class LyricMetrics:
    """Comprehensive lyric evaluation metrics"""
    # Core metrics
    rhyme_density: float  # 0-1 score
    syllable_consistency: float  # 0-1 score
    sentiment_score: float  # -1 to 1
    uniqueness: float  # 0-1 score
    complexity: float  # 0-1 score
    
    # Detailed breakdowns
    total_lines: int
    rhyme_schemes: List[str]
    avg_syllables_per_line: float
    vocabulary_size: int
    multisyllabic_rhymes: int
    
    # Battle-specific
    punchline_count: int
    metaphor_count: int
    flow_consistency: float
    
    def overall_score(self) -> float:
        """Calculate weighted overall score"""
        return (
            self.rhyme_density * 0.25 +
            self.syllable_consistency * 0.15 +
            (self.sentiment_score + 1) / 2 * 0.10 +  # Normalize to 0-1
            self.uniqueness * 0.20 +
            self.complexity * 0.15 +
            self.flow_consistency * 0.15
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score(), 3),
            "rhyme_density": round(self.rhyme_density, 3),
            "syllable_consistency": round(self.syllable_consistency, 3),
            "sentiment_score": round(self.sentiment_score, 3),
            "uniqueness": round(self.uniqueness, 3),
            "complexity": round(self.complexity, 3),
            "flow_consistency": round(self.flow_consistency, 3),
            "total_lines": self.total_lines,
            "rhyme_schemes": self.rhyme_schemes,
            "avg_syllables_per_line": round(self.avg_syllables_per_line, 2),
            "vocabulary_size": self.vocabulary_size,
            "multisyllabic_rhymes": self.multisyllabic_rhymes,
            "punchline_count": self.punchline_count,
            "metaphor_count": self.metaphor_count,
        }


class LyricEvaluator:
    """Evaluate lyric quality using objective metrics"""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path
        self.previous_lyrics = []  # For uniqueness checking
        
        # Initialize NLP tools
        self._init_nlp_tools()
        
        # Load previous lyrics for comparison
        if database_path:
            self._load_previous_lyrics()
    
    def _init_nlp_tools(self):
        """Initialize NLP libraries"""
        try:
            import nltk
            # Download required NLTK data if not present
            for package in ['cmudict', 'punkt', 'averaged_perceptron_tagger']:
                try:
                    nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}' if package == 'cmudict' else f'taggers/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
            
            from nltk.corpus import cmudict
            self.cmu_dict = cmudict.dict()
            
        except Exception as e:
            logger.warning(f"⚠️  NLTK initialization warning: {e}")
            self.cmu_dict = {}
        
        try:
            import pronouncing
            self.pronouncing = pronouncing
        except ImportError:
            logger.warning("⚠️  pronouncing library not available")
            self.pronouncing = None
        
        try:
            import syllables
            self.syllables = syllables
        except ImportError:
            logger.warning("⚠️  syllables library not available")
            self.syllables = None
    
    def _load_previous_lyrics(self):
        """Load previous lyrics for uniqueness checking"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute("SELECT lyrics FROM generations ORDER BY timestamp DESC LIMIT 100")
            self.previous_lyrics = [row[0] for row in cursor.fetchall()]
            conn.close()
            logger.info(f"✅ Loaded {len(self.previous_lyrics)} previous lyrics for comparison")
        except Exception as e:
            logger.warning(f"⚠️  Could not load previous lyrics: {e}")
    
    def evaluate(self, lyrics: str, previous_work: Optional[List[str]] = None) -> LyricMetrics:
        """
        Comprehensive evaluation of lyrics
        
        Args:
            lyrics: The lyrics to evaluate
            previous_work: Optional list of previous lyrics for uniqueness comparison
            
        Returns:
            LyricMetrics with all scores
        """
        lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
        
        if not lines:
            raise ValueError("No lyrics provided for evaluation")
        
        # Calculate all metrics
        rhyme_density = self._calculate_rhyme_density(lines)
        syllable_consistency = self._calculate_syllable_consistency(lines)
        sentiment = self._calculate_sentiment(lyrics)
        uniqueness = self._calculate_uniqueness(lyrics, previous_work or self.previous_lyrics)
        complexity = self._calculate_complexity(lyrics)
        
        # Rhyme scheme analysis
        rhyme_schemes = self._analyze_rhyme_scheme(lines)
        multisyllabic_rhymes = self._count_multisyllabic_rhymes(lines)
        
        # Syllable analysis
        syllables_per_line = [self._count_syllables(line) for line in lines]
        avg_syllables = sum(syllables_per_line) / len(syllables_per_line) if syllables_per_line else 0
        
        # Vocabulary analysis
        words = re.findall(r'\b\w+\b', lyrics.lower())
        vocabulary_size = len(set(words))
        
        # Battle-specific metrics
        punchlines = self._detect_punchlines(lines)
        metaphors = self._detect_metaphors(lyrics)
        flow = self._calculate_flow_consistency(syllables_per_line)
        
        return LyricMetrics(
            rhyme_density=rhyme_density,
            syllable_consistency=syllable_consistency,
            sentiment_score=sentiment,
            uniqueness=uniqueness,
            complexity=complexity,
            total_lines=len(lines),
            rhyme_schemes=rhyme_schemes,
            avg_syllables_per_line=avg_syllables,
            vocabulary_size=vocabulary_size,
            multisyllabic_rhymes=multisyllabic_rhymes,
            punchline_count=punchlines,
            metaphor_count=metaphors,
            flow_consistency=flow,
        )
    
    def _calculate_rhyme_density(self, lines: List[str]) -> float:
        """Calculate what percentage of lines rhyme with another line"""
        if len(lines) < 2:
            return 0.0
        
        rhyming_lines = 0
        
        for i, line1 in enumerate(lines):
            last_word1 = self._get_last_word(line1)
            if not last_word1:
                continue
            
            for j, line2 in enumerate(lines):
                if i >= j:
                    continue
                last_word2 = self._get_last_word(line2)
                if not last_word2:
                    continue
                
                if self._words_rhyme(last_word1, last_word2):
                    rhyming_lines += 1
                    break
        
        return min(rhyming_lines / len(lines), 1.0)
    
    def _calculate_syllable_consistency(self, lines: List[str]) -> float:
        """Calculate how consistent syllable counts are across lines"""
        if len(lines) < 2:
            return 1.0
        
        syllables = [self._count_syllables(line) for line in lines]
        
        if not syllables:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean = sum(syllables) / len(syllables)
        if mean == 0:
            return 0.0
        
        variance = sum((s - mean) ** 2 for s in syllables) / len(syllables)
        std_dev = variance ** 0.5
        cv = std_dev / mean
        
        # Convert to 0-1 score (lower CV = higher score)
        consistency = max(0, 1 - (cv / 0.5))  # 0.5 CV = 0 score
        return min(consistency, 1.0)
    
    def _calculate_sentiment(self, lyrics: str) -> float:
        """Calculate sentiment score using VADER"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(lyrics)
            return scores['compound']  # -1 to 1
        except Exception as e:
            logger.warning(f"⚠️  Sentiment analysis failed: {e}")
            return 0.0
    
    def _calculate_uniqueness(self, lyrics: str, previous_work: List[str]) -> float:
        """Calculate how unique lyrics are compared to previous work"""
        if not previous_work:
            return 1.0
        
        current_words = set(re.findall(r'\b\w+\b', lyrics.lower()))
        
        if not current_words:
            return 0.0
        
        # Compare to previous work
        max_overlap = 0
        for prev_lyrics in previous_work:
            prev_words = set(re.findall(r'\b\w+\b', prev_lyrics.lower()))
            if prev_words:
                overlap = len(current_words & prev_words) / len(current_words)
                max_overlap = max(max_overlap, overlap)
        
        return 1.0 - max_overlap
    
    def _calculate_complexity(self, lyrics: str) -> float:
        """Calculate linguistic complexity"""
        words = re.findall(r'\b\w+\b', lyrics.lower())
        
        if not words:
            return 0.0
        
        # Metrics for complexity
        avg_word_length = sum(len(w) for w in words) / len(words)
        unique_ratio = len(set(words)) / len(words)
        
        # Normalize to 0-1
        length_score = min(avg_word_length / 10, 1.0)  # 10+ chars = max
        uniqueness_score = unique_ratio
        
        return (length_score + uniqueness_score) / 2
    
    def _analyze_rhyme_scheme(self, lines: List[str]) -> List[str]:
        """Detect rhyme scheme patterns (AABB, ABAB, etc.)"""
        if len(lines) < 2:
            return []
        
        last_words = [self._get_last_word(line) for line in lines]
        scheme = []
        label_map = {}
        current_label = 'A'
        
        for word in last_words:
            if not word:
                scheme.append('-')
                continue
            
            # Check if this word rhymes with a previous word
            found_rhyme = False
            for prev_word, label in label_map.items():
                if self._words_rhyme(word, prev_word):
                    scheme.append(label)
                    found_rhyme = True
                    break
            
            if not found_rhyme:
                label_map[word] = current_label
                scheme.append(current_label)
                current_label = chr(ord(current_label) + 1)
        
        return scheme
    
    def _count_multisyllabic_rhymes(self, lines: List[str]) -> int:
        """Count rhymes that involve multiple syllables"""
        count = 0
        
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines):
                if i >= j:
                    continue
                
                word1 = self._get_last_word(line1)
                word2 = self._get_last_word(line2)
                
                if word1 and word2 and self._words_rhyme(word1, word2):
                    if self._count_syllables(word1) > 1:
                        count += 1
        
        return count
    
    def _detect_punchlines(self, lines: List[str]) -> int:
        """Detect potential punchlines (heuristic-based)"""
        punchline_indicators = [
            r'\!',  # Exclamation marks
            r'\?',  # Questions
            r'\b(like|so|that\'s|call me|they call)\b.*\b(that|it)\b',  # Comparisons
        ]
        
        count = 0
        for line in lines:
            for pattern in punchline_indicators:
                if re.search(pattern, line, re.IGNORECASE):
                    count += 1
                    break
        
        return count
    
    def _detect_metaphors(self, lyrics: str) -> int:
        """Detect potential metaphors (heuristic-based)"""
        metaphor_indicators = [
            r'\b(like|as|than)\b',  # Similes
            r'\bis\b.*\ba\b',  # "X is a Y"
            r'\b(become|became|turn into|transform)\b',  # Transformations
        ]
        
        count = 0
        for pattern in metaphor_indicators:
            matches = re.findall(pattern, lyrics, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def _calculate_flow_consistency(self, syllables_per_line: List[int]) -> float:
        """Calculate how well the flow is maintained"""
        if len(syllables_per_line) < 4:
            return 0.5
        
        # Check for patterns (e.g., bars of 4)
        bar_size = 4
        consistency_scores = []
        
        for i in range(0, len(syllables_per_line) - bar_size + 1, bar_size):
            bar = syllables_per_line[i:i+bar_size]
            mean = sum(bar) / len(bar)
            variance = sum((s - mean) ** 2 for s in bar) / len(bar)
            consistency_scores.append(1.0 / (1.0 + variance))
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    
    def _get_last_word(self, line: str) -> Optional[str]:
        """Extract last meaningful word from line"""
        words = re.findall(r'\b\w+\b', line)
        return words[-1].lower() if words else None
    
    def _words_rhyme(self, word1: str, word2: str) -> bool:
        """Check if two words rhyme"""
        if word1 == word2:
            return False
        
        if self.pronouncing:
            try:
                rhymes1 = self.pronouncing.rhymes(word1)
                return word2 in rhymes1
            except Exception:
                pass
        
        # Fallback: simple ending match
        if len(word1) >= 2 and len(word2) >= 2:
            return word1[-2:] == word2[-2:]
        
        return False
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text"""
        words = re.findall(r'\b\w+\b', text.lower())
        total = 0
        
        for word in words:
            if self.syllables:
                try:
                    count = self.syllables.estimate(word)
                    total += count
                    continue
                except Exception:
                    pass
            
            # Fallback: CMU dict
            if word in self.cmu_dict:
                pronunciations = self.cmu_dict[word]
                if pronunciations:
                    # Count vowel sounds in first pronunciation
                    total += len([p for p in pronunciations[0] if p[-1].isdigit()])
                    continue
            
            # Simple fallback: count vowel groups
            total += max(1, len(re.findall(r'[aeiouy]+', word)))
        
        return total


def compare_lyrics(lyrics_list: List[str], evaluator: Optional[LyricEvaluator] = None) -> Dict[str, Any]:
    """
    Compare multiple lyrics generations
    Useful for ensemble mode
    
    Args:
        lyrics_list: List of lyrics to compare
        evaluator: Optional LyricEvaluator instance
        
    Returns:
        Comparison results with rankings
    """
    if not evaluator:
        evaluator = LyricEvaluator()
    
    results = []
    for i, lyrics in enumerate(lyrics_list):
        metrics = evaluator.evaluate(lyrics)
        results.append({
            "index": i,
            "lyrics": lyrics,
            "metrics": metrics.to_dict(),
            "overall_score": metrics.overall_score(),
        })
    
    # Sort by overall score
    results.sort(key=lambda x: x["overall_score"], reverse=True)
    
    return {
        "rankings": results,
        "best": results[0],
        "comparison": {
            "count": len(results),
            "avg_score": sum(r["overall_score"] for r in results) / len(results),
            "score_range": (results[-1]["overall_score"], results[0]["overall_score"]),
        }
    }
