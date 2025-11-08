"""
Sentiment Analysis Module (PHASE 0 - Data & Ethos)
Open-source sentiment analysis using VADER and TextBlob
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze sentiment and emotional content of lyrics
    Uses VADER for social media-style text and TextBlob for general sentiment
    """
    
    def __init__(self):
        self._init_analyzers()
    
    def _init_analyzers(self):
        """Initialize sentiment analysis tools"""
        # VADER - great for social media and slang
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            logger.info("✅ VADER sentiment analyzer initialized")
        except ImportError:
            logger.warning("⚠️  VADER not available. Install: pip install vaderSentiment")
            self.vader = None
        
        # TextBlob - general sentiment
        try:
            from textblob import TextBlob
            self.TextBlob = TextBlob
            logger.info("✅ TextBlob sentiment analyzer initialized")
        except ImportError:
            logger.warning("⚠️  TextBlob not available. Install: pip install textblob")
            self.TextBlob = None
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and analysis
        """
        results = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # VADER analysis
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            results["vader"] = {
                "compound": vader_scores["compound"],  # Overall: -1 to 1
                "positive": vader_scores["pos"],
                "neutral": vader_scores["neu"],
                "negative": vader_scores["neg"],
                "intensity": abs(vader_scores["compound"]),
            }
        
        # TextBlob analysis
        if self.TextBlob:
            blob = self.TextBlob(text)
            results["textblob"] = {
                "polarity": blob.sentiment.polarity,  # -1 to 1
                "subjectivity": blob.sentiment.subjectivity,  # 0 to 1
            }
        
        # Combined overall sentiment
        if self.vader and self.TextBlob:
            results["overall"] = {
                "sentiment": (vader_scores["compound"] + blob.sentiment.polarity) / 2,
                "confidence": self._calculate_confidence(vader_scores, blob.sentiment),
                "classification": self._classify_sentiment(
                    (vader_scores["compound"] + blob.sentiment.polarity) / 2
                ),
            }
        elif self.vader:
            results["overall"] = {
                "sentiment": vader_scores["compound"],
                "classification": self._classify_sentiment(vader_scores["compound"]),
            }
        elif self.TextBlob:
            results["overall"] = {
                "sentiment": blob.sentiment.polarity,
                "classification": self._classify_sentiment(blob.sentiment.polarity),
            }
        
        return results
    
    def analyze_motivational_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze motivational and confidence-building aspects
        Specifically for Data & Ethos module
        
        Args:
            text: Text to analyze
            
        Returns:
            Motivational analysis results
        """
        analysis = self.analyze(text)
        
        # Motivational keywords
        motivational_keywords = {
            "confidence": ["confident", "strong", "power", "winner", "best", "king", "queen", "boss"],
            "resilience": ["overcome", "rise", "fight", "persist", "survive", "endure", "bounce back"],
            "ambition": ["dream", "goal", "achieve", "success", "grind", "hustle", "climb", "top"],
            "defiance": ["never", "won't stop", "can't break", "unbreakable", "unstoppable"],
            "pride": ["proud", "earned", "deserve", "worth", "respect", "honor"],
        }
        
        text_lower = text.lower()
        keyword_matches = {}
        
        for category, keywords in motivational_keywords.items():
            matches = [kw for kw in keywords if kw in text_lower]
            keyword_matches[category] = {
                "matches": matches,
                "count": len(matches),
            }
        
        total_matches = sum(cat["count"] for cat in keyword_matches.values())
        
        analysis["motivational"] = {
            "categories": keyword_matches,
            "total_keywords": total_matches,
            "motivational_density": total_matches / len(text.split()) if text.split() else 0,
            "dominant_theme": max(keyword_matches.items(), key=lambda x: x[1]["count"])[0] if total_matches > 0 else "neutral",
        }
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        return [self.analyze(text) for text in texts]
    
    def _calculate_confidence(self, vader_scores: Dict, textblob_sentiment) -> float:
        """Calculate confidence in sentiment classification"""
        # Higher intensity and agreement = higher confidence
        vader_intensity = abs(vader_scores["compound"])
        textblob_intensity = abs(textblob_sentiment.polarity)
        
        # Check agreement
        agreement = 1.0 if (vader_scores["compound"] * textblob_sentiment.polarity) >= 0 else 0.5
        
        return ((vader_intensity + textblob_intensity) / 2) * agreement
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment score into category"""
        if score >= 0.5:
            return "very_positive"
        elif score >= 0.1:
            return "positive"
        elif score >= -0.1:
            return "neutral"
        elif score >= -0.5:
            return "negative"
        else:
            return "very_negative"


class EthosDataManager:
    """
    Manage motivational content and ethos data
    PHASE 0 - Data & Ethos Module
    """
    
    def __init__(self, data_path: str = "./data/ethos_data.json"):
        self.data_path = data_path
        self.analyzer = SentimentAnalyzer()
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load existing ethos data"""
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info("No existing ethos data found, creating new dataset")
            return {
                "motivational_quotes": [],
                "battle_phrases": [],
                "confidence_builders": [],
                "analyzed_content": [],
            }
        except Exception as e:
            logger.error(f"Error loading ethos data: {e}")
            return {}
    
    def _save_data(self):
        """Save ethos data"""
        try:
            import os
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"✅ Ethos data saved to {self.data_path}")
        except Exception as e:
            logger.error(f"Error saving ethos data: {e}")
    
    def add_motivational_content(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        manual_sentiment: Optional[str] = None,
    ):
        """
        Add motivational content to database
        
        Args:
            content: The motivational text
            category: Category (motivational_quotes, battle_phrases, confidence_builders)
            tags: Optional tags for categorization
            manual_sentiment: Optional manual sentiment override
        """
        # Analyze content
        analysis = self.analyzer.analyze_motivational_content(content)
        
        entry = {
            "content": content,
            "category": category,
            "tags": tags or [],
            "analysis": analysis,
            "manual_sentiment": manual_sentiment,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Add to appropriate category
        if category in self.data:
            self.data[category].append(entry)
        
        # Add to analyzed content
        self.data["analyzed_content"].append(entry)
        
        self._save_data()
        logger.info(f"✅ Added motivational content to {category}")
    
    def get_motivational_prompt(self, mood: Optional[str] = None) -> str:
        """
        Get motivational content for prompt enhancement
        
        Args:
            mood: Optional mood filter (confident, aggressive, resilient, etc.)
            
        Returns:
            Formatted motivational prompt
        """
        content_pieces = []
        
        for category in ["motivational_quotes", "battle_phrases", "confidence_builders"]:
            if category in self.data and self.data[category]:
                # Filter by mood if specified
                items = self.data[category]
                if mood:
                    items = [
                        item for item in items
                        if mood.lower() in [t.lower() for t in item.get("tags", [])]
                    ]
                
                if items:
                    import random
                    content_pieces.append(random.choice(items)["content"])
        
        if not content_pieces:
            return "Write with confidence and authenticity."
        
        return "\n".join(content_pieces)
    
    def export_csv_template(self, output_path: str = "./data/ethos_template.csv"):
        """
        Export CSV template for manual tagging
        
        Args:
            output_path: Path to save CSV template
        """
        try:
            import csv
            import os
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "content",
                    "category",
                    "tags",
                    "manual_sentiment",
                    "notes"
                ])
                
                # Add example rows
                writer.writerow([
                    "I'm the definition of what confidence is",
                    "confidence_builders",
                    "confident,aggressive",
                    "very_positive",
                    "Example entry"
                ])
                writer.writerow([
                    "Never back down, always stand tall",
                    "motivational_quotes",
                    "resilient,defiant",
                    "positive",
                    "Example entry"
                ])
            
            logger.info(f"✅ CSV template exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV template: {e}")
    
    def import_csv(self, csv_path: str):
        """
        Import manually tagged content from CSV
        
        Args:
            csv_path: Path to CSV file
        """
        try:
            import csv
            
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["content"] and row["category"]:
                        tags = [t.strip() for t in row["tags"].split(",")] if row["tags"] else []
                        self.add_motivational_content(
                            content=row["content"],
                            category=row["category"],
                            tags=tags,
                            manual_sentiment=row.get("manual_sentiment"),
                        )
            
            logger.info(f"✅ CSV data imported from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error importing CSV: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about ethos data"""
        stats = {
            "total_entries": sum(len(self.data.get(cat, [])) for cat in ["motivational_quotes", "battle_phrases", "confidence_builders"]),
            "categories": {},
        }
        
        for category in ["motivational_quotes", "battle_phrases", "confidence_builders"]:
            if category in self.data:
                stats["categories"][category] = {
                    "count": len(self.data[category]),
                    "avg_sentiment": sum(
                        item["analysis"].get("overall", {}).get("sentiment", 0)
                        for item in self.data[category]
                    ) / len(self.data[category]) if self.data[category] else 0,
                }
        
        return stats
