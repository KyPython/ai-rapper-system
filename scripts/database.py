"""
Database Schema and Management
SQLite for metrics, generations, and evaluations
"""

import sqlite3
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class Database:
    """SQLite database manager for the rapper system"""
    
    def __init__(self, db_path: str = "./data/metrics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Generations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                tokens_used INTEGER,
                latency_ms REAL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation_id INTEGER,
                overall_score REAL,
                rhyme_density REAL,
                syllable_consistency REAL,
                sentiment_score REAL,
                uniqueness REAL,
                complexity REAL,
                flow_consistency REAL,
                total_lines INTEGER,
                vocabulary_size INTEGER,
                multisyllabic_rhymes INTEGER,
                punchline_count INTEGER,
                metaphor_count INTEGER,
                rhyme_schemes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (generation_id) REFERENCES generations(id)
            )
        """)
        
        # Provider usage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS provider_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                total_requests INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_latency_ms REAL DEFAULT 0,
                avg_latency_ms REAL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_used DATETIME,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_data_path TEXT,
                epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                final_loss REAL,
                duration_seconds INTEGER,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Manual practice log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS manual_practice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                practice_type TEXT NOT NULL,
                content TEXT NOT NULL,
                duration_minutes INTEGER,
                notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Database initialized at {self.db_path}")
    
    def save_generation(
        self,
        prompt: str,
        lyrics: str,
        provider: str,
        model: str,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save a generation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO generations (prompt, lyrics, provider, model, tokens_used, latency_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt,
            lyrics,
            provider,
            model,
            tokens_used,
            latency_ms,
            json.dumps(metadata) if metadata else None,
        ))
        
        generation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return generation_id
    
    def save_evaluation(self, generation_id: int, metrics: Dict[str, Any]):
        """Save evaluation metrics to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO evaluations (
                generation_id, overall_score, rhyme_density, syllable_consistency,
                sentiment_score, uniqueness, complexity, flow_consistency,
                total_lines, vocabulary_size, multisyllabic_rhymes,
                punchline_count, metaphor_count, rhyme_schemes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            generation_id,
            metrics["overall_score"],
            metrics["rhyme_density"],
            metrics["syllable_consistency"],
            metrics["sentiment_score"],
            metrics["uniqueness"],
            metrics["complexity"],
            metrics["flow_consistency"],
            metrics["total_lines"],
            metrics["vocabulary_size"],
            metrics["multisyllabic_rhymes"],
            metrics["punchline_count"],
            metrics["metaphor_count"],
            json.dumps(metrics["rhyme_schemes"]),
        ))
        
        conn.commit()
        conn.close()
    
    def update_provider_usage(
        self,
        provider: str,
        model: str,
        tokens_used: int,
        latency_ms: float,
        success: bool,
    ):
        """Update provider usage statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if entry exists
        cursor.execute("""
            SELECT id, total_requests, total_tokens, total_latency_ms, success_count, failure_count
            FROM provider_usage
            WHERE provider = ? AND model = ?
        """, (provider, model))
        
        row = cursor.fetchone()
        
        if row:
            # Update existing
            cursor.execute("""
                UPDATE provider_usage
                SET total_requests = total_requests + 1,
                    total_tokens = total_tokens + ?,
                    total_latency_ms = total_latency_ms + ?,
                    avg_latency_ms = (total_latency_ms + ?) / (total_requests + 1),
                    success_count = success_count + ?,
                    failure_count = failure_count + ?,
                    last_used = CURRENT_TIMESTAMP
                WHERE provider = ? AND model = ?
            """, (
                tokens_used,
                latency_ms,
                latency_ms,
                1 if success else 0,
                0 if success else 1,
                provider,
                model,
            ))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO provider_usage (
                    provider, model, total_requests, total_tokens, total_latency_ms,
                    avg_latency_ms, success_count, failure_count, last_used
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                provider,
                model,
                tokens_used,
                latency_ms,
                latency_ms,
                1 if success else 0,
                0 if success else 1,
            ))
        
        conn.commit()
        conn.close()
    
    def save_training_session(
        self,
        model_name: str,
        training_data_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        final_loss: float,
        duration_seconds: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Save training session information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_sessions (
                model_name, training_data_path, epochs, batch_size,
                learning_rate, final_loss, duration_seconds, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            training_data_path,
            epochs,
            batch_size,
            learning_rate,
            final_loss,
            duration_seconds,
            json.dumps(metadata) if metadata else None,
        ))
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def log_manual_practice(
        self,
        practice_type: str,
        content: str,
        duration_minutes: int,
        notes: Optional[str] = None,
    ):
        """Log manual writing practice"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO manual_practice (practice_type, content, duration_minutes, notes)
            VALUES (?, ?, ?, ?)
        """, (practice_type, content, duration_minutes, notes))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total generations
        cursor.execute("SELECT COUNT(*) FROM generations")
        stats["total_generations"] = cursor.fetchone()[0]
        
        # Provider breakdown
        cursor.execute("""
            SELECT provider, COUNT(*), AVG(tokens_used), AVG(latency_ms)
            FROM generations
            GROUP BY provider
        """)
        stats["by_provider"] = {
            row[0]: {
                "count": row[1],
                "avg_tokens": row[2] or 0,
                "avg_latency_ms": row[3] or 0,
            }
            for row in cursor.fetchall()
        }
        
        # Average evaluation scores
        cursor.execute("""
            SELECT AVG(overall_score), AVG(rhyme_density), AVG(uniqueness)
            FROM evaluations
        """)
        row = cursor.fetchone()
        stats["avg_scores"] = {
            "overall": row[0] or 0,
            "rhyme_density": row[1] or 0,
            "uniqueness": row[2] or 0,
        }
        
        # Manual practice
        cursor.execute("SELECT COUNT(*) FROM manual_practice")
        stats["manual_practice_sessions"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def get_recent_generations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent generations"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM generations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
