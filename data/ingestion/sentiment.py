"""
Sentiment Analysis Module

Uses FinBERT for financial sentiment analysis of crypto-related social media posts.
Implements CARVS (Credibility-Adjusted Relevance-Volume-Sentiment) scoring
based on academic research for filtered sentiment signals.

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text."""
    text: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float  # Max of the three scores


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment across multiple posts."""
    timestamp: datetime
    num_posts: int
    avg_sentiment_score: float  # -1 to 1 scale
    sentiment_std: float
    bullish_ratio: float  # % positive posts
    bearish_ratio: float  # % negative posts
    volume_weighted_sentiment: float  # Weighted by engagement
    carvs_score: float  # Credibility-adjusted score


class FinBERTAnalyzer:
    """
    Financial sentiment analysis using FinBERT.

    FinBERT is a BERT model fine-tuned on financial text, providing better
    accuracy than general-purpose sentiment models for market-related content.

    Uses ProsusAI/finbert from HuggingFace.

    Example:
        >>> analyzer = FinBERTAnalyzer()
        >>> result = analyzer.analyze("Bitcoin is showing strong momentum!")
        >>> print(f"Sentiment: {result.sentiment} ({result.confidence:.2f})")
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize FinBERT analyzer.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum token length
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install with: "
                "pip install transformers torch"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading FinBERT from {model_name} on {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # FinBERT labels
        self.labels = ['positive', 'negative', 'neutral']

        logger.info("FinBERT loaded successfully")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove Reddit-specific formatting
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)

        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Truncate very long texts
        if len(text) > 1000:
            text = text[:1000]

        return text.strip()

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with scores
        """
        import torch

        text = self._preprocess_text(text)

        if not text:
            return SentimentResult(
                text="",
                sentiment='neutral',
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                confidence=1.0
            )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Map to our labels (FinBERT order: positive, negative, neutral)
        positive_score = float(probs[0])
        negative_score = float(probs[1])
        neutral_score = float(probs[2])

        # Determine sentiment
        max_idx = int(np.argmax(probs))
        sentiment = self.labels[max_idx]
        confidence = float(probs[max_idx])

        return SentimentResult(
            text=text[:100],  # Truncate for storage
            sentiment=sentiment,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            confidence=confidence
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of multiple texts in batches.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResults
        """
        import torch

        results = []

        # Preprocess all texts
        processed = [self._preprocess_text(t) for t in texts]

        # Process in batches
        for i in range(0, len(processed), self.batch_size):
            batch_texts = processed[i:i + self.batch_size]
            original_texts = texts[i:i + self.batch_size]

            # Filter empty texts
            valid_indices = [j for j, t in enumerate(batch_texts) if t]
            valid_texts = [batch_texts[j] for j in valid_indices]

            if not valid_texts:
                # All empty in this batch
                for t in batch_texts:
                    results.append(SentimentResult(
                        text="",
                        sentiment='neutral',
                        positive_score=0.0,
                        negative_score=0.0,
                        neutral_score=1.0,
                        confidence=1.0
                    ))
                continue

            # Tokenize batch
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Process results
            valid_results = []
            for j, prob in enumerate(probs):
                positive_score = float(prob[0])
                negative_score = float(prob[1])
                neutral_score = float(prob[2])
                max_idx = int(np.argmax(prob))

                valid_results.append(SentimentResult(
                    text=valid_texts[j][:100],
                    sentiment=self.labels[max_idx],
                    positive_score=positive_score,
                    negative_score=negative_score,
                    neutral_score=neutral_score,
                    confidence=float(prob[max_idx])
                ))

            # Reconstruct full results with empties
            valid_iter = iter(valid_results)
            for j, t in enumerate(batch_texts):
                if j in valid_indices:
                    results.append(next(valid_iter))
                else:
                    results.append(SentimentResult(
                        text="",
                        sentiment='neutral',
                        positive_score=0.0,
                        negative_score=0.0,
                        neutral_score=1.0,
                        confidence=1.0
                    ))

        return results


class SentimentAggregator:
    """
    Aggregates sentiment from multiple posts into trading signals.

    Implements CARVS scoring:
    - Credibility: Author reputation, post score
    - Relevance: Keyword matching, subreddit quality
    - Volume: Post/comment count
    - Sentiment: FinBERT scores
    """

    # Crypto keywords for relevance scoring
    CRYPTO_KEYWORDS = {
        'high_relevance': [
            'bitcoin', 'btc', 'ethereum', 'eth', 'price', 'market',
            'bull', 'bear', 'pump', 'dump', 'moon', 'crash',
            'buy', 'sell', 'long', 'short', 'bullish', 'bearish'
        ],
        'medium_relevance': [
            'crypto', 'blockchain', 'defi', 'nft', 'altcoin', 'token',
            'wallet', 'exchange', 'binance', 'coinbase', 'trading'
        ],
    }

    # Subreddit quality scores (higher = more reliable)
    SUBREDDIT_QUALITY = {
        'bitcoin': 0.9,
        'cryptocurrency': 0.85,
        'ethtrader': 0.8,
        'btc': 0.75,
        'cryptomarkets': 0.7,
        'cryptomoonshots': 0.3,  # Lower quality
    }

    def __init__(self, analyzer: FinBERTAnalyzer):
        """
        Initialize aggregator.

        Args:
            analyzer: FinBERT analyzer instance
        """
        self.analyzer = analyzer

    def calculate_relevance_score(self, text: str) -> float:
        """
        Calculate relevance score for crypto trading.

        Args:
            text: Post text

        Returns:
            Relevance score 0-1
        """
        text_lower = text.lower()
        score = 0.0

        # High relevance keywords
        for keyword in self.CRYPTO_KEYWORDS['high_relevance']:
            if keyword in text_lower:
                score += 0.15

        # Medium relevance keywords
        for keyword in self.CRYPTO_KEYWORDS['medium_relevance']:
            if keyword in text_lower:
                score += 0.08

        return min(1.0, score)

    def calculate_credibility_score(
        self,
        post: Dict[str, Any]
    ) -> float:
        """
        Calculate credibility score for a post.

        Args:
            post: Post dict with score, upvote_ratio, subreddit, etc.

        Returns:
            Credibility score 0-1
        """
        score = 0.5  # Base score

        # Upvote score contribution
        post_score = post.get('score', 0)
        if post_score > 100:
            score += 0.2
        elif post_score > 20:
            score += 0.1
        elif post_score < 0:
            score -= 0.1

        # Upvote ratio
        ratio = post.get('upvote_ratio', 0.5)
        if ratio > 0.8:
            score += 0.1
        elif ratio < 0.4:
            score -= 0.1

        # Comment count (engagement)
        comments = post.get('num_comments', 0)
        if comments > 50:
            score += 0.1
        elif comments > 10:
            score += 0.05

        # Subreddit quality
        subreddit = post.get('subreddit', '').lower()
        sub_quality = self.SUBREDDIT_QUALITY.get(subreddit, 0.5)
        score = score * 0.7 + sub_quality * 0.3

        return max(0.0, min(1.0, score))

    def analyze_posts(
        self,
        posts: List[Dict[str, Any]]
    ) -> Tuple[List[SentimentResult], List[float], List[float]]:
        """
        Analyze sentiment and calculate scores for posts.

        Args:
            posts: List of post dicts

        Returns:
            Tuple of (sentiment_results, relevance_scores, credibility_scores)
        """
        # Extract text (title + body)
        texts = []
        for post in posts:
            text = post.get('title', '') + ' ' + post.get('body', '')
            texts.append(text)

        # Run sentiment analysis
        sentiments = self.analyzer.analyze_batch(texts)

        # Calculate relevance and credibility
        relevance_scores = [self.calculate_relevance_score(t) for t in texts]
        credibility_scores = [self.calculate_credibility_score(p) for p in posts]

        return sentiments, relevance_scores, credibility_scores

    def aggregate_sentiment(
        self,
        posts: List[Dict[str, Any]],
        timestamp: Optional[datetime] = None
    ) -> AggregatedSentiment:
        """
        Aggregate sentiment across posts with CARVS scoring.

        Args:
            posts: List of post dicts
            timestamp: Timestamp for the aggregation

        Returns:
            AggregatedSentiment with CARVS score
        """
        if not posts:
            return AggregatedSentiment(
                timestamp=timestamp or datetime.now(),
                num_posts=0,
                avg_sentiment_score=0.0,
                sentiment_std=0.0,
                bullish_ratio=0.0,
                bearish_ratio=0.0,
                volume_weighted_sentiment=0.0,
                carvs_score=0.0
            )

        timestamp = timestamp or datetime.now()

        # Analyze all posts
        sentiments, relevances, credibilities = self.analyze_posts(posts)

        # Convert sentiment to -1 to 1 scale
        sentiment_scores = []
        for s in sentiments:
            # positive=1, neutral=0, negative=-1
            score = s.positive_score - s.negative_score
            sentiment_scores.append(score)

        sentiment_scores = np.array(sentiment_scores)

        # Calculate basic metrics
        bullish_count = sum(1 for s in sentiments if s.sentiment == 'positive')
        bearish_count = sum(1 for s in sentiments if s.sentiment == 'negative')

        # Volume weighting (by engagement)
        volumes = np.array([
            max(1, p.get('score', 1) + p.get('num_comments', 0))
            for p in posts
        ])
        volume_weighted = np.average(sentiment_scores, weights=volumes)

        # CARVS score: sentiment weighted by credibility and relevance
        carvs_weights = np.array([
            c * r for c, r in zip(credibilities, relevances)
        ])

        if carvs_weights.sum() > 0:
            carvs_score = np.average(sentiment_scores, weights=carvs_weights)
        else:
            carvs_score = np.mean(sentiment_scores)

        return AggregatedSentiment(
            timestamp=timestamp,
            num_posts=len(posts),
            avg_sentiment_score=float(np.mean(sentiment_scores)),
            sentiment_std=float(np.std(sentiment_scores)),
            bullish_ratio=bullish_count / len(posts),
            bearish_ratio=bearish_count / len(posts),
            volume_weighted_sentiment=float(volume_weighted),
            carvs_score=float(carvs_score)
        )

    def generate_signal(
        self,
        aggregated: AggregatedSentiment,
        min_posts: int = 10,
        carvs_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate trading signal from aggregated sentiment.

        Args:
            aggregated: AggregatedSentiment
            min_posts: Minimum posts required for signal
            carvs_threshold: CARVS score threshold for signal

        Returns:
            Signal dict with strength and direction
        """
        if aggregated.num_posts < min_posts:
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'direction': 'neutral',
                'reason': f'Insufficient posts ({aggregated.num_posts} < {min_posts})'
            }

        carvs = aggregated.carvs_score

        if abs(carvs) < carvs_threshold:
            direction = 'neutral'
            signal = 0.0
        elif carvs > 0:
            direction = 'bullish'
            signal = min(1.0, carvs)
        else:
            direction = 'bearish'
            signal = max(-1.0, carvs)

        # Confidence based on agreement and volume
        agreement = 1.0 - aggregated.sentiment_std
        volume_factor = min(1.0, aggregated.num_posts / 50)
        confidence = agreement * 0.5 + volume_factor * 0.5

        return {
            'signal': signal,
            'confidence': confidence,
            'direction': direction,
            'carvs_score': carvs,
            'bullish_ratio': aggregated.bullish_ratio,
            'bearish_ratio': aggregated.bearish_ratio,
            'num_posts': aggregated.num_posts
        }


if __name__ == "__main__":
    # Quick test
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== FinBERT Sentiment Analysis Demo ===\n")

    # Initialize analyzer
    analyzer = FinBERTAnalyzer()

    # Test texts
    test_texts = [
        "Bitcoin is showing incredible strength! Bull run confirmed!",
        "The market is crashing, I'm selling everything.",
        "BTC traded at $50,000 today with moderate volume.",
        "This is going to the moon! 🚀 Best investment ever!",
        "Crypto winter is here, expect more pain ahead.",
    ]

    print("Individual analysis:")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"\n  Text: {text[:50]}...")
        print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"  Scores: +{result.positive_score:.2f} / -{result.negative_score:.2f} / ={result.neutral_score:.2f}")

    # Batch analysis
    print("\n\nBatch analysis:")
    results = analyzer.analyze_batch(test_texts)
    for text, result in zip(test_texts, results):
        score = result.positive_score - result.negative_score
        print(f"  {result.sentiment:>8} ({score:+.2f}): {text[:40]}...")

    print("\n✓ FinBERT demo complete!")
