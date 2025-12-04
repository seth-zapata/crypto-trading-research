"""
Reddit Data Sources

Abstraction layer for fetching Reddit data, supporting both:
1. Public JSON endpoints (no auth required, ~10 req/min)
2. PRAW OAuth (requires approval, 100 req/min) - placeholder for future

Author: Claude Opus 4.5
Date: 2024-12-04
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class RedditDataSource(ABC):
    """Abstract interface for Reddit data - swap implementations via config."""

    @abstractmethod
    async def fetch_subreddit_posts(
        self, subreddit: str, limit: int = 100, sort: str = 'hot'
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            limit: Maximum number of posts to fetch (max 100)
            sort: Sort order ('hot', 'new', 'top', 'rising')

        Returns:
            List of normalized post data
        """
        pass

    @abstractmethod
    async def search_subreddit(
        self, subreddit: str, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search within a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            query: Search query
            limit: Maximum results

        Returns:
            List of normalized post data
        """
        pass

    @abstractmethod
    async def fetch_comments(
        self, subreddit: str, post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch comments for a post.

        Args:
            subreddit: Subreddit name
            post_id: Post ID
            limit: Maximum comments

        Returns:
            List of normalized comment data
        """
        pass


class RedditPublicJSON(RedditDataSource):
    """
    Reddit data via public .json endpoints.

    No authentication required, but rate limited to ~10 requests/minute.
    Uses 2.5 second delays between requests to stay within limits.
    """

    # Crypto-focused subreddits for sentiment analysis
    CRYPTO_SUBREDDITS = [
        'cryptocurrency',
        'bitcoin',
        'ethtrader',
        'cryptomarkets',
        'btc',
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Reddit public JSON client.

        Args:
            config: Optional config dict with:
                - reddit_user_agent: Custom user agent string
                - request_delay: Seconds between requests (default 2.5)
        """
        config = config or {}
        self.base_url = "https://www.reddit.com"
        self.user_agent = config.get(
            'reddit_user_agent',
            'CryptoSentimentAnalyzer/1.0 (educational research project)'
        )
        self.headers = {'User-Agent': self.user_agent}
        self.delay = config.get('request_delay', 2.5)
        self._last_request_time: Optional[float] = None

        logger.info(f"RedditPublicJSON initialized (delay={self.delay}s)")

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        if self._last_request_time is not None:
            elapsed = asyncio.get_event_loop().time() - self._last_request_time
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    async def _fetch_json(self, url: str) -> Dict[str, Any]:
        """
        Fetch JSON from Reddit with rate limiting.

        Args:
            url: Full URL to fetch

        Returns:
            JSON response as dict

        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid JSON
        """
        await self._rate_limit()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        return await self._fetch_json(url)

                    if response.status != 200:
                        logger.error(f"Reddit API error: {response.status}")
                        return {'data': {'children': []}}

                    return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def _normalize_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Reddit post data to standard format.

        Args:
            post_data: Raw post data from Reddit API

        Returns:
            Normalized post dict
        """
        return {
            'id': post_data.get('id', ''),
            'title': post_data.get('title', ''),
            'body': post_data.get('selftext', ''),
            'score': post_data.get('score', 0),
            'upvote_ratio': post_data.get('upvote_ratio', 0.5),
            'num_comments': post_data.get('num_comments', 0),
            'created_utc': post_data.get('created_utc', 0),
            'created_dt': datetime.fromtimestamp(post_data.get('created_utc', 0)),
            'subreddit': post_data.get('subreddit', ''),
            'author': post_data.get('author', '[deleted]'),
            'url': post_data.get('url', ''),
            'permalink': f"https://reddit.com{post_data.get('permalink', '')}",
            'is_self': post_data.get('is_self', True),
            'link_flair_text': post_data.get('link_flair_text', ''),
        }

    def _normalize_comment(self, comment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Reddit comment data to standard format.

        Args:
            comment_data: Raw comment data from Reddit API

        Returns:
            Normalized comment dict
        """
        return {
            'id': comment_data.get('id', ''),
            'body': comment_data.get('body', ''),
            'score': comment_data.get('score', 0),
            'created_utc': comment_data.get('created_utc', 0),
            'created_dt': datetime.fromtimestamp(comment_data.get('created_utc', 0)),
            'author': comment_data.get('author', '[deleted]'),
            'parent_id': comment_data.get('parent_id', ''),
            'is_submitter': comment_data.get('is_submitter', False),
        }

    async def fetch_subreddit_posts(
        self, subreddit: str, limit: int = 100, sort: str = 'hot'
    ) -> List[Dict[str, Any]]:
        """Fetch posts from a subreddit."""
        limit = min(limit, 100)  # Reddit caps at 100 per request
        url = f"{self.base_url}/r/{subreddit}/{sort}.json?limit={limit}"

        logger.debug(f"Fetching r/{subreddit}/{sort} (limit={limit})")

        data = await self._fetch_json(url)
        posts = []

        for child in data.get('data', {}).get('children', []):
            if child.get('kind') == 't3':  # t3 = post/link
                posts.append(self._normalize_post(child['data']))

        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        return posts

    async def search_subreddit(
        self, subreddit: str, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search within a subreddit."""
        limit = min(limit, 100)
        # URL encode the query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        url = (
            f"{self.base_url}/r/{subreddit}/search.json"
            f"?q={encoded_query}&restrict_sr=1&limit={limit}&sort=relevance"
        )

        logger.debug(f"Searching r/{subreddit} for '{query}'")

        data = await self._fetch_json(url)
        posts = []

        for child in data.get('data', {}).get('children', []):
            if child.get('kind') == 't3':
                posts.append(self._normalize_post(child['data']))

        logger.info(f"Found {len(posts)} posts matching '{query}' in r/{subreddit}")
        return posts

    async def fetch_comments(
        self, subreddit: str, post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch comments for a post."""
        url = f"{self.base_url}/r/{subreddit}/comments/{post_id}.json?limit={limit}"

        logger.debug(f"Fetching comments for post {post_id}")

        data = await self._fetch_json(url)
        comments = []

        # Response is [post, comments] array
        if len(data) >= 2:
            comment_data = data[1].get('data', {}).get('children', [])
            for child in comment_data:
                if child.get('kind') == 't1':  # t1 = comment
                    comments.append(self._normalize_comment(child['data']))

        logger.info(f"Fetched {len(comments)} comments for post {post_id}")
        return comments

    async def fetch_multiple_subreddits(
        self, subreddits: Optional[List[str]] = None, limit_per_sub: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from multiple crypto subreddits.

        Args:
            subreddits: List of subreddit names (defaults to CRYPTO_SUBREDDITS)
            limit_per_sub: Posts to fetch per subreddit

        Returns:
            Combined list of posts from all subreddits
        """
        subreddits = subreddits or self.CRYPTO_SUBREDDITS
        all_posts = []

        for subreddit in subreddits:
            try:
                posts = await self.fetch_subreddit_posts(
                    subreddit, limit=limit_per_sub, sort='hot'
                )
                all_posts.extend(posts)
            except Exception as e:
                logger.warning(f"Failed to fetch r/{subreddit}: {e}")
                continue

        # Sort by created time (newest first)
        all_posts.sort(key=lambda x: x['created_utc'], reverse=True)

        logger.info(f"Fetched {len(all_posts)} total posts from {len(subreddits)} subreddits")
        return all_posts


class RedditPRAW(RedditDataSource):
    """
    Reddit data via PRAW OAuth.

    Requires API approval (as of Nov 2025 Responsible Builder Policy).
    This is a placeholder for future use when approval is obtained.

    Rate limit: ~100 requests/minute with OAuth.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PRAW client.

        Args:
            config: Config dict with reddit_client_id, reddit_client_secret, reddit_user_agent

        Raises:
            ImportError: If PRAW is not installed
            ValueError: If required config is missing
        """
        try:
            import praw
        except ImportError:
            raise ImportError(
                "PRAW not installed. Install with: pip install praw"
            )

        if not config.get('reddit_client_id') or not config.get('reddit_client_secret'):
            raise ValueError("reddit_client_id and reddit_client_secret required")

        self.reddit = praw.Reddit(
            client_id=config['reddit_client_id'],
            client_secret=config['reddit_client_secret'],
            user_agent=config.get('reddit_user_agent', 'CryptoSentiment/1.0')
        )

        logger.info("RedditPRAW initialized with OAuth")

    def _normalize_post(self, post) -> Dict[str, Any]:
        """Normalize PRAW Submission to standard format."""
        return {
            'id': post.id,
            'title': post.title,
            'body': post.selftext,
            'score': post.score,
            'upvote_ratio': post.upvote_ratio,
            'num_comments': post.num_comments,
            'created_utc': post.created_utc,
            'created_dt': datetime.fromtimestamp(post.created_utc),
            'subreddit': post.subreddit.display_name,
            'author': str(post.author) if post.author else '[deleted]',
            'url': post.url,
            'permalink': f"https://reddit.com{post.permalink}",
            'is_self': post.is_self,
            'link_flair_text': post.link_flair_text or '',
        }

    def _normalize_comment(self, comment) -> Dict[str, Any]:
        """Normalize PRAW Comment to standard format."""
        return {
            'id': comment.id,
            'body': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
            'created_dt': datetime.fromtimestamp(comment.created_utc),
            'author': str(comment.author) if comment.author else '[deleted]',
            'parent_id': comment.parent_id,
            'is_submitter': comment.is_submitter,
        }

    async def fetch_subreddit_posts(
        self, subreddit: str, limit: int = 100, sort: str = 'hot'
    ) -> List[Dict[str, Any]]:
        """Fetch posts using PRAW."""
        sub = self.reddit.subreddit(subreddit)

        if sort == 'hot':
            posts = sub.hot(limit=limit)
        elif sort == 'new':
            posts = sub.new(limit=limit)
        elif sort == 'top':
            posts = sub.top(limit=limit)
        elif sort == 'rising':
            posts = sub.rising(limit=limit)
        else:
            posts = sub.hot(limit=limit)

        return [self._normalize_post(p) for p in posts]

    async def search_subreddit(
        self, subreddit: str, query: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search subreddit using PRAW."""
        sub = self.reddit.subreddit(subreddit)
        posts = sub.search(query, limit=limit)
        return [self._normalize_post(p) for p in posts]

    async def fetch_comments(
        self, subreddit: str, post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch comments using PRAW."""
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=0)  # Skip "load more" links
        comments = list(submission.comments.list())[:limit]
        return [self._normalize_comment(c) for c in comments]


def get_reddit_source(config: Optional[Dict[str, Any]] = None) -> RedditDataSource:
    """
    Factory function - returns appropriate Reddit data source based on config.

    If config has reddit_use_oauth=True and valid credentials, returns PRAW client.
    Otherwise returns public JSON client (no auth required).

    Args:
        config: Optional configuration dict

    Returns:
        RedditDataSource implementation
    """
    config = config or {}

    if config.get('reddit_use_oauth') and config.get('reddit_client_id'):
        try:
            return RedditPRAW(config)
        except (ImportError, ValueError) as e:
            logger.warning(f"Failed to init PRAW ({e}), falling back to public JSON")
            return RedditPublicJSON(config)

    return RedditPublicJSON(config)


if __name__ == "__main__":
    # Quick test
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def test():
        source = get_reddit_source()

        # Fetch from r/cryptocurrency
        posts = await source.fetch_subreddit_posts('cryptocurrency', limit=5)

        print(f"\n=== Top 5 posts from r/cryptocurrency ===")
        for post in posts:
            print(f"\n[{post['score']:>5}] {post['title'][:60]}...")
            print(f"        {post['num_comments']} comments | {post['created_dt']}")

        # Search for Bitcoin
        print(f"\n=== Searching for 'bitcoin price' ===")
        results = await source.search_subreddit('cryptocurrency', 'bitcoin price', limit=3)
        for post in results:
            print(f"[{post['score']:>5}] {post['title'][:60]}...")

    asyncio.run(test())
