"""
Web Scraper Script
Demonstrates:
- Making HTTP requests with error handling
- Parsing HTML with BeautifulSoup
- Working with environment variables
- Data extraction and transformation
"""
import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ScrapedArticle:
    """Data class to store article information."""
    title: str
    url: str
    summary: str
    tags: List[str]
    word_count: int

class WebScraper:
    """A simple web scraper for extracting article information."""
    
    BASE_URL = "https://blog.cloudera.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a web page."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article links from the blog page."""
        article_links = []
        articles = soup.find_all('article')
        
        for article in articles:
            link = article.find('a', href=True)
            if link:
                full_url = urljoin(self.BASE_URL, link['href'])
                article_links.append(full_url)
                
        return article_links[:5]  # Limit to first 5 articles for demo
    
    def extract_article_details(self, url: str) -> Optional[ScrapedArticle]:
        """Extract detailed information from an article page."""
        soup = self.get_page_content(url)
        if not soup:
            return None
            
        title = soup.find('h1', class_='entry-title')
        content = soup.find('div', class_='entry-content')
        
        if not all([title, content]):
            return None
            
        # Extract tags
        tags_section = soup.find('span', class_='tags-links')
        tags = [a.text.strip() for a in tags_section.find_all('a')] if tags_section else []
        
        # Calculate word count (simple whitespace split)
        text_content = content.get_text()
        word_count = len(text_content.split())
        
        # Extract a summary (first 200 chars of content)
        summary = text_content[:200].strip() + '...' if len(text_content) > 200 else text_content
        
        return ScrapedArticle(
            title=title.text.strip(),
            url=url,
            summary=summary,
            tags=tags,
            word_count=word_count
        )
    
    def scrape_blog(self) -> List[Dict]:
        """Main method to scrape the blog and return article data."""
        print(f"Scraping articles from {self.BASE_URL}...")
        
        # Get the main blog page
        soup = self.get_page_content(self.BASE_URL)
        if not soup:
            print("Failed to retrieve the blog page.")
            return []
        
        # Extract article links
        article_links = self.extract_article_links(soup)
        if not article_links:
            print("No articles found on the page.")
            return []
        
        # Process each article
        articles = []
        for link in article_links:
            print(f"Processing article: {link}")
            article = self.extract_article_details(link)
            if article:
                articles.append(asdict(article))
        
        return articles

def save_articles_to_json(articles: List[Dict], filename: str = 'articles.json') -> None:
    """Save scraped articles to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(articles)} articles to {filename}")
    except IOError as e:
        print(f"Error saving articles: {e}")

def main():
    scraper = WebScraper()
    articles = scraper.scrape_blog()
    
    if articles:
        print("\nScraping complete!")
        for i, article in enumerate(articles, 1):
            print(f"\nArticle {i}: {article['title']}")
            print(f"URL: {article['url']}")
            print(f"Word count: {article['word_count']}")
            print(f"Tags: {', '.join(article['tags'])}")
        
        # Save results
        save_articles_to_json(articles)
    else:
        print("No articles were scraped.")

if __name__ == "__main__":
    main()
