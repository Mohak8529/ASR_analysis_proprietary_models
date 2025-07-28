import json
import nltk
import spacy
import skfuzzy as fuzz
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple
import os

# Check if vader_lexicon is already downloaded
def ensure_vader_lexicon():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        # Only attempt download if not found
        try:
            
            nltk.download('vader_lexicon')
        except Exception as e:
            raise Exception(f"Failed to download vader_lexicon and it's not locally available: {e}")

# Initialize NLP tools
ensure_vader_lexicon()
sid = SentimentIntensityAnalyzer()
nlp = spacy.load('en_core_web_sm')

def load_transcription(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def get_sentiment_score(dialogue: List[Dict]) -> Tuple[str, Dict[str, float], List[str]]:
    """Compute a single sentiment value, VADER scores for intent, and collect keywords."""
    customer_text = ' '.join(entry['dialogue'] for entry in dialogue if entry['speaker'] == 'spk2')
    scores = sid.polarity_scores(customer_text)
    keywords = []
    
    # Determine single sentiment value
    sentiment_scores = {
        'Positive': max(0.0, scores['pos']),
        'Neutral': max(0.0, scores['neu']),
        'Negative': max(0.0, scores['neg'])
    }
    
    # Decision rule: Prioritize Positive/Negative over Neutral if significant
    if sentiment_scores['Positive'] > sentiment_scores['Negative'] and sentiment_scores['Positive'] > 0.2:
        sentiment = 'positive'
        keywords.extend(['positive', 'agree', 'thanks'])
    elif sentiment_scores['Negative'] > sentiment_scores['Positive'] and sentiment_scores['Negative'] > 0.2:
        sentiment = 'negative'
        keywords.extend(['negative', 'issue'])
    else:
        sentiment = 'neutral'
        keywords.append('neutral')
    
    return sentiment, sentiment_scores, list(set(keywords))

def get_cooperation_level(dialogue: List[Dict]) -> Tuple[Dict[str, float], List[str]]:
    """Compute cooperation level and collect keywords."""
    cooperation_score = 0.0
    rep_supportive = 0.0
    keywords = []
    for entry in dialogue:
        text = entry['dialogue'].lower()
        if entry['speaker'] == 'spk2':
            if any(phrase in text for phrase in ['i’ll pay', 'sure', 'meaning to clear']):
                cooperation_score += 0.4
                keywords.extend(['i’ll pay', 'sure', 'meaning to clear'])
            if any(phrase in text for phrase in ['no', 'won’t', 'can’t']) and 'pay' in text:
                cooperation_score -= 0.3
                keywords.extend(['no', 'won’t', 'can’t'])
        if entry['speaker'] == 'spk1':
            if any(phrase in text for phrase in ['of course', 'perfect', 'thank you']):
                rep_supportive += 0.2
                keywords.extend(['of course', 'perfect', 'thank you'])
    cooperation_score += rep_supportive * 0.3  # Weight representative's tone
    return {
        'High': min(cooperation_score, 0.9),
        'Medium': max(0.1, 1 - cooperation_score),
        'Low': max(0.0, 0.3 - cooperation_score)
    }, list(set(keywords))

def get_resolution_level(dialogue: List[Dict]) -> Tuple[Dict[str, float], List[str]]:
    """Compute resolution level and collect keywords."""
    resolution_score = 0.0
    keywords = []
    for i, entry in enumerate(dialogue):
        text = entry['dialogue'].lower()
        if 'pay' in text and any(word in text for word in ['today', 'evening']):
            resolution_score += 0.5
            keywords.extend(['pay', 'today', 'evening'])
        if any(word in text for word in ['reflects', 'resolved', 'cleared']):
            resolution_score += 0.3
            keywords.extend(['reflects', 'resolved', 'cleared'])
        # Weight later dialogue more (context of resolution)
        if i > len(dialogue) / 2:
            resolution_score *= 1.2
    return {
        'High': min(resolution_score, 0.9),
        'Medium': max(0.1, 1 - resolution_score),
        'Low': max(0.0, 0.3 - resolution_score)
    }, list(set(keywords))

def get_entity_score(dialogue: List[Dict]) -> Tuple[Dict[str, float], List[str]]:
    """Detect product mentions using NER and keyword matching, collect keywords."""
    product_scores = {'credit card': 0.05, 'saving account': 0.05, 'insurance': 0.05, 'travel card': 0.05}
    keywords = []
    all_text = ' '.join(entry['dialogue'] for entry in dialogue)
    doc = nlp(all_text.lower())
    
    # Keyword-based detection with NER context
    for token in doc:
        text = token.text
        if text in ['credit', 'card'] or 'card ending' in all_text:
            product_scores['credit card'] += 0.4
            keywords.extend(['credit', 'card', 'card ending'])
        if text in ['savings', 'account']:
            product_scores['saving account'] += 0.4
            keywords.extend(['savings', 'account'])
        if text == 'insurance' or 'policy' in all_text:
            product_scores['insurance'] += 0.4
            keywords.extend(['insurance', 'policy'])
        if 'travel' in text or 'foreign transaction' in all_text:
            product_scores['travel card'] += 0.4
            keywords.extend(['travel', 'foreign transaction'])
    
    # NER for entities (e.g., PRODUCT)
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT':
            if 'card' in ent.text.lower():
                product_scores['credit card'] += 0.3
                keywords.append(ent.text.lower())
            elif 'account' in ent.text.lower():
                product_scores['saving account'] += 0.3
                keywords.append(ent.text.lower())
            elif 'insurance' in ent.text.lower():
                product_scores['insurance'] += 0.3
                keywords.append(ent.text.lower())
    
    return {k: min(v, 0.95) for k, v in product_scores.items()}, list(set(keywords))

def get_contextual_relevance(dialogue: List[Dict]) -> Tuple[Dict[str, float], List[str]]:
    """Compute contextual relevance for products and collect keywords."""
    relevance_scores = {'credit card': 0.05, 'saving account': 0.05, 'insurance': 0.05, 'travel card': 0.05}
    keywords = []
    for entry in dialogue:
        text = entry['dialogue'].lower()
        if 'overdue' in text or 'late fee' in text:
            relevance_scores['credit card'] += 0.4
            keywords.extend(['overdue', 'late fee'])
        if 'balance' in text or 'deposit' in text:
            relevance_scores['saving account'] += 0.4
            keywords.extend(['balance', 'deposit'])
        if 'premium' in text or 'claim' in text:
            relevance_scores['insurance'] += 0.4
            keywords.extend(['premium', 'claim'])
        if 'foreign' in text or 'travel' in text:
            relevance_scores['travel card'] += 0.4
            keywords.extend(['foreign', 'travel'])
    return {k: min(v, 0.9) for k, v in relevance_scores.items()}, list(set(keywords))

def generate_consumables_insights(dialogue: List[Dict], product: str, resolution_level: Dict[str, float]) -> List[str]:
    """Generate a list of keywords representing actionable insights."""
    keywords = []
    
    for entry in dialogue:
        text = entry['dialogue'].lower()
        # Extract issue details
        if 'overdue payment' in text:
            keywords.append('overdue')
            # Extract amount if mentioned
            import re
            amount_match = re.search(r'₹([\d,]+)', text)
            if amount_match:
                keywords.append(amount_match.group(0))
        if 'late fee' in text:
            keywords.append('late fee')
        # Check resolution actions
        if 'i’ll pay' in text or 'pay it online' in text:
            keywords.extend(['pay', 'committed'])
        if 'reflects' in text and 'late fee' in text:
            keywords.append('resolve')
    
    # Add product
    if product:
        keywords.append(product)
    
    # Add resolution likelihood
    if resolution_level['High'] > 0.5:
        keywords.append('high')
    elif resolution_level['Medium'] > 0.5:
        keywords.append('medium')
    else:
        keywords.append('low')
    
    return list(set(keywords)) if keywords else ['no_insights']

def fuzzy_intent(sentiment_scores: Dict, cooperation: Dict, resolution: Dict) -> Tuple[str, float]:
    """Compute intent using fuzzy logic."""
    positive = min(sentiment_scores['Positive'], cooperation['High'], resolution['High'])
    positive += min(sentiment_scores['Neutral'], cooperation['Medium'], resolution['Medium']) * 0.5
    negative = max(sentiment_scores['Negative'], cooperation['Low'], resolution['Low'])
    
    intent_score = positive / (positive + negative + 1e-10)
    intent = 'positive' if intent_score > 0.5 else 'negative'
    confidence = max(positive, negative)
    return intent, confidence

def fuzzy_product(entity_scores: Dict, relevance_scores: Dict) -> Tuple[str, float]:
    """Compute product using fuzzy logic."""
    product_scores = {}
    for product in entity_scores:
        product_scores[product] = min(entity_scores[product], relevance_scores[product])
    
    selected_product = max(product_scores, key=product_scores.get)
    confidence = product_scores[selected_product]
    return selected_product, confidence

def process_transcription(transcription: List[Dict]) -> Dict:
    """Process transcription to produce intent, sentiment, product, keywords, insights, and confidence."""
    sentiment, sentiment_scores, sentiment_keywords = get_sentiment_score(transcription)
    cooperation, cooperation_keywords = get_cooperation_level(transcription)
    resolution, resolution_keywords = get_resolution_level(transcription)
    entity_scores, entity_keywords = get_entity_score(transcription)
    relevance_scores, relevance_keywords = get_contextual_relevance(transcription)
    
    intent, intent_confidence = fuzzy_intent(sentiment_scores, cooperation, resolution)
    product, product_confidence = fuzzy_product(entity_scores, relevance_scores)
    
    # Combine all keywords, ensuring uniqueness
    critical_keywords = list(set(
        sentiment_keywords + cooperation_keywords + resolution_keywords +
        entity_keywords + relevance_keywords
    ))
    
    # Generate consumables insights
    consumables_insights = generate_consumables_insights(transcription, product, resolution)
    
    return {
        'intent': intent,
        'sentiment': sentiment,
        'product': product,
        'critical_keywords': critical_keywords,
        'consumables_insights': consumables_insights,
        'confidence': {
            'intent': round(intent_confidence, 2),
            'product': round(product_confidence, 2)
        }
    }

# Example usage
if __name__ == "__main__":
    transcription = load_transcription('transcription.json')
    result = process_transcription(transcription)
    print(json.dumps(result, indent=2))