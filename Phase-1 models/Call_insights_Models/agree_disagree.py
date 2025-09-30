from textblob import TextBlob
import re
from typing import Dict, List, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json

# Ensure NLTK data is available
nltk.data.path.append("nltk_data")

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Custom VADER dictionary for polite phrases
custom_vader_scores = {
    'no problem': 0.8,
    'thank you': 0.9,
    'thanks': 0.9,
    'bye': 0.7,
    'correct': 0.8,
    'yeah': 0.7,
    'sorry': 0.7,  # Apology in positive context
    "take care": 0.8,
    "goodbye": 0.7,
    "have a nice day": 0.8,
    "appreciate": 0.9,
    "glad to assist": 0.9,
    "happy to help": 0.9,
    "will do": 0.8,
    "can you": 0.8,
    "ready": 0.8,
    "check for you": 0.8,
    "order to ready": 0.8,      
}

def calculate_agreeableness(transcription: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate agreeableness and disagreeableness for Agent and Customer based on English transcription.
    
    Args:
        transcription (List[Dict[str, Any]]): List of transcription entries with dialogues and speakers.
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary with agreeableness and disagreeableness scores for Agent and Customer.
    """
    # Initialize dictionaries to store dialogues by speaker
    dialogues = {"Agent": [], "Customer": []}
    for entry in transcription:
        speaker = entry.get("speaker", "")
        # Map spk1 to Agent and spk2 to Customer
        if speaker == "spk1":
            speaker = "Agent"
        elif speaker == "spk2":
            speaker = "Customer"
        dialogue = entry.get("dialogue", "").strip()
        if dialogue and speaker in dialogues:
            dialogues[speaker].append(dialogue)
    
    # Define patterns for agreeableness and disagreeableness
    agree_patterns = [
    r"\b(okay|ok|okey|sure|of course|please|sorry|thank you|thanks|alright|agree|understood|yes|absolutely|definitely|correct|done|no problem|yeah|yep|yup|certainly|exactly|right|sure thing|by all means|affirmative|roger|sounds good|gladly|fine|cool|great|perfect|ready|go ahead|as you wish|happy to|will do|i can|consider it done|appreciate|grateful|cheers|good afternoon|good morning|hi|hello|bye|check for you|order is ready|can i do|call me again)\b"
]
    # Exclude "no problem" from disagree patterns
    disagree_patterns = [
    r"\b(disagree|nah|no way|nope|never|not really|don’t think so|doesn’t work|no chance|not possible|not at all|don’t agree|not true|not correct|that’s wrong|wrong|incorrect|false|that’s not right|untrue|i doubt|i don’t believe|not sure|unsure|impossible|unlikely|don’t buy it|nonsense|rubbish|absurd|ridiculous|nuh uh|what do you mean|why|don’t think|don’t agree with that)\b"
]

    agree_regex = re.compile("|".join(agree_patterns), re.I)
    disagree_regex = re.compile("|".join(disagree_patterns), re.I)
    
    def compute_speaker_agreeableness(speaker_dialogues: List[str]) -> Dict[str, float]:
        """
        Compute agreeableness for a single speaker's dialogues.
        
        Args:
            speaker_dialogues (List[str]): List of dialogue strings for the speaker.
        
        Returns:
            Dict[str, float]: Dictionary with agreeableness and disagreeableness scores.
        """
        if not speaker_dialogues:
            return {"agreeableness": 0.5, "disagreeableness": 0.5}  # Neutral default
        
        sentiment_scores = []
        agree_scores = []
        disagree_scores = []
        
        # Debug: Print per-dialogue scores
        print(f"\nDebug for {speaker_dialogues[0].split()[0]} (first dialogue):")
        for dialogue in speaker_dialogues:
            # Sentiment analysis with VADER and custom overrides
            dialogue_lower = dialogue.lower().strip('.')
            polarity = custom_vader_scores.get(dialogue_lower, sid.polarity_scores(dialogue)['compound'])  # [-1, 1]
            sentiment_score = (polarity + 1) / 2  # Normalize to [0, 1]
            sentiment_scores.append(sentiment_score)
            
            # Keyword-based scoring
            agree_count = len(agree_regex.findall(dialogue_lower))
            disagree_count = len(disagree_regex.findall(dialogue_lower))
            total_words = max(len(word_tokenize(dialogue_lower)), 3)  # Cap denominator
            
            # Boost short agreeable dialogues
            agree_score = agree_count / total_words
            if agree_count > 0 and total_words <= 5:  # Boost for short, agreeable phrases
                agree_score *= 2.0
            disagree_score = disagree_count / total_words
            agree_scores.append(agree_score)
            disagree_scores.append(disagree_score)
            
            # Debug: Print scores for each dialogue
            print(f"Dialogue: '{dialogue}'")
            print(f"  Sentiment: {sentiment_score:.3f}, Agree Score: {agree_score:.3f}, Disagree Score: {disagree_score:.3f}")
        
        # Aggregate scores
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
        avg_agree = sum(agree_scores) / len(agree_scores) if agree_scores else 0.0
        avg_disagree = sum(disagree_scores) / len(disagree_scores) if disagree_scores else 0.0
        
        # Combine sentiment and keyword scores with baseline
        baseline = 0.5  # Neutral baseline to avoid overly low scores
        agreeableness = baseline + (0.2 * (avg_sentiment - 0.5)) + (0.65 * avg_agree) - (0.15 * avg_disagree)
        
        # Normalize to [0, 1]
        agreeableness = max(0.0, min(1.0, agreeableness))
        disagreeableness = 1.0 - agreeableness
        
        # Debug: Print aggregated scores
        print(f"\nAggregated for {speaker_dialogues[0].split()[0]}:")
        print(f"  Avg Sentiment: {avg_sentiment:.3f}, Avg Agree: {avg_agree:.3f}, Avg Disagree: {avg_disagree:.3f}")
        print(f"  Final Agreeableness: {agreeableness:.3f}, Disagreeableness: {disagreeableness:.3f}")
        
        return {
            "agreeableness": round(agreeableness, 2),
            "disagreeableness": round(disagreeableness, 2)
        }
    
    # Compute scores for both speakers
    result = {
        "Agent": compute_speaker_agreeableness(dialogues["Agent"]),
        "Customer": compute_speaker_agreeableness(dialogues["Customer"])
    }
    
    return result

def main():
    try:
        # Load transcription from transcription.json
        with open("transcription.json", "r") as infile:
            transcription = json.load(infile)
        
        # Run the function
        result = calculate_agreeableness(transcription)
        
        # Print results
        print("\nFinal Results:")
        print(json.dumps(result, indent=4))
        
        # Save results to a file
        with open("agreeableness_results.json", "w") as outfile:
            json.dump(result, outfile, indent=4)
        print("\nResults saved to 'agreeableness_results.json'")
        
    except FileNotFoundError:
        print("Error: 'transcription.json' file not found. Please ensure the file exists in the current directory.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'transcription.json'. Please check the file content.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()