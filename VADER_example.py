# in anaconda: pip install vaderSentiment
"""
The Compound score is a metric that calculates the sum of all the 
lexicon ratings which have been normalized between -1(most extreme negative) and +1 
(most extreme positive). In the case above, lexicon ratings for andsupercool are 2.9and respectively1.3. 
The compound score turns out to be 0.75 , denoting a very high positive sentiment.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
sentence = "The phone is super bad."
score = analyser.polarity_scores(sentence)
print(score)
