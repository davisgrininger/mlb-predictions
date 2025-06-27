import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from jinja2 import Template
import tweepy
import requests

# Import your existing predictor
from baseball_predictor import BaseballSavantPredictor

class GitHubTwitterAutomation:
    def __init__(self):
        """Initialize with environment variables for GitHub Actions"""
        self.odds_api_key = os.getenv('ODDS_API_KEY')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.twitter_api_key = os.getenv('TWITTER_API_KEY')
        self.twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        self.twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        self.twitter_access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        
        self.predictor = BaseballSavantPredictor(self.odds_api_key)
        
        # Initialize Twitter API
        self.setup_twitter()
    
    def setup_twitter(self):
        """Setup Twitter API v2 client"""
        try:
            self.twitter_client = tweepy.Client(
                bearer_token=self.twitter_bearer_token,
                consumer_key=self.twitter_api_key,
                consumer_secret=self.twitter_api_secret,
                access_token=self.twitter_access_token,
                access_token_secret=self.twitter_access_token_secret,
                wait_on_rate_limit=True
            )
            print("✅ Twitter API initialized")
        except Exception as e:
            print(f"❌ Twitter setup failed: {e}")
            self.twitter_client = None
    
    def generate_predictions(self):
        """Generate today's predictions"""
        print("🤖 Generating MLB predictions...")
        
        # Train model
        self.predictor.train_model()
        
        # Get predictions vs odds
        comparisons = self.predictor.compare_predictions_with_odds()
        
        return {
            'predictions': comparisons,
            'generated_at': datetime.now(),
            'game_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        }
    
    def create_github_pages_content(self, data):
        """Create beautiful GitHub Pages content"""
        
        # Get game date for display
        game_date = datetime.strptime(data['game_date'], '%Y-%m-%d')
        game_date_formatted = game_date.strftime('%B %d, %Y')
        
        # HTML template with modern styling
        html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>⚾ MLB Predictions - {{ game_date_formatted }}</title>
    <meta name="description" content="AI-powered MLB predictions using Statcast data for {{ game_date_formatted }}">
    
    <!-- Open Graph for social sharing -->
    <meta property="og:title" content="MLB Predictions - {{ game_date_formatted }}">
    <meta property="og:description" content="AI predictions for {{ total_games }} MLB games using Statcast data">
    <meta property="og:type" content="article">
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .stat-number {
            font-size: 2.2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        
        .value-bet {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
            border-left: 5px solid #ff6b6b;
            margin-bottom: 20px;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .value-bet-header {
            background: rgba(255,255,255,0.2);
            padding: 15px 20px;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .value-bet-content {
            padding: 20px;
        }
        
        .value-indicator {
            display: inline-block;
            background: #ff6b6b;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .game-card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        
        .game-card:hover {
            transform: translateY(-5px);
        }
        
        .game-header {
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
            color: #2c3e50;
        }
        
        .pitchers {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            text-align: center;
            font-style: italic;
        }
        
        .prediction {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .confidence {
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .agreement {
            text-align: center;
            margin-top: 15px;
            padding: 8px;
            border-radius: 8px;
        }
        
        .agreement.agree {
            background: #d4edda;
            color: #155724;
        }
        
        .agreement.disagree {
            background: #f8d7da;
            color: #721c24;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: rgba(255,255,255,0.8);
        }
        
        .footer a {
            color: rgba(255,255,255,0.9);
            text-decoration: none;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2em; }
            .container { padding: 10px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>⚾ MLB Predictions</h1>
            <p class="subtitle">{{ game_date_formatted }} • AI-Powered Statcast Analysis</p>
        </header>
        
        <div class="summary-cards">
            <div class="card">
                <h3>📊 Games Analyzed</h3>
                <div class="stat-number">{{ total_games }}</div>
                <p>With betting odds</p>
            </div>
            
            <div class="card">
                <h3>🎯 Avg Confidence</h3>
                <div class="stat-number">{{ "%.0f"|format(avg_confidence) }}%</div>
                <p>Model certainty</p>
            </div>
            
            <div class="card">
                <h3>🤝 Agreement</h3>
                <div class="stat-number">{{ agreement_pct }}%</div>
                <p>With Vegas odds</p>
            </div>
        </div>
        
        {% if value_bets %}
        <section class="card">
            <h3>💰 Top Value Opportunities</h3>
            <p style="margin-bottom: 20px;">Where our model significantly disagrees with the market:</p>
            
            {% for bet in value_bets %}
            <div class="value-bet">
                <div class="value-bet-header">
                    {{ bet.game }}
                </div>
                <div class="value-bet-content">
                    <strong>{{ bet.team }}</strong> {{ bet.odds if bet.odds > 0 else bet.odds }}
                    <br>
                    Model: {{ "%.1f"|format(bet.model_prob) }}% | Market: {{ "%.1f"|format(bet.market_prob) }}%
                    <span class="value-indicator">{{ "+%.1f"|format(bet.value) if bet.value > 0 else "%.1f"|format(bet.value) }}% Value</span>
                </div>
            </div>
            {% endfor %}
        </section>
        {% endif %}
        
        <section class="card">
            <h3>🎯 All Predictions</h3>
            <div class="prediction-grid">
                {% for pred in predictions %}
                <div class="game-card">
                    <div class="game-header">{{ pred.game }}</div>
                    
                    {% if pred.home_pitcher and pred.away_pitcher and pred.home_pitcher != 'TBD' %}
                    <div class="pitchers">
                        {{ pred.away_pitcher }} vs {{ pred.home_pitcher }}
                    </div>
                    {% endif %}
                    
                    <div class="prediction">
                        <strong>Pick: {{ pred.predicted_winner }}</strong>
                        <span class="confidence">{{ "%.0f"|format(pred.confidence * 100) }}%</span>
                    </div>
                    
                    <div style="font-size: 0.9em; color: #666;">
                        Model: {{ pred.home_team }} {{ "%.0f"|format(pred.model_home_prob * 100) }}% | 
                        {{ pred.away_team }} {{ "%.0f"|format(pred.model_away_prob * 100) }}%
                    </div>
                    
                    <div class="agreement {{ 'agree' if pred.agreement else 'disagree' }}">
                        {% if pred.agreement %}
                        ✅ Agrees with Vegas ({{ pred.odds_favorite }} favored)
                        {% else %}
                        ❌ Disagrees with Vegas ({{ pred.odds_favorite }} favored)
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
            </div>
        </section>
        
        <footer class="footer">
            <p>Generated at {{ generated_at.strftime('%I:%M %p ET') }} using Statcast data & machine learning</p>
            <p><small>Not gambling advice • <a href="https://github.com/yourusername/mlb-predictions">View source code</a></small></p>
        </footer>
    </div>
</body>
</html>
        """)
        
        # Calculate summary stats
        total_games = len(data['predictions'])
        if total_games > 0:
            agreements = sum(1 for p in data['predictions'] if p['agreement'])
            agreement_pct = round(agreements / total_games * 100)
            avg_confidence = np.mean([max(p['model_home_prob'], p['model_away_prob']) for p in data['predictions']]) * 100
            
            # Find value bets
            value_bets = []
            for pred in data['predictions']:
                home_diff = pred['prob_diff_home']
                away_diff = pred['prob_diff_away']
                
                if abs(home_diff) > 0.05:  # 5% threshold
                    value_bets.append({
                        'game': pred['game'],
                        'team': pred['home_team'],
                        'model_prob': pred['model_home_prob'] * 100,
                        'market_prob': pred['odds_home_prob'] * 100,
                        'value': home_diff * 100,
                        'odds': pred['home_odds']
                    })
                
                if abs(away_diff) > 0.05:
                    value_bets.append({
                        'game': pred['game'],
                        'team': pred['away_team'],
                        'model_prob': pred['model_away_prob'] * 100,
                        'market_prob': pred['odds_away_prob'] * 100,
                        'value': away_diff * 100,
                        'odds': pred['away_odds']
                    })
            
            # Sort by biggest value
            value_bets.sort(key=lambda x: abs(x['value']), reverse=True)
            value_bets = value_bets[:3]  # Top 3
        else:
            agreement_pct = 0
            avg_confidence = 0
            value_bets = []
        
        content = html_template.render(
            predictions=data['predictions'],
            generated_at=data['generated_at'],
            game_date_formatted=game_date_formatted,
            total_games=total_games,
            agreement_pct=agreement_pct,
            avg_confidence=avg_confidence,
            value_bets=value_bets
        )
        
        return content
    
    def create_twitter_thread(self, data):
        """Create beautiful Twitter thread"""
        
        if not data['predictions']:
            return ["🚨 No MLB games with odds today. Check back tomorrow for predictions! ⚾"]
        
        game_date = datetime.strptime(data['game_date'], '%Y-%m-%d')
        game_date_formatted = game_date.strftime('%B %d')
        
        tweets = []
        
        # Main tweet with summary
        total_games = len(data['predictions'])
        agreements = sum(1 for p in data['predictions'] if p['agreement'])
        agreement_pct = round(agreements / total_games * 100)
        avg_confidence = np.mean([max(p['model_home_prob'], p['model_away_prob']) for p in data['predictions']]) * 100
        
        main_tweet = f"""🤖⚾ MLB PREDICTIONS - {game_date_formatted}
        
📊 {total_games} games analyzed using Statcast data
🎯 {avg_confidence:.0f}% average model confidence  
🤝 {agreement_pct}% agreement with Vegas

AI vs The House 👇 🧵"""
        
        tweets.append(main_tweet)
        
        # Find best picks (highest confidence)
        best_picks = sorted(data['predictions'], key=lambda x: max(x['model_home_prob'], x['model_away_prob']), reverse=True)[:3]
        
        picks_tweet = "🔥 TOP CONFIDENT PICKS:\n\n"
        for i, pick in enumerate(best_picks, 1):
            confidence = max(pick['model_home_prob'], pick['model_away_prob']) * 100
            winner = pick['predicted_winner']
            agreement_emoji = "✅" if pick['agreement'] else "🚨"
            
            # Get pitcher info if available
            pitcher_info = ""
            if pick.get('home_pitcher') and pick.get('away_pitcher') and pick['home_pitcher'] != 'TBD':
                pitcher_info = f"\n🥎 {pick['away_pitcher']} vs {pick['home_pitcher']}"
            
            picks_tweet += f"{i}. {pick['game']}{pitcher_info}\n📈 {winner} ({confidence:.0f}%) {agreement_emoji}\n\n"
        
        tweets.append(picks_tweet.strip())
        
        # Value bets
        value_bets = []
        for pred in data['predictions']:
            home_diff = pred['prob_diff_home']
            away_diff = pred['prob_diff_away']
            
            if abs(home_diff) > 0.05:
                value_bets.append({
                    'game': pred['game'], 'team': pred['home_team'],
                    'value': home_diff * 100, 'odds': pred['home_odds']
                })
            
            if abs(away_diff) > 0.05:
                value_bets.append({
                    'game': pred['game'], 'team': pred['away_team'],
                    'value': away_diff * 100, 'odds': pred['away_odds']
                })
        
        if value_bets:
            value_bets.sort(key=lambda x: abs(x['value']), reverse=True)
            value_tweet = "💰 VALUE ALERTS (Model vs Vegas):\n\n"
            
            for bet in value_bets[:2]:  # Top 2 value bets
                value_tweet += f"🎯 {bet['team']} in {bet['game']}\n"
                value_tweet += f"💡 {bet['value']:+.1f}% edge vs market\n"
                value_tweet += f"📊 Odds: {bet['odds']:+d}\n\n"
            
            tweets.append(value_tweet.strip())
        
        # Upset alerts
        upset_alerts = []
        for pred in data['predictions']:
            # Find underdogs with >40% chance
            if pred['odds_home_prob'] < pred['odds_away_prob']:  # Home is underdog
                if pred['model_home_prob'] > 0.4:
                    upset_alerts.append({
                        'team': pred['home_team'],
                        'game': pred['game'],
                        'model_prob': pred['model_home_prob'] * 100
                    })
            else:  # Away is underdog
                if pred['model_away_prob'] > 0.4:
                    upset_alerts.append({
                        'team': pred['away_team'],
                        'game': pred['game'],
                        'model_prob': pred['model_away_prob'] * 100
                    })
        
        if upset_alerts:
            upset_tweet = "🚨 UPSET WATCH:\n\n"
            for alert in upset_alerts[:2]:  # Top 2 upsets
                upset_tweet += f"⚡ {alert['team']} in {alert['game']}\n"
                upset_tweet += f"🎲 {alert['model_prob']:.0f}% chance (underdog!)\n\n"
            
            tweets.append(upset_tweet.strip())
        
        # Final tweet with disclaimer and link
        final_tweet = f"""📈 Full analysis with all {total_games} predictions:
🔗 [GitHub Pages link will be added]

🤖 Powered by Statcast data & machine learning
⚠️ Not gambling advice - bet responsibly!

#MLB #BaseballPredictions #Statcast #AI"""
        
        tweets.append(final_tweet)
        
        return tweets
    
    def post_twitter_thread(self, tweets):
        """Post Twitter thread"""
        if not self.twitter_client:
            print("❌ Twitter client not available")
            return
        
        try:
            tweet_ids = []
            
            for i, tweet in enumerate(tweets):
                if i == 0:
                    # First tweet
                    response = self.twitter_client.create_tweet(text=tweet)
                    tweet_ids.append(response.data['id'])
                    print(f"✅ Posted main tweet: {tweet[:50]}...")
                else:
                    # Reply to previous tweet
                    response = self.twitter_client.create_tweet(
                        text=tweet,
                        in_reply_to_tweet_id=tweet_ids[-1]
                    )
                    tweet_ids.append(response.data['id'])
                    print(f"✅ Posted reply {i}: {tweet[:50]}...")
            
            print(f"🐦 Successfully posted {len(tweets)} tweet thread!")
            return tweet_ids
            
        except Exception as e:
            print(f"❌ Twitter posting failed: {e}")
            return None
    
    def save_to_github_pages(self, content):
        """Save content for GitHub Pages"""
        try:
            # Create docs directory if it doesn't exist
            os.makedirs('docs', exist_ok=True)
            
            # Save main page
            with open('docs/index.html', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Also save dated version
            date_str = datetime.now().strftime('%Y-%m-%d')
            with open(f'docs/predictions-{date_str}.html', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Saved to GitHub Pages")
            return f"https://yourusername.github.io/mlb-predictions/"
            
        except Exception as e:
            print(f"❌ GitHub Pages save failed: {e}")
            return None
    
    def run_automation(self):
        """Main automation function"""
        print("🚀 Starting GitHub + Twitter automation...")
        
        try:
            # Generate predictions
            data = self.generate_predictions()
            
            if not data['predictions']:
                print("⚠️ No games with odds today")
                # Still post a tweet about it
                no_games_tweet = "🚨 No MLB games with betting odds today. The robots are taking a rest day! 🤖⚾\n\nCheck back tomorrow for AI-powered predictions! 📊"
                if self.twitter_client:
                    self.twitter_client.create_tweet(text=no_games_tweet)
                return
            
            # Create GitHub Pages content
            html_content = self.create_github_pages_content(data)
            
            # Save to GitHub Pages
            github_url = self.save_to_github_pages(html_content)
            
            # Create Twitter thread
            tweets = self.create_twitter_thread(data)
            
            # Update final tweet with actual link
            if github_url and len(tweets) > 0:
                tweets[-1] = tweets[-1].replace('[GitHub Pages link will be added]', github_url)
            
            # Post to Twitter
            self.post_twitter_thread(tweets)
            
            print("✅ Automation completed successfully!")
            
        except Exception as e:
            print(f"❌ Automation failed: {e}")
            # Post error tweet
            error_tweet = "🚨 Prediction bot encountered an error today. The humans are investigating! 🔧🤖\n\n#MLBPredictions #TechnicalDifficulties"
            if self.twitter_client:
                try:
                    self.twitter_client.create_tweet(text=error_tweet)
                except:
                    pass

if __name__ == "__main__":
    automation = GitHubTwitterAutomation()
    automation.run_automation()
