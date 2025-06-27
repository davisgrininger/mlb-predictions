import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pybaseball as pb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import requests
import json
import warnings
warnings.filterwarnings('ignore')

class BaseballSavantPredictor:
    def __init__(self, odds_api_key=None):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.odds_api_key = odds_api_key
        self.odds_api_base_url = "https://api.the-odds-api.com/v4"
        
        # Enable pybaseball cache for faster subsequent calls
        pb.cache.enable()
        
        # Team name mappings for odds API
        self.team_mappings = {
            'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
            'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
            'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
            'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
            'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
            'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
            'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
            'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
            'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
            'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
            'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
            'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX',
            'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN'
        }
        
        # Team full names for Baseball Savant
        self.savant_teams = {
            'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves',
            'BAL': 'Baltimore Orioles', 'BOS': 'Boston Red Sox',
            'CHC': 'Chicago Cubs', 'CWS': 'Chicago White Sox',
            'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians',
            'COL': 'Colorado Rockies', 'DET': 'Detroit Tigers',
            'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
            'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers',
            'MIA': 'Miami Marlins', 'MIL': 'Milwaukee Brewers',
            'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
            'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics',
            'PHI': 'Philadelphia Phillies', 'PIT': 'Pittsburgh Pirates',
            'SD': 'San Diego Padres', 'SF': 'San Francisco Giants',
            'SEA': 'Seattle Mariners', 'STL': 'St. Louis Cardinals',
            'TB': 'Tampa Bay Rays', 'TEX': 'Texas Rangers',
            'TOR': 'Toronto Blue Jays', 'WSN': 'Washington Nationals'
        }
    
    def get_probable_pitchers(self, home_team, away_team, game_date=None):
        """Get probable starting pitchers for a game"""
        try:
            if not game_date:
                game_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            print(f"Getting probable pitchers for {away_team} @ {home_team} on {game_date}")
            
            # Try MLB Stats API approach
            url = 'https://statsapi.mlb.com/api/v1/schedule'
            params = {
                'sportId': '1',
                'date': game_date,
                'hydrate': 'probablePitcher,team'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                for date_data in data.get('dates', []):
                    for game in date_data.get('games', []):
                        home_team_name = game['teams']['home']['team']['name']
                        away_team_name = game['teams']['away']['team']['name']
                        
                        # Map to our abbreviations
                        game_home_abbr = self._map_team_name(home_team_name)
                        game_away_abbr = self._map_team_name(away_team_name)
                        
                        if game_home_abbr == home_team and game_away_abbr == away_team:
                            home_pitcher = game['teams']['home'].get('probablePitcher', {})
                            away_pitcher = game['teams']['away'].get('probablePitcher', {})
                            
                            return {
                                'home_pitcher': {
                                    'id': home_pitcher.get('id'),
                                    'name': home_pitcher.get('fullName', 'TBD'),
                                    'mlb_id': home_pitcher.get('id')
                                },
                                'away_pitcher': {
                                    'id': away_pitcher.get('id'),
                                    'name': away_pitcher.get('fullName', 'TBD'),
                                    'mlb_id': away_pitcher.get('id')
                                }
                            }
            
            # Fallback: Return TBD pitchers
            return {
                'home_pitcher': {'id': None, 'name': 'TBD', 'mlb_id': None},
                'away_pitcher': {'id': None, 'name': 'TBD', 'mlb_id': None}
            }
            
        except Exception as e:
            print(f"Error getting probable pitchers: {e}")
            return {
                'home_pitcher': {'id': None, 'name': 'TBD', 'mlb_id': None},
                'away_pitcher': {'id': None, 'name': 'TBD', 'mlb_id': None}
            }
    
    def get_pitcher_stats(self, pitcher_id, pitcher_name):
        """Get comprehensive pitcher statistics using Statcast and pybaseball"""
        try:
            if not pitcher_id or not pitcher_name:
                return {}
            
            print(f"Fetching stats for pitcher: {pitcher_name} (ID: {pitcher_id})")
            
            # Get recent Statcast data for this pitcher
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # Last 60 days
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get Statcast data for this specific pitcher
            pitcher_data = pb.statcast_pitcher(start_str, end_str, pitcher_id)
            
            if pitcher_data.empty:
                print(f"No recent Statcast data for {pitcher_name}")
                return self._get_pitcher_fallback_stats()
            
            # Calculate pitcher metrics
            stats = {}
            
            # Basic performance metrics
            stats['recent_innings'] = len(pitcher_data) / 6.0  # Rough estimate (6 pitches per batter avg)
            stats['total_pitches'] = len(pitcher_data)
            
            # Velocity metrics
            fastballs = pitcher_data[pitcher_data['pitch_type'].isin(['FF', 'SI', 'FC'])]
            if not fastballs.empty:
                stats['avg_fastball_velo'] = fastballs['release_speed'].mean()
                stats['max_fastball_velo'] = fastballs['release_speed'].max()
            
            # Spin rate
            stats['avg_spin_rate'] = pitcher_data['release_spin_rate'].mean()
            
            # Control metrics
            strikes = pitcher_data[pitcher_data['type'].isin(['S', 'X'])]  # Strikes and balls in play
            stats['strike_rate'] = len(strikes) / len(pitcher_data) if len(pitcher_data) > 0 else 0
            
            # Zone metrics
            in_zone = pitcher_data[(pitcher_data['zone'] >= 1) & (pitcher_data['zone'] <= 9)]
            stats['zone_rate'] = len(in_zone) / len(pitcher_data) if len(pitcher_data) > 0 else 0
            
            # Whiff rate (swinging strikes)
            whiffs = pitcher_data[pitcher_data['description'].isin(['swinging_strike', 'swinging_strike_blocked'])]
            swings = pitcher_data[pitcher_data['description'].isin([
                'swinging_strike', 'swinging_strike_blocked', 'foul', 'hit_into_play'
            ])]
            stats['whiff_rate'] = len(whiffs) / len(swings) if len(swings) > 0 else 0
            
            # Contact quality against
            contact = pitcher_data[pitcher_data['type'] == 'X']  # Balls in play
            if not contact.empty:
                stats['avg_exit_velo_against'] = contact['launch_speed'].mean()
                stats['hard_hit_rate_against'] = (contact['launch_speed'] >= 95).mean()
                stats['barrel_rate_against'] = (contact['barrel'] == 1).mean() if 'barrel' in contact.columns else 0
                
                # xwOBA against (if available)
                if 'estimated_woba_using_speedangle' in contact.columns:
                    stats['xwoba_against'] = contact['estimated_woba_using_speedangle'].mean()
            
            # Recent performance trend
            recent_games = pitcher_data.groupby('game_date').size()
            stats['recent_games'] = len(recent_games)
            stats['recent_game_dates'] = list(recent_games.index)
            
            return stats
            
        except Exception as e:
            print(f"Error getting pitcher stats for {pitcher_name}: {e}")
            return self._get_pitcher_fallback_stats()
    
    def _get_pitcher_fallback_stats(self):
        """Fallback pitcher stats when real data unavailable"""
        return {
            'avg_fastball_velo': 92.5,
            'avg_spin_rate': 2250,
            'strike_rate': 0.65,
            'zone_rate': 0.50,
            'whiff_rate': 0.25,
            'avg_exit_velo_against': 88.0,
            'hard_hit_rate_against': 0.35,
            'barrel_rate_against': 0.08,
            'recent_games': 3
        }
    
    def calculate_pitcher_matchup_advantage(self, home_pitcher_stats, away_pitcher_stats):
        """Calculate pitching matchup advantage"""
        try:
            advantages = {}
            
            # Velocity advantage
            home_velo = home_pitcher_stats.get('avg_fastball_velo', 92.5)
            away_velo = away_pitcher_stats.get('avg_fastball_velo', 92.5)
            advantages['fastball_velo_diff'] = home_velo - away_velo
            
            # Control advantage (higher zone rate is better)
            home_zone = home_pitcher_stats.get('zone_rate', 0.50)
            away_zone = away_pitcher_stats.get('zone_rate', 0.50)
            advantages['control_advantage'] = home_zone - away_zone
            
            # Stuff advantage (higher whiff rate is better)
            home_whiff = home_pitcher_stats.get('whiff_rate', 0.25)
            away_whiff = away_pitcher_stats.get('whiff_rate', 0.25)
            advantages['stuff_advantage'] = home_whiff - away_whiff
            
            # Contact quality advantage (lower exit velo against is better)
            home_exit_velo = home_pitcher_stats.get('avg_exit_velo_against', 88.0)
            away_exit_velo = away_pitcher_stats.get('avg_exit_velo_against', 88.0)
            advantages['contact_quality_advantage'] = away_exit_velo - home_exit_velo  # Reversed
            
            # Overall pitching advantage score
            advantages['overall_pitching_advantage'] = (
                (advantages['control_advantage'] * 0.3) +
                (advantages['stuff_advantage'] * 0.4) +
                (advantages['contact_quality_advantage'] * 0.3)
            )
            
            return advantages
            
        except Exception as e:
            print(f"Error calculating pitcher advantage: {e}")
            return {'overall_pitching_advantage': 0}
    
    def get_team_statcast_data(self, team_abbr, days_back=30):
        """Get real Statcast data for a team from the last X days"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            print(f"Fetching Statcast data for {team_abbr} from {start_str} to {end_str}...")
            
            # Get Statcast data for the date range
            statcast_data = pb.statcast(start_dt=start_str, end_dt=end_str)
            
            if statcast_data.empty:
                print(f"No Statcast data found for {team_abbr}")
                return {}
            
            # Filter for the specific team (both home and away)
            team_full_name = self.savant_teams.get(team_abbr, team_abbr)
            team_data = statcast_data[
                (statcast_data['home_team'] == team_full_name) | 
                (statcast_data['away_team'] == team_full_name)
            ].copy()
            
            if team_data.empty:
                print(f"No team-specific data found for {team_abbr}")
                return {}
            
            # Calculate offensive stats when team is batting
            batting_data = team_data[
                ((team_data['home_team'] == team_full_name) & (team_data['inning_topbot'] == 'Bot')) |
                ((team_data['away_team'] == team_full_name) & (team_data['inning_topbot'] == 'Top'))
            ]
            
            # Calculate pitching stats when team is pitching
            pitching_data = team_data[
                ((team_data['home_team'] == team_full_name) & (team_data['inning_topbot'] == 'Top')) |
                ((team_data['away_team'] == team_full_name) & (team_data['inning_topbot'] == 'Bot'))
            ]
            
            stats = {}
            
            # Batting statistics
            if not batting_data.empty:
                stats.update({
                    'avg_exit_velocity': batting_data['launch_speed'].mean(),
                    'avg_launch_angle': batting_data['launch_angle'].mean(),
                    'barrel_rate': (batting_data['barrel'] == 1).mean() * 100,
                    'hard_hit_rate': (batting_data['launch_speed'] >= 95).mean() * 100,
                    'sweet_spot_rate': ((batting_data['launch_angle'] >= 8) & 
                                      (batting_data['launch_angle'] <= 32)).mean() * 100,
                    'avg_distance': batting_data['hit_distance_sc'].mean(),
                    'xba': batting_data['estimated_ba_using_speedangle'].mean(),
                    'xslg': batting_data['estimated_slg_using_speedangle'].mean()
                })
            
            # Pitching statistics  
            if not pitching_data.empty:
                stats.update({
                    'avg_fastball_velo': pitching_data[pitching_data['pitch_type'].isin(['FF', 'SI'])]['release_speed'].mean(),
                    'avg_spin_rate': pitching_data['release_spin_rate'].mean(),
                    'strike_rate': ((pitching_data['zone'] >= 1) & (pitching_data['zone'] <= 9)).mean() * 100,
                    'whiff_rate': (pitching_data['description'].isin(['swinging_strike', 'swinging_strike_blocked'])).mean() * 100,
                    'chase_rate': (pitching_data['zone'] > 9).mean() * 100,
                    'xera': pitching_data['estimated_ba_using_speedangle'].mean() if 'estimated_ba_using_speedangle' in pitching_data.columns else None
                })
            
            return stats
            
        except Exception as e:
            print(f"Error getting Statcast data for {team_abbr}: {e}")
            return {}
    
    def get_team_standings(self):
        """Get current MLB standings"""
        try:
            current_year = datetime.now().year
            standings_data = pb.standings(current_year)
            
            team_records = {}
            for division in standings_data:
                for _, team in division.iterrows():
                    # Extract team abbreviation from team name
                    team_name = team['Tm']
                    # Simple mapping - you might need to adjust this
                    for abbr, full_name in self.savant_teams.items():
                        if full_name.split()[-1] in team_name or abbr in team_name:
                            team_records[abbr] = {
                                'wins': team['W'],
                                'losses': team['L'],
                                'win_pct': team['W-L%'],
                                'games_back': team['GB'] if team['GB'] != '--' else 0
                            }
                            break
            
            return team_records
            
        except Exception as e:
            print(f"Error getting standings: {e}")
            return {}
    
    def get_recent_game_results(self, team_abbr, games_back=10):
        """Get recent game results for momentum calculation"""
        try:
            # This would ideally use game-by-game results
            # For now, we'll estimate from Statcast game data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=games_back * 2)  # Rough estimate
            
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Get game results from Statcast data
            data = pb.statcast(start_dt=start_str, end_dt=end_str)
            
            if data.empty:
                return {'recent_form': 0.5}  # Neutral
            
            team_full_name = self.savant_teams.get(team_abbr)
            team_games = data[
                (data['home_team'] == team_full_name) | 
                (data['away_team'] == team_full_name)
            ]
            
            # Group by game and calculate results
            games = team_games.groupby('game_pk').agg({
                'home_team': 'first',
                'away_team': 'first',
                'home_score': 'first',
                'away_score': 'first'
            })
            
            wins = 0
            total_games = len(games)
            
            for _, game in games.iterrows():
                if game['home_team'] == team_full_name:
                    if game['home_score'] > game['away_score']:
                        wins += 1
                else:  # away team
                    if game['away_score'] > game['home_score']:
                        wins += 1
            
            recent_form = wins / total_games if total_games > 0 else 0.5
            
            return {
                'recent_form': recent_form,
                'recent_wins': wins,
                'recent_games': total_games
            }
            
        except Exception as e:
            print(f"Error getting recent results for {team_abbr}: {e}")
            return {'recent_form': 0.5}
    
    def create_features(self, home_team, away_team):
        """Create feature vector using real Statcast, standings, and PITCHER data"""
        print(f"Creating comprehensive features for {away_team} @ {home_team}...")
        
        # Get probable pitchers first
        pitchers = self.get_probable_pitchers(home_team, away_team)
        home_pitcher = pitchers['home_pitcher']
        away_pitcher = pitchers['away_pitcher']
        
        print(f"Starting Pitchers: {away_pitcher['name']} vs {home_pitcher['name']}")
        
        # Get team Statcast data
        home_stats = self.get_team_statcast_data(home_team)
        away_stats = self.get_team_statcast_data(away_team)
        
        # Get pitcher stats
        home_pitcher_stats = self.get_pitcher_stats(home_pitcher['mlb_id'], home_pitcher['name'])
        away_pitcher_stats = self.get_pitcher_stats(away_pitcher['mlb_id'], away_pitcher['name'])
        
        # Calculate pitcher matchup advantage
        pitcher_advantages = self.calculate_pitcher_matchup_advantage(home_pitcher_stats, away_pitcher_stats)
        
        # Get standings data
        standings = self.get_team_standings()
        home_record = standings.get(home_team, {})
        away_record = standings.get(away_team, {})
        
        # Get recent form
        home_form = self.get_recent_game_results(home_team)
        away_form = self.get_recent_game_results(away_team)
        
        features = {}
        
        # PITCHER FEATURES (Most Important!)
        features['home_pitcher_fastball_velo'] = home_pitcher_stats.get('avg_fastball_velo', 92.5)
        features['away_pitcher_fastball_velo'] = away_pitcher_stats.get('avg_fastball_velo', 92.5)
        features['home_pitcher_whiff_rate'] = home_pitcher_stats.get('whiff_rate', 0.25)
        features['away_pitcher_whiff_rate'] = away_pitcher_stats.get('whiff_rate', 0.25)
        features['home_pitcher_zone_rate'] = home_pitcher_stats.get('zone_rate', 0.50)
        features['away_pitcher_zone_rate'] = away_pitcher_stats.get('zone_rate', 0.50)
        features['home_pitcher_exit_velo_against'] = home_pitcher_stats.get('avg_exit_velo_against', 88.0)
        features['away_pitcher_exit_velo_against'] = away_pitcher_stats.get('avg_exit_velo_against', 88.0)
        
        # Pitcher advantage features
        features['pitching_velo_advantage'] = pitcher_advantages.get('fastball_velo_diff', 0)
        features['pitching_control_advantage'] = pitcher_advantages.get('control_advantage', 0)
        features['pitching_stuff_advantage'] = pitcher_advantages.get('stuff_advantage', 0)
        features['overall_pitching_advantage'] = pitcher_advantages.get('overall_pitching_advantage', 0)
        
        # Team Offensive Statcast features
        for key, value in home_stats.items():
            if pd.notna(value) and 'exit_velocity' in key or 'barrel_rate' in key or 'hard_hit' in key:
                features[f'home_{key}'] = value
        
        for key, value in away_stats.items():
            if pd.notna(value) and 'exit_velocity' in key or 'barrel_rate' in key or 'hard_hit' in key:
                features[f'away_{key}'] = value
        
        # Record-based features - ensure they're numeric
        home_win_pct = home_record.get('win_pct', 0.500)
        away_win_pct = away_record.get('win_pct', 0.500)
        
        if isinstance(home_win_pct, str):
            try:
                home_win_pct = float(home_win_pct)
            except:
                home_win_pct = 0.500
        
        if isinstance(away_win_pct, str):
            try:
                away_win_pct = float(away_win_pct)
            except:
                away_win_pct = 0.500
        
        features['home_win_pct'] = home_win_pct
        features['away_win_pct'] = away_win_pct
        features['win_pct_diff'] = home_win_pct - away_win_pct
        
        # Recent form - ensure they're numeric
        home_recent_form = home_form.get('recent_form', 0.5)
        away_recent_form = away_form.get('recent_form', 0.5)
        
        if isinstance(home_recent_form, str):
            try:
                home_recent_form = float(home_recent_form)
            except:
                home_recent_form = 0.5
        
        if isinstance(away_recent_form, str):
            try:
                away_recent_form = float(away_recent_form)
            except:
                away_recent_form = 0.5
        
        features['home_recent_form'] = home_recent_form
        features['away_recent_form'] = away_recent_form
        features['form_diff'] = home_recent_form - away_recent_form
        
        # Home field advantage
        features['home_field_advantage'] = 1
        
        # Advanced team vs pitcher matchups
        home_offense_exit_velo = features.get('home_avg_exit_velocity', 88.0)
        away_pitcher_exit_allowed = features.get('away_pitcher_exit_velo_against', 88.0)
        if home_offense_exit_velo and away_pitcher_exit_allowed:
            features['home_offense_vs_away_pitcher'] = home_offense_exit_velo - away_pitcher_exit_allowed
        
        away_offense_exit_velo = features.get('away_avg_exit_velocity', 88.0)
        home_pitcher_exit_allowed = features.get('home_pitcher_exit_velo_against', 88.0)
        if away_offense_exit_velo and home_pitcher_exit_allowed:
            features['away_offense_vs_home_pitcher'] = away_offense_exit_velo - home_pitcher_exit_allowed
        
        return features
    
    def prepare_training_data(self, force_real_data=False):
        """Prepare training data using real historical game results"""
        print("Preparing training data from recent games...")
        
        # Get data from the last 60 days for training
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        try:
            # Get all games from this period
            print(f"Fetching game data from {start_str} to {end_str}...")
            game_data = pb.statcast(start_dt=start_str, end_dt=end_str)
            
            if game_data.empty:
                if force_real_data:
                    raise Exception("No real training data available and force_real_data=True")
                print("No training data available, using synthetic data...")
                return self._create_synthetic_training_data()
            
            # Get unique games
            games = game_data.groupby('game_pk').agg({
                'home_team': 'first',
                'away_team': 'first', 
                'home_score': 'first',
                'away_score': 'first'
            }).reset_index()
            
            training_data = []
            
            print(f"Processing {len(games)} games for training...")
            
            # Process fewer games but with real data
            max_games = min(50, len(games))  # Limit for performance
            
            for idx, game in games.head(max_games).iterrows():
                if idx % 10 == 0:
                    print(f"Processing game {idx + 1}/{max_games}")
                
                # Map team names to abbreviations
                home_team_abbr = None
                away_team_abbr = None
                
                for abbr, full_name in self.savant_teams.items():
                    if game['home_team'] == full_name:
                        home_team_abbr = abbr
                    if game['away_team'] == full_name:
                        away_team_abbr = abbr
                
                if not home_team_abbr or not away_team_abbr:
                    continue
                
                # Create features for this game
                try:
                    features = self.create_features(home_team_abbr, away_team_abbr)
                    
                    # Determine outcome
                    home_wins = 1 if game['home_score'] > game['away_score'] else 0
                    features['outcome'] = home_wins
                    
                    training_data.append(features)
                except Exception as e:
                    print(f"Error creating features for game {idx}: {e}")
                    continue
                
                # Stop if we have enough data
                if len(training_data) >= 30:
                    break
            
            if not training_data:
                if force_real_data:
                    raise Exception("Could not create any real training data and force_real_data=True")
                print("No valid training data found, using synthetic data...")
                return self._create_synthetic_training_data()
            
            print(f"✅ Created {len(training_data)} training samples from REAL games")
            return pd.DataFrame(training_data)
            
        except Exception as e:
            print(f"Error preparing training data: {e}")
            if force_real_data:
                raise Exception(f"Could not get real training data: {e}")
            print("Falling back to synthetic training data...")
            return self._create_synthetic_training_data()
    
    def _create_synthetic_training_data(self):
        """Create MINIMAL synthetic training data as last resort fallback"""
        print("⚠️  WARNING: Using synthetic training data as fallback!")
        print("This should only happen if no real historical data is available.")
        
        teams = list(self.savant_teams.keys())
        training_data = []
        
        # Create a much smaller synthetic dataset
        for i in range(50):  # Reduced from 200
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Create minimal features that match what real data would provide
            features = {
                'home_win_pct': np.random.uniform(0.4, 0.6),
                'away_win_pct': np.random.uniform(0.4, 0.6),
                'home_recent_form': np.random.uniform(0.3, 0.7),
                'away_recent_form': np.random.uniform(0.3, 0.7),
                'home_field_advantage': 1,
                # Minimal pitcher features
                'home_pitcher_fastball_velo': np.random.uniform(90, 96),
                'away_pitcher_fastball_velo': np.random.uniform(90, 96),
                'overall_pitching_advantage': np.random.uniform(-0.2, 0.2)
            }
            
            features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
            features['form_diff'] = features['home_recent_form'] - features['away_recent_form']
            
            # Determine outcome based on realistic baseball factors
            home_win_prob = (
                0.54 +  # Base home field advantage
                (features['win_pct_diff'] * 0.3) +  # Record matters
                (features['form_diff'] * 0.2) +     # Recent form matters
                (features['overall_pitching_advantage'] * 0.4)  # Pitching is key
            )
            home_win_prob = max(0.25, min(0.75, home_win_prob))  # Realistic range
            
            features['outcome'] = 1 if np.random.random() < home_win_prob else 0
            training_data.append(features)
        
        print(f"Created minimal synthetic dataset with {len(training_data)} samples")
        print("🎯 Recommendation: Run during active season for real training data")
        
        return pd.DataFrame(training_data)
    
    def train_model(self):
        """Train the prediction model"""
        print("Training model with real baseball data...")
        
        # Get training data
        df = self.prepare_training_data()
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Separate features and target
        X = df.drop('outcome', axis=1)
        y = df['outcome']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Accuracy: {accuracy:.3f}")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return accuracy
    
    def predict_game(self, home_team, away_team):
        """Predict outcome of a single game with detailed pitcher analysis"""
        print(f"Predicting: {away_team} @ {home_team}")
        
        # Get probable pitchers info for display
        pitchers = self.get_probable_pitchers(home_team, away_team)
        
        # Create features (includes pitcher analysis)
        features = self.create_features(home_team, away_team)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Ensure all training columns are present
        for col in self.feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        # Reorder columns to match training
        feature_df = feature_df[self.feature_columns]
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        # Scale and predict
        features_scaled = self.scaler.transform(feature_df)
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_pitcher': pitchers['home_pitcher']['name'],
            'away_pitcher': pitchers['away_pitcher']['name'],
            'predicted_winner': home_team if prediction == 1 else away_team,
            'home_win_probability': probability[1],
            'away_win_probability': probability[0],
            'confidence': max(probability),
            'pitching_advantage': features.get('overall_pitching_advantage', 0),
            'key_factors': self._identify_key_factors(features)
        }
    
    def _identify_key_factors(self, features):
        """Identify the most important factors in the prediction"""
        key_factors = []
        
        # Pitching advantage
        pitching_adv = features.get('overall_pitching_advantage', 0)
        if abs(pitching_adv) > 0.05:
            if pitching_adv > 0:
                key_factors.append("Home pitcher has significant advantage")
            else:
                key_factors.append("Away pitcher has significant advantage")
        
        # Record advantage
        win_pct_diff = features.get('win_pct_diff', 0)
        if abs(win_pct_diff) > 0.1:
            if win_pct_diff > 0:
                key_factors.append("Home team has much better record")
            else:
                key_factors.append("Away team has much better record")
        
        # Recent form
        form_diff = features.get('form_diff', 0)
        if abs(form_diff) > 0.2:
            if form_diff > 0:
                key_factors.append("Home team in much better recent form")
            else:
                key_factors.append("Away team in much better recent form")
        
        # Home field advantage always applies
        key_factors.append("Home field advantage")
        
        return key_factors
    
    def get_mlb_odds(self):
        """Fetch MLB odds from the Odds API"""
        if not self.odds_api_key:
            print("No Odds API key provided. Using sample data.")
            return self._get_sample_odds()
        
        try:
            url = f"{self.odds_api_base_url}/sports/baseball_mlb/odds"
            params = {
                'apiKey': self.odds_api_key,
                'regions': 'us',
                'markets': 'h2h',
                'oddsFormat': 'american',
                'dateFormat': 'iso'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_odds_data(data)
            else:
                print(f"Error fetching odds: {response.status_code}")
                return self._get_sample_odds()
                
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return self._get_sample_odds()
    
    def _get_sample_odds(self):
        """Generate sample odds data"""
        return [
            {'home_team': 'NYY', 'away_team': 'BOS', 'home_odds': -150, 'away_odds': +130, 'commence_time': '2025-06-26T19:05:00Z'},
            {'home_team': 'LAD', 'away_team': 'SF', 'home_odds': -180, 'away_odds': +155, 'commence_time': '2025-06-26T20:10:00Z'},
            {'home_team': 'HOU', 'away_team': 'SEA', 'home_odds': -120, 'away_odds': +100, 'commence_time': '2025-06-26T20:10:00Z'},
            {'home_team': 'ATL', 'away_team': 'NYM', 'home_odds': -140, 'away_odds': +120, 'commence_time': '2025-06-26T19:20:00Z'},
            {'home_team': 'PHI', 'away_team': 'WSN', 'home_odds': -200, 'away_odds': +170, 'commence_time': '2025-06-26T19:05:00Z'}
        ]
    
    def _parse_odds_data(self, odds_data):
        """Parse odds data from API response"""
        parsed_games = []
        
        for game in odds_data:
            try:
                home_team = self._map_team_name(game['home_team'])
                away_team = self._map_team_name(game['away_team'])
                
                if not home_team or not away_team:
                    continue
                
                bookmaker = game['bookmakers'][0] if game['bookmakers'] else None
                if not bookmaker:
                    continue
                
                h2h_market = None
                for market in bookmaker['markets']:
                    if market['key'] == 'h2h':
                        h2h_market = market
                        break
                
                if not h2h_market:
                    continue
                
                home_odds = None
                away_odds = None
                
                for outcome in h2h_market['outcomes']:
                    team_abbr = self._map_team_name(outcome['name'])
                    if team_abbr == home_team:
                        home_odds = outcome['price']
                    elif team_abbr == away_team:
                        away_odds = outcome['price']
                
                if home_odds and away_odds:
                    parsed_games.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'commence_time': game['commence_time']
                    })
                    
            except Exception as e:
                print(f"Error parsing game: {e}")
                continue
        
        return parsed_games
    
    def _map_team_name(self, team_name):
        """Enhanced team name mapping"""
        # Direct mapping
        direct_mapping = self.team_mappings.get(team_name)
        if direct_mapping:
            return direct_mapping
        
        # Fuzzy matching for common variations
        fuzzy_mappings = {
            'Yankees': 'NYY', 'Red Sox': 'BOS', 'Dodgers': 'LAD', 'Giants': 'SF',
            'Astros': 'HOU', 'Braves': 'ATL', 'Mets': 'NYM', 'Phillies': 'PHI',
            'Rays': 'TB', 'Blue Jays': 'TOR', 'White Sox': 'CWS', 'Twins': 'MIN',
            'Athletics': 'OAK', 'Mariners': 'SEA', 'Rangers': 'TEX', 'Angels': 'LAA',
            'Guardians': 'CLE', 'Tigers': 'DET', 'Royals': 'KC', 'Brewers': 'MIL',
            'Cardinals': 'STL', 'Cubs': 'CHC', 'Reds': 'CIN', 'Pirates': 'PIT',
            'Nationals': 'WSN', 'Marlins': 'MIA', 'Rockies': 'COL', 'Diamondbacks': 'ARI',
            'Padres': 'SD', 'Orioles': 'BAL'
        }
        
        for variation, abbr in fuzzy_mappings.items():
            if variation in team_name:
                return abbr
        
        return None
    
    def american_odds_to_probability(self, odds):
        """Convert American odds to probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def compare_predictions_with_odds(self):
        """Compare model predictions with betting odds using real games"""
        print("\n" + "="*60)
        print("COMPARING STATCAST MODEL VS BETTING ODDS")
        print("="*60)
        
        # Get real upcoming games
        upcoming_games = self.get_todays_games()
        
        # Get odds (which should match the upcoming games)
        odds_games = self.get_mlb_odds()
        
        if not odds_games:
            print("No odds data available, analyzing upcoming games without odds...")
            # Still make predictions for the real games
            for home_team, away_team in upcoming_games:
                print(f"\n🏟️ Analyzing: {away_team} @ {home_team}")
                prediction = self.predict_game(home_team, away_team)
                
                print(f"📊 STATCAST MODEL:")
                print(f"   {prediction['home_team']}: {prediction['home_win_probability']*100:.1f}%")
                print(f"   {prediction['away_team']}: {prediction['away_win_probability']*100:.1f}%")
                print(f"   Winner: {prediction['predicted_winner']} ({prediction['confidence']*100:.1f}% confidence)")
            
            return []
        
        comparisons = []
        
        # Try to match odds games with upcoming games
        for game in odds_games:
            print(f"\n🏟️ Analyzing: {game['away_team']} @ {game['home_team']}")
            
            # Get model prediction
            prediction = self.predict_game(game['home_team'], game['away_team'])
            
            # Convert odds to probabilities
            home_odds_prob = self.american_odds_to_probability(game['home_odds'])
            away_odds_prob = self.american_odds_to_probability(game['away_odds'])
            
            # Normalize probabilities
            total_prob = home_odds_prob + away_odds_prob
            home_odds_prob_norm = home_odds_prob / total_prob
            away_odds_prob_norm = away_odds_prob / total_prob
            
            comparison = {
                'game': f"{game['away_team']} @ {game['home_team']}",
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_pitcher': prediction['home_pitcher'],
                'away_pitcher': prediction['away_pitcher'],
                'model_home_prob': prediction['home_win_probability'],
                'model_away_prob': prediction['away_win_probability'],
                'predicted_winner': prediction['predicted_winner'],
                'odds_home_prob': home_odds_prob_norm,
                'odds_away_prob': away_odds_prob_norm,
                'odds_favorite': game['home_team'] if home_odds_prob_norm > away_odds_prob_norm else game['away_team'],
                'home_odds': game['home_odds'],
                'away_odds': game['away_odds'],
                'prob_diff_home': prediction['home_win_probability'] - home_odds_prob_norm,
                'prob_diff_away': prediction['away_win_probability'] - away_odds_prob_norm,
                'agreement': prediction['predicted_winner'] == (game['home_team'] if home_odds_prob_norm > away_odds_prob_norm else game['away_team']),
                'confidence': prediction['confidence']
            }
            
            comparisons.append(comparison)
            
            # Display results
            print(f"📊 STATCAST MODEL:")
            print(f"   {prediction['home_team']}: {prediction['home_win_probability']*100:.1f}%")
            print(f"   {prediction['away_team']}: {prediction['away_win_probability']*100:.1f}%")
            print(f"   Winner: {prediction['predicted_winner']} ({prediction['confidence']*100:.1f}% confidence)")
            
            print(f"\n💰 BETTING MARKET:")
            print(f"   {game['home_team']}: {game['home_odds']:+d} ({home_odds_prob_norm*100:.1f}%)")
            print(f"   {game['away_team']}: {game['away_odds']:+d} ({away_odds_prob_norm*100:.1f}%)")
            print(f"   Favorite: {comparison['odds_favorite']}")
            
            print(f"\n🔍 COMPARISON:")
            agreement_emoji = "✅" if comparison['agreement'] else "❌"
            print(f"   {agreement_emoji} Agreement: {comparison['agreement']}")
            print(f"   Home Difference: {comparison['prob_diff_home']*100:+.1f}%")
            print(f"   Away Difference: {comparison['prob_diff_away']*100:+.1f}%")
            
            # Value opportunities
            max_diff = max(abs(comparison['prob_diff_home']), abs(comparison['prob_diff_away']))
            if max_diff > 0.05:
                if abs(comparison['prob_diff_home']) > abs(comparison['prob_diff_away']):
                    value_team = game['home_team']
                    value_diff = comparison['prob_diff_home']
                else:
                    value_team = game['away_team']  
                    value_diff = comparison['prob_diff_away']
                
                if value_diff > 0:
                    print(f"   💡 VALUE BET: Model sees {value_team} as {value_diff*100:.1f}% more likely than market")
        
        # Summary
        if comparisons:
            print(f"\n📈 SUMMARY:")
            print("-" * 40)
            total = len(comparisons)
            agreements = sum(1 for c in comparisons if c['agreement'])
            print(f"Games Analyzed: {total}")
            print(f"Model-Market Agreement: {agreements}/{total} ({agreements/total*100:.1f}%)")
            
            avg_home_diff = np.mean([c['prob_diff_home'] for c in comparisons])
            avg_away_diff = np.mean([c['prob_diff_away'] for c in comparisons])
            
            print(f"Average Home Team Difference: {avg_home_diff*100:+.1f}%")
            print(f"Average Away Team Difference: {avg_away_diff*100:+.1f}%")
        
        return comparisons
    
    def get_todays_games(self):
        """Get tomorrow's actual MLB schedule"""
        try:
            # Get tomorrow's date
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Fetching games for {tomorrow}...")
            
            # Try MLB Stats API approach
            try:
                import requests
                url = 'https://statsapi.mlb.com/api/v1/schedule'
                params = {
                    'sportId': '1',
                    'date': tomorrow,
                    'hydrate': 'team'
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    games = []
                    
                    for date_data in data.get('dates', []):
                        for game in date_data.get('games', []):
                            home_team_name = game['teams']['home']['team']['name']
                            away_team_name = game['teams']['away']['team']['name']
                            
                            # Map to abbreviations
                            home_abbr = self._map_team_name(home_team_name)
                            away_abbr = self._map_team_name(away_team_name)
                            
                            if home_abbr and away_abbr:
                                games.append((home_abbr, away_abbr))
                    
                    if games:
                        print(f"Found {len(games)} games for {tomorrow} from MLB API")
                        return games
                
            except Exception as e:
                print(f"Error with MLB API: {e}")
            
            # Final fallback: Use sample games but warn user
            print(f"⚠️  Could not fetch real schedule for {tomorrow}")
            print("Using sample games for demonstration:")
            
            sample_games = [
                ('NYY', 'BOS'),
                ('LAD', 'SF'), 
                ('HOU', 'SEA'),
                ('ATL', 'NYM'),
                ('PHI', 'WSN')
            ]
            
            return sample_games
            
        except Exception as e:
            print(f"Error getting tomorrow's games: {e}")
            print("Using sample games...")
            return [
                ('NYY', 'BOS'),
                ('LAD', 'SF'),
                ('HOU', 'SEA'),
                ('ATL', 'NYM'),
                ('PHI', 'WSN')
            ]
