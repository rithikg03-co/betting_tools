"""
NBA Player Prop Betting Analyzer with ML and Monte Carlo Simulation
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NBA API imports
from nba_api.stats.endpoints import PlayerGameLog, TeamGameLog, LeagueDashTeamStats
from nba_api.stats.static import players, teams

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import threading


class NBAPlayerPropAnalyzer:
    def __init__(self):
        self.current_season = "2025-26"
        self.scaler = StandardScaler()
        self.model = None
        
    def get_player_recent_games(self, player_name, num_games=None):
        """Fetch recent game logs for a player"""
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                return None
            
            player_id = player_dict[0]['id']
            gamelog = PlayerGameLog(player_id=player_id, season=self.current_season, season_type_all_star='Regular Season')
            df = gamelog.get_data_frames()[0]
            
            # Return all games if num_games is None, otherwise return specified number
            if num_games is None:
                return df
            return df.head(num_games)
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return None
    
    def _resolve_team(self, team_input):
        """Resolve a team string ("BOS" or "Boston Celtics") to {team_id, abbr, full_name}."""
        if team_input is None:
            return None

        raw = str(team_input).strip()
        if not raw:
            return None

        all_teams = teams.get_teams()

        # 1) Exact abbreviation match (fast path)
        if len(raw) == 3:
            abbr = raw.upper()
            for t in all_teams:
                if t.get('abbreviation', '').upper() == abbr:
                    return {'team_id': t['id'], 'abbr': t['abbreviation'].upper(), 'full_name': t.get('full_name', '')}

        # 2) nba_api helpers (they expect "full name" style input)
        # These can return multiple matches; prefer an exact case-insensitive full_name match.
        candidates = []
        try:
            candidates.extend(teams.find_teams_by_full_name(raw))
        except Exception:
            pass
        try:
            candidates.extend(teams.find_teams_by_nickname(raw))
        except Exception:
            pass

        raw_lower = raw.lower()

        def score_team(t):
            full = (t.get('full_name') or '').lower()
            nick = (t.get('nickname') or '').lower()
            city = (t.get('city') or '').lower()
            abbr = (t.get('abbreviation') or '').lower()

            # Higher is better
            if full == raw_lower:
                return 100
            if abbr == raw_lower:
                return 95
            if nick == raw_lower:
                return 90
            if raw_lower in full:
                return 80
            if raw_lower in nick:
                return 70
            if raw_lower in city:
                return 60
            return 0

        # 3) Fallback: scan all teams
        if not candidates:
            candidates = all_teams

        best = max(candidates, key=score_team, default=None)
        if best and score_team(best) > 0:
            return {'team_id': best['id'], 'abbr': best['abbreviation'].upper(), 'full_name': best.get('full_name', '')}

        return None

    def get_team_defense_stats(self, team_input):
        """Get defensive statistics for opponent team.

        Accepts either TEAM_ABBREVIATION (e.g., "BOS") or a team name (e.g., "Boston Celtics").
        """
        resolved = self._resolve_team(team_input)
        if not resolved:
            print(f"[team-defense] Could not resolve team input: {team_input!r}")
            return None

        try:
            # Ask specifically for defensive measures (important; otherwise DEF_RATING may not exist)
            team_stats = LeagueDashTeamStats(
                season=self.current_season,
                season_type_all_star='Regular Season',
                # "Advanced" contains both DEF_RATING and PACE; "Defense" often does NOT include PACE.
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame',
                # PaceAdjust is a Y/N flag in nba_api parameter mapping.
                pace_adjust='N'
            )
            df = team_stats.get_data_frames()[0]

            # Match by TEAM_ID first (most reliable)
            team_data = df[df.get('TEAM_ID') == resolved['team_id']] if 'TEAM_ID' in df.columns else pd.DataFrame()
            if team_data.empty:
                # Fallback to abbreviation match
                if 'TEAM_ABBREVIATION' in df.columns:
                    team_data = df[df['TEAM_ABBREVIATION'].astype(str).str.upper() == resolved['abbr']]

            if team_data.empty:
                print(f"[team-defense] Team resolved to {resolved}, but not found in LeagueDashTeamStats output")
                return None

            row = team_data.iloc[0]

            def _num(col, default=np.nan):
                if col not in team_data.columns:
                    return default
                try:
                    return pd.to_numeric(row[col], errors='coerce')
                except Exception:
                    return default

            def_rating = _num('DEF_RATING')
            pace = _num('PACE')

            # This column is not consistently present across nba_api versions/endpoints
            opp_pts = _num('OPP_PTS')

            # Fallback: estimate points allowed per game using DEF_RATING and PACE
            if np.isnan(opp_pts) and not np.isnan(def_rating) and not np.isnan(pace):
                opp_pts = (def_rating * pace) / 100.0

            return {
                'opp_def_rating': float(def_rating) if not np.isnan(def_rating) else 112.0,
                'opp_pace': float(pace) if not np.isnan(pace) else 99.0,
                'opp_pts_allowed': float(opp_pts) if not np.isnan(opp_pts) else 112.0,
                'opp_team_abbr_resolved': resolved['abbr'],
                'opp_team_name_resolved': resolved['full_name'],
            }
        except Exception as e:
            # Don’t silently swallow: defaulting makes the UI “look wrong” with no clue why
            print(f"Error fetching team defense for {resolved}: {e}")
            return None
    
    def calculate_recent_trend(self, game_log, stat_column):
        """Calculate weighted recent performance trend"""
        if game_log is None or len(game_log) == 0:
            return 0
        
        recent_stats = game_log[stat_column].head(5).values
        weights = np.array([0.35, 0.25, 0.20, 0.15, 0.05])[:len(recent_stats)]
        weights = weights / weights.sum()
        
        return np.average(recent_stats, weights=weights)
    
    def calculate_variance(self, game_log, stat_column):
        """Calculate statistical variance for Monte Carlo"""
        if game_log is None or len(game_log) < 3:
            return 5.0  # Default variance
        
        return game_log[stat_column].head(10).std()
    
    def engineer_features(self, player_name, opponent_team, stat_type, home_away='HOME'):
        """Engineer features for ML model"""
        # Get recent games for trends
        game_log_recent = self.get_player_recent_games(player_name, 15)
        
        # Get ALL season games for accurate season average
        game_log_full = self.get_player_recent_games(player_name, None)
        
        if game_log_recent is None or game_log_full is None:
            return None
        
        # Map stat type to NBA API columns
        stat_mapping = {
            'points': 'PTS',
            'rebounds': 'REB',
            'assists': 'AST',
            'threes': 'FG3M',
            'pts+rebs': ['PTS', 'REB'],
            'pts+asts': ['PTS', 'AST'],
            'rebs+asts': ['REB', 'AST']
        }
        
        stat_col = stat_mapping.get(stat_type.lower(), 'PTS')
        
        # Calculate combined stats if needed
        if isinstance(stat_col, list):
            game_log_recent['TARGET_STAT'] = game_log_recent[stat_col].sum(axis=1)
            game_log_full['TARGET_STAT'] = game_log_full[stat_col].sum(axis=1)
            stat_col = 'TARGET_STAT'
        else:
            game_log_recent['TARGET_STAT'] = game_log_recent[stat_col]
            game_log_full['TARGET_STAT'] = game_log_full[stat_col]
        
        # Feature engineering
        features = {
            'recent_avg_5': self.calculate_recent_trend(game_log_recent, 'TARGET_STAT'),
            'recent_avg_10': game_log_recent['TARGET_STAT'].head(10).mean(),
            'season_avg': game_log_full['TARGET_STAT'].mean(),  # Use ALL games for season average
            'games_played': len(game_log_full),  # Track total games played
            'std_dev': self.calculate_variance(game_log_recent, 'TARGET_STAT'),
            'recent_min_avg': game_log_recent['MIN'].head(5).mean(),
            'home_away': 1 if home_away.upper() == 'HOME' else 0,
        }
        
        # Get opponent defensive stats
        opponent_stats = self.get_team_defense_stats(opponent_team)
        if opponent_stats:
            features.update(opponent_stats)
        else:
            # Keep the GUI running, but make it obvious (in console) why numbers look "stuck".
            resolved = self._resolve_team(opponent_team)
            print(
                "[team-defense] Falling back to defaults. "
                f"input={opponent_team!r} resolved={resolved} season={self.current_season}"
            )
            features.update({
                'opp_def_rating': 112.0,
                'opp_pace': 99.0,
                'opp_pts_allowed': 112.0,
                'opp_team_abbr_resolved': (resolved['abbr'] if resolved else str(opponent_team).strip().upper()),
                'opp_team_name_resolved': (resolved['full_name'] if resolved else '')
            })
        
        # Form indicator (last 3 games vs season average)
        recent_3 = game_log_recent['TARGET_STAT'].head(3).mean()
        features['hot_cold_factor'] = (recent_3 - features['season_avg']) / (features['std_dev'] + 1)
        
        return features, game_log_recent, stat_col
    
    def simulate_betting_line_movement(self, base_line):
        """Simulate betting market line movement (in production, use real API)"""
        # Simulate market movement
        movement = np.random.normal(0, 1.5)
        current_line = base_line + movement
        
        # Simulate opening line
        opening_line = base_line - np.random.uniform(-2, 2)
        
        # Calculate implied probability shift
        line_shift = current_line - opening_line
        
        return {
            'opening_line': opening_line,
            'current_line': current_line,
            'line_movement': line_shift,
            'sharp_money_indicator': 1 if abs(line_shift) > 1.5 else 0
        }
    
    def run_monte_carlo_simulation(self, predicted_value, std_dev, prop_line, num_simulations=10000):
        """Run Monte Carlo simulation to estimate probability"""
        # Add some noise to std_dev based on prediction uncertainty
        adjusted_std = std_dev * 1.1
        
        # Run simulations
        simulations = np.random.normal(predicted_value, adjusted_std, num_simulations)
        
        # Calculate probabilities
        over_prob = (simulations > prop_line).sum() / num_simulations
        under_prob = 1 - over_prob
        
        # Calculate expected values
        avg_simulation = simulations.mean()
        median_simulation = np.median(simulations)
        
        # Confidence intervals
        ci_95_lower = np.percentile(simulations, 2.5)
        ci_95_upper = np.percentile(simulations, 97.5)
        
        return {
            'over_probability': over_prob,
            'under_probability': under_prob,
            'expected_value': avg_simulation,
            'median_value': median_simulation,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
            'simulations': simulations
        }
    
    def calculate_betting_recommendation(self, features, prop_line, betting_odds_over, betting_odds_under):
        """Generate betting recommendation with Kelly Criterion"""
        # Predict expected performance
        predicted_value = features['recent_avg_5']
        std_dev = features['std_dev']
        
        # Adjust prediction based on factors
        # Opponent defense adjustment
        def_adjustment = (115 - features['opp_def_rating']) / 100
        predicted_value *= (1 + def_adjustment * 0.1)
        
        # Pace adjustment
        pace_adjustment = (features['opp_pace'] - 99) / 100
        predicted_value *= (1 + pace_adjustment * 0.05)
        
        # Hot/cold adjustment
        predicted_value += features['hot_cold_factor'] * 0.5
        
        # Home/away adjustment
        if features['home_away'] == 1:
            predicted_value *= 1.02
        else:
            predicted_value *= 0.98
        
        # Run Monte Carlo simulation
        mc_results = self.run_monte_carlo_simulation(predicted_value, std_dev, prop_line)
        
        # Convert American odds to decimal
        def american_to_decimal(odds):
            if odds > 0:
                return (odds / 100) + 1
            else:
                return (100 / abs(odds)) + 1
        
        decimal_odds_over = american_to_decimal(betting_odds_over)
        decimal_odds_under = american_to_decimal(betting_odds_under)
        
        # Calculate implied probabilities from odds
        implied_prob_over = 1 / decimal_odds_over
        implied_prob_under = 1 / decimal_odds_under
        
        # Calculate edge (model probability - implied probability)
        edge_over = mc_results['over_probability'] - implied_prob_over
        edge_under = mc_results['under_probability'] - implied_prob_under
        
        # Kelly Criterion for bet sizing (fractional Kelly)
        kelly_fraction = 0.25  # Conservative
        kelly_over = max(0, kelly_fraction * edge_over / (decimal_odds_over - 1))
        kelly_under = max(0, kelly_fraction * edge_under / (decimal_odds_under - 1))
        
        # Determine recommendation
        min_edge = 0.05  # 5% minimum edge to recommend bet
        
        recommendation = {
            'predicted_value': predicted_value,
            'prop_line': prop_line,
            'edge_over': edge_over,
            'edge_under': edge_under,
            'kelly_over': kelly_over,
            'kelly_under': kelly_under,
            'mc_results': mc_results
        }
        
        if edge_over > min_edge and edge_over > edge_under:
            recommendation['bet'] = 'OVER'
            recommendation['confidence'] = min(edge_over * 10, 5)  # 1-5 scale
            recommendation['edge'] = edge_over
            recommendation['kelly_size'] = kelly_over
        elif edge_under > min_edge and edge_under > edge_over:
            recommendation['bet'] = 'UNDER'
            recommendation['confidence'] = min(edge_under * 10, 5)
            recommendation['edge'] = edge_under
            recommendation['kelly_size'] = kelly_under
        else:
            recommendation['bet'] = 'NO BET'
            recommendation['confidence'] = 0
            recommendation['edge'] = max(edge_over, edge_under)
            recommendation['kelly_size'] = 0
        
        return recommendation


class PropAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Player Prop Betting Analyzer")
        self.root.geometry("900x850")
        self.root.configure(bg='#1a1a2e')
        
        self.analyzer = NBAPlayerPropAnalyzer()
        
        # Load all NBA players for autocomplete
        self.all_players = [player['full_name'] for player in players.get_active_players()]
        self.all_players.sort()
        
        self.setup_ui()
    
    def setup_ui(self):
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='#eee', font=('Arial', 10))
        style.configure('TButton', background='#0f3460', foreground='white', font=('Arial', 10, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'), foreground='#16db93')
        
        # Header
        header = ttk.Label(self.root, text="🏀 NBA Prop Bet Analyzer", style='Header.TLabel')
        header.pack(pady=20)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill='x', pady=10)
        
        # Player name with autocomplete
        ttk.Label(input_frame, text="Player Name:").grid(row=0, column=0, sticky='w', pady=5)
        self.player_entry = ttk.Entry(input_frame, width=30)
        self.player_entry.grid(row=0, column=1, padx=5, pady=5)
        self.player_entry.insert(0, "LeBron James")
        
        # Autocomplete listbox
        self.autocomplete_listbox = tk.Listbox(input_frame, height=6, bg='#16213e', fg='#eee', 
                                               font=('Arial', 9), selectmode=tk.SINGLE)
        self.autocomplete_listbox.grid(row=1, column=1, padx=5, pady=0, sticky='ew')
        self.autocomplete_listbox.grid_remove()  # Hide initially
        
        # Bind events for autocomplete
        self.player_entry.bind('<KeyRelease>', self.on_player_keyrelease)
        self.autocomplete_listbox.bind('<<ListboxSelect>>', self.on_autocomplete_select)
        self.autocomplete_listbox.bind('<Double-Button-1>', self.on_autocomplete_select)
        
        # Click outside to close autocomplete
        self.root.bind('<Button-1>', self.close_autocomplete)
        
        # Opponent team
        ttk.Label(input_frame, text="Opponent Team:").grid(row=2, column=0, sticky='w', pady=5)
        self.opponent_entry = ttk.Entry(input_frame, width=30)
        self.opponent_entry.grid(row=2, column=1, padx=5, pady=5)
        self.opponent_entry.insert(0, "LAL")
        
        # Stat type
        ttk.Label(input_frame, text="Stat Type:").grid(row=3, column=0, sticky='w', pady=5)
        self.stat_var = tk.StringVar(value="points")
        stat_options = ['points', 'rebounds', 'assists', 'threes', 'pts+rebs', 'pts+asts', 'rebs+asts']
        stat_menu = ttk.Combobox(input_frame, textvariable=self.stat_var, values=stat_options, width=28)
        stat_menu.grid(row=3, column=1, padx=5, pady=5)
        
        # Prop line
        ttk.Label(input_frame, text="Prop Line:").grid(row=4, column=0, sticky='w', pady=5)
        self.line_entry = ttk.Entry(input_frame, width=30)
        self.line_entry.grid(row=4, column=1, padx=5, pady=5)
        self.line_entry.insert(0, "25.5")
        
        # Odds
        ttk.Label(input_frame, text="Over Odds (American):").grid(row=5, column=0, sticky='w', pady=5)
        self.over_odds_entry = ttk.Entry(input_frame, width=30)
        self.over_odds_entry.grid(row=5, column=1, padx=5, pady=5)
        self.over_odds_entry.insert(0, "-110")
        
        ttk.Label(input_frame, text="Under Odds (American):").grid(row=6, column=0, sticky='w', pady=5)
        self.under_odds_entry = ttk.Entry(input_frame, width=30)
        self.under_odds_entry.grid(row=6, column=1, padx=5, pady=5)
        self.under_odds_entry.insert(0, "-110")
        
        # Home/Away
        ttk.Label(input_frame, text="Home/Away:").grid(row=7, column=0, sticky='w', pady=5)
        self.location_var = tk.StringVar(value="HOME")
        location_menu = ttk.Combobox(input_frame, textvariable=self.location_var, values=['HOME', 'AWAY'], width=28)
        location_menu.grid(row=7, column=1, padx=5, pady=5)
        
        # Analyze button
        analyze_btn = tk.Button(main_frame, text="🎲 ANALYZE BET", command=self.analyze_prop,
                               bg='#16db93', fg='black', font=('Arial', 12, 'bold'),
                               relief='raised', bd=3, padx=20, pady=10)
        analyze_btn.pack(pady=15)
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill='both', expand=True, pady=10)
        
        ttk.Label(results_frame, text="Analysis Results:", font=('Arial', 12, 'bold')).pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=80, height=20,
                                                      bg='#16213e', fg='#eee',
                                                      font=('Courier', 10))
        self.results_text.pack(fill='both', expand=True, pady=5)
    
    def on_player_keyrelease(self, event):
        """Handle autocomplete as user types"""
        # Ignore navigation keys
        if event.keysym in ('Up', 'Down', 'Left', 'Right', 'Return', 'Escape'):
            return
        
        typed = self.player_entry.get().strip()
        
        if len(typed) < 2:  # Only show autocomplete after 2 characters
            self.autocomplete_listbox.grid_remove()
            return
        
        # Filter players
        matches = [player for player in self.all_players 
                   if typed.lower() in player.lower()][:10]  # Limit to 10 results
        
        # Update listbox
        self.autocomplete_listbox.delete(0, tk.END)
        
        if matches:
            for player in matches:
                self.autocomplete_listbox.insert(tk.END, player)
            self.autocomplete_listbox.grid()
        else:
            self.autocomplete_listbox.grid_remove()
    
    def on_autocomplete_select(self, event):
        """Handle player selection from autocomplete"""
        if self.autocomplete_listbox.curselection():
            selected = self.autocomplete_listbox.get(self.autocomplete_listbox.curselection()[0])
            self.player_entry.delete(0, tk.END)
            self.player_entry.insert(0, selected)
            self.autocomplete_listbox.grid_remove()
    
    def close_autocomplete(self, event):
        """Close autocomplete when clicking outside"""
        # Check if click was outside the listbox and entry
        if event.widget not in (self.autocomplete_listbox, self.player_entry):
            self.autocomplete_listbox.grid_remove()
    
    def analyze_prop(self):
        """Main analysis function"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🔄 Analyzing... Please wait...\n\n")
        self.root.update()
        
        # Run analysis in thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        try:
            player_name = self.player_entry.get()
            opponent = self.opponent_entry.get().upper()
            stat_type = self.stat_var.get()
            prop_line = float(self.line_entry.get())
            over_odds = int(self.over_odds_entry.get())
            under_odds = int(self.under_odds_entry.get())
            location = self.location_var.get()
            
            # Engineer features
            result = self.analyzer.engineer_features(player_name, opponent, stat_type, location)
            
            if result is None:
                self.root.after(0, lambda: self.results_text.insert(tk.END, "❌ Error: Could not fetch player data\n"))
                return
            
            features, game_log, stat_col = result
            
            # Get recommendation
            recommendation = self.analyzer.calculate_betting_recommendation(
                features, prop_line, over_odds, under_odds
            )
            
            # Display results
            self.root.after(0, lambda: self.display_results(
                player_name, stat_type, prop_line, features, 
                game_log, stat_col, recommendation
            ))
            
        except Exception as e:
            self.root.after(0, lambda: self.results_text.insert(tk.END, f"❌ Error: {str(e)}\n"))
    
    def display_results(self, player_name, stat_type, prop_line, features, game_log, stat_col, rec):
        """Display analysis results"""
        self.results_text.delete(1.0, tk.END)
        
        output = f"""
{'='*70}
  NBA PLAYER PROP ANALYSIS - {player_name.upper()}
{'='*70}

📊 PROP DETAILS:
   Stat Type: {stat_type.upper()}
   Prop Line: {prop_line}
   Player: {player_name}
   
{'='*70}
📈 STATISTICAL ANALYSIS:
{'='*70}

Recent Performance (Last 5 games avg): {features['recent_avg_5']:.2f}
Recent Performance (Last 10 games avg): {features['recent_avg_10']:.2f}
Season Average ({features['games_played']} games): {features['season_avg']:.2f}
Standard Deviation: {features['std_dev']:.2f}
Form Indicator: {'🔥 HOT' if features['hot_cold_factor'] > 0.5 else '🧊 COLD' if features['hot_cold_factor'] < -0.5 else '➡️  NEUTRAL'}

Opponent Defense Rating: {features['opp_def_rating']:.1f}
Opponent Pace: {features['opp_pace']:.1f}
Opponent Resolved: {features.get('opp_team_abbr_resolved','')} {features.get('opp_team_name_resolved','')}

{'='*70}
🎲 MONTE CARLO SIMULATION RESULTS (10,000 iterations):
{'='*70}

Model Prediction: {rec['predicted_value']:.2f}
Expected Value: {rec['mc_results']['expected_value']:.2f}
Median Outcome: {rec['mc_results']['median_value']:.2f}

95% Confidence Interval: [{rec['mc_results']['ci_95_lower']:.2f}, {rec['mc_results']['ci_95_upper']:.2f}]

Probability OVER {prop_line}: {rec['mc_results']['over_probability']*100:.1f}%
Probability UNDER {prop_line}: {rec['mc_results']['under_probability']*100:.1f}%

{'='*70}
💰 BETTING EDGE ANALYSIS:
{'='*70}

Edge on OVER: {rec['edge_over']*100:+.2f}%
Edge on UNDER: {rec['edge_under']*100:+.2f}%

Kelly Criterion Bet Size (OVER): {rec['kelly_over']*100:.2f}% of bankroll
Kelly Criterion Bet Size (UNDER): {rec['kelly_under']*100:.2f}% of bankroll

{'='*70}
🎯 RECOMMENDATION:
{'='*70}

"""
        
        if rec['bet'] == 'OVER':
            output += f"""
✅ BET: OVER {prop_line}
💪 Confidence: {'⭐' * int(rec['confidence'])} ({rec['confidence']:.1f}/5.0)
📊 Edge: {rec['edge']*100:.2f}%
💵 Suggested Bet Size: {rec['kelly_size']*100:.2f}% of bankroll

REASONING: Model predicts {rec['predicted_value']:.2f}, which is {rec['predicted_value']-prop_line:.2f} 
points above the line. Monte Carlo simulation shows {rec['mc_results']['over_probability']*100:.1f}% 
probability of hitting the OVER with positive expected value.
"""
        elif rec['bet'] == 'UNDER':
            output += f"""
✅ BET: UNDER {prop_line}
💪 Confidence: {'⭐' * int(rec['confidence'])} ({rec['confidence']:.1f}/5.0)
📊 Edge: {rec['edge']*100:.2f}%
💵 Suggested Bet Size: {rec['kelly_size']*100:.2f}% of bankroll

REASONING: Model predicts {rec['predicted_value']:.2f}, which is {prop_line-rec['predicted_value']:.2f} 
points below the line. Monte Carlo simulation shows {rec['mc_results']['under_probability']*100:.1f}% 
probability of hitting the UNDER with positive expected value.
"""
        else:
            output += f"""
⚠️  NO BET RECOMMENDED
📊 Insufficient Edge

The model does not identify sufficient edge (>5%) on either side of this bet.
Best edge found: {max(rec['edge_over'], rec['edge_under'])*100:.2f}%

Consider passing on this prop or waiting for better lines.
"""
        
        output += f"""
{'='*70}
📋 LAST 5 GAMES:
{'='*70}

"""
        
        for idx, row in game_log.head(5).iterrows():
            output += f"  {row['GAME_DATE']}: {row[stat_col]:.0f} vs {row['MATCHUP']}\n"
        
        output += f"\n{'='*70}\n"
        output += "⚠️  DISCLAIMER: This is for educational purposes. Gambling involves risk.\n"
        output += "    Always bet responsibly and within your means.\n"
        output += f"{'='*70}\n"
        
        self.results_text.insert(tk.END, output)


def main():
    root = tk.Tk()
    app = PropAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()