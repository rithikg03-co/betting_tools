import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.endpoints import PlayerGameLog, TeamGameLog, LeagueDashTeamStats
from nba_api.stats.static import players, teams
import random

# Fetch league-wide averages for better scaling
def get_league_averages():
    league_stats = LeagueDashTeamStats(season="2024-25", season_type_all_star="Regular Season").get_data_frames()[0]
    league_avg_def_rtg = league_stats['PTS'].mean()
    league_avg_steals = league_stats['STL'].mean()
    league_avg_blocks = league_stats['BLK'].mean()
    league_avg_turnovers = league_stats['TOV'].mean()
    return league_avg_def_rtg, league_avg_steals, league_avg_blocks, league_avg_turnovers

# Function to retrieve player home and away performance
def get_player_home_away_stats(player_name, season='2024-25'):
    player_dict = players.get_players()
    player = next((p for p in player_dict if p['full_name'].lower() == player_name.lower()), None)
    if not player:
        raise ValueError(f"Player {player_name} not found.")
    player_id = player['id']
    
    game_log = PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    game_log['Location'] = game_log['MATCHUP'].apply(lambda x: 'Home' if 'vs.' in x else 'Away')
    
    home_stats = game_log[game_log['Location'] == 'Home'].mean(numeric_only=True)
    away_stats = game_log[game_log['Location'] == 'Away'].mean(numeric_only=True)
    
    return home_stats, away_stats

# Function to retrieve opponent's home/away defensive performance
def get_opponent_home_away_defense(opponent_team, season='2024-25'):
    opponent = [t for t in teams.get_teams() if t['full_name'].lower() == opponent_team.lower()]
    if not opponent:
        raise ValueError(f"Opponent team {opponent_team} not found.")
    opponent_id = opponent[0]['id']
    
    opponent_games = TeamGameLog(team_id=opponent_id, season=season).get_data_frames()[0]
    opponent_games['Location'] = opponent_games['MATCHUP'].apply(lambda x: 'Home' if 'vs.' in x else 'Away')
    
    home_defense = opponent_games[opponent_games['Location'] == 'Home'].mean(numeric_only=True)
    away_defense = opponent_games[opponent_games['Location'] == 'Away'].mean(numeric_only=True)
    
    return home_defense, away_defense

# Function to predict player stats with weighted recent games and improved defensive & home/away impact
def predict_weighted_player_stat(player_name, stat, opponent_team, home_game, n_estimators=200, simulations=100, weight_recent=1.2, defense_weight=0.7):
    player = [p for p in players.get_players() if p['full_name'].lower() == player_name.lower()]
    if not player:
        raise ValueError(f"Player {player_name} not found.")
    player_id = player[0]['id']
    player_games = PlayerGameLog(player_id=player_id, season="2024-25", season_type_all_star="Regular Season").get_data_frames()[0]
    
    # Ensure sorting is done correctly to get the full season stats
    player_games = player_games.sort_values(by='GAME_DATE', ascending=False)
    player_games['MIN'] = player_games['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x))
    stats = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'STL', 'BLK', 'TOV']
    full_season_games = player_games[stats].to_dict(orient='list')
    
    X = np.array([full_season_games['MIN'], full_season_games['FGA'], full_season_games['FGM'], full_season_games['FG3A'], full_season_games['FG3M'], full_season_games['FTA'], full_season_games['FTM'], full_season_games['STL'], full_season_games['BLK'], full_season_games['TOV']]).T
    y = np.array(full_season_games[stat])
    
    if len(X) < 10:
        raise ValueError(f"Not enough data for reliable prediction for {player_name}.")
    
    # Adjust for opponent defensive impact considering home/away differences
    home_defense, away_defense = get_opponent_home_away_defense(opponent_team)
    opponent_defensive_stats = home_defense if home_game else away_defense
    
    defense_adjustment = 1 + defense_weight * ((110 - opponent_defensive_stats['PTS']) / 120)
    defense_adjustment = max(0.9, min(1.1, defense_adjustment))  # Clamp adjustment
    
    # Adjust for player home vs away performance
    home_stats, away_stats = get_player_home_away_stats(player_name)
    home_away_adjustment = (home_stats[stat] / away_stats[stat]) if home_game else (away_stats[stat] / home_stats[stat])
    home_away_adjustment = max(0.95, min(1.1, home_away_adjustment))  # Ensure realistic home advantage
    
    simulation_results = []
    for _ in range(simulations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=None)
        model.fit(X_train, y_train)
        predicted_stat = model.predict([X.mean(axis=0)])[0] * defense_adjustment * home_away_adjustment
        
        # Factor in game volatility (hot or cold games)
        game_variance = random.random()
        if game_variance < 0.15:
            predicted_stat *= random.uniform(1.2, 1.4)
        elif game_variance < 0.25 and game_variance > 0.15:
            predicted_stat *= random.uniform(0.6, 0.8)
        
        simulation_results.append(predicted_stat)
    
    return np.mean(simulation_results), simulation_results
