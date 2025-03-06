import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.endpoints import PlayerGameLog, TeamGameLog
from nba_api.stats.static import players, teams

# Function to predict player stats using full season stats with added weight for last 5 games and adjusted defensive impact
def predict_weighted_player_stat(player_name, stat, opponent_team, n_estimators=200, simulations=10, weight_recent=1.5, defense_weight=0.5):
    player = [p for p in players.get_players() if p['full_name'] == player_name][0]
    player_id = player['id']
    player_games = PlayerGameLog(player_id=player_id, season="2023-24", season_type_all_star="Regular Season").get_data_frames()[0]
    
    # Ensure sorting is done correctly to get the full season stats
    player_games = player_games.sort_values(by='GAME_DATE', ascending=False)
    player_games['MIN'] = player_games['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x))
    stats = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM', 'STL', 'BLK', 'TOV']
    full_season_games = player_games[stats].to_dict(orient='list')
    
    X = np.array([full_season_games['MIN'], full_season_games['FGA'], full_season_games['FGM'], full_season_games['FG3A'], full_season_games['FG3M'], full_season_games['FTA'], full_season_games['FTM'], full_season_games['STL'], full_season_games['BLK'], full_season_games['TOV']]).T
    y = np.array(full_season_games[stat])
    
    if len(X) < 10:
        raise ValueError(f"Not enough data for reliable prediction for {player_name}.")
    
    # Weight the last 5 games more heavily
    recent_games_X = X[:5] * weight_recent
    recent_games_y = y[:5] * weight_recent
    X_weighted = np.vstack((X, recent_games_X))
    y_weighted = np.concatenate((y, recent_games_y))
    
    # Adjust for opponent defensive impact (less impact now)
    opponent = [t for t in teams.get_teams() if t['full_name'] == opponent_team][0]
    opponent_id = opponent['id']
    opponent_games = TeamGameLog(team_id=opponent_id, season="2023-24", season_type_all_star="Regular Season").get_data_frames()[0]
    
    defensive_rating = opponent_games['PTS'].mean()  # Approximate defensive impact
    steals_per_game = opponent_games['STL'].mean()
    blocks_per_game = opponent_games['BLK'].mean()
    turnovers_forced = opponent_games['TOV'].mean()
    
    defense_adjustment = max(0.85, 1 - defense_weight * ((defensive_rating / 120) + (steals_per_game / 15) + (blocks_per_game / 10) + (turnovers_forced / 15)))  # Reduce defensive weight
    
    simulation_results = []
    for _ in range(simulations):
        X_train, X_test, y_train, y_test = train_test_split(X_weighted, y_weighted, test_size=0.2, random_state=None)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=None)
        model.fit(X_train, y_train)
        predicted_stat = model.predict([X.mean(axis=0)])[0] * defense_adjustment  # Apply adjusted defense impact
        simulation_results.append(predicted_stat)
    
    return np.mean(simulation_results)  # Return average of simulations

# Predict Stephen Curry's points (vs Nets)
predicted_curry_points = predict_weighted_player_stat("Stephen Curry", "PTS", "Brooklyn Nets")
print(f"Predicted Points for Stephen Curry vs. Nets: {predicted_curry_points:.2f}")

# Predict Luka Dončić's points (now on Lakers, vs Knicks)
predicted_luka_points = predict_weighted_player_stat("Luka Dončić", "PTS", "New York Knicks")
print(f"Predicted Points for Luka Dončić vs. Knicks: {predicted_luka_points:.2f}")

# Predict LeBron James' points (vs Knicks)
predicted_lebron_points = predict_weighted_player_stat("LeBron James", "PTS", "New York Knicks")
print(f"Predicted Points for LeBron James vs. Knicks: {predicted_lebron_points:.2f}")
