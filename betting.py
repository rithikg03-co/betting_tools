import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players

# Function to predict player stats using the most recent 10 games
def predict_player_stat(player_name, stat, n_estimators=200, simulations=10):
    player = [p for p in players.get_players() if p['full_name'] == player_name][0]
    player_id = player['id']
    player_games = PlayerGameLog(player_id=player_id, season="2023-24", season_type_all_star="Regular Season").get_data_frames()[0]
    
    # Ensure sorting is done correctly to get the latest 10 games
    player_games = player_games.sort_values(by='GAME_DATE', ascending=False).head(10)
    player_games['MIN'] = player_games['MIN'].astype(str).apply(lambda x: float(x.split(':')[0]) if ':' in x else float(x))
    stats = ['PTS', 'REB', 'MIN', 'FGA', 'FGM', 'FG3A', 'FG3M', 'FTA', 'FTM']
    recent_games = player_games[stats].to_dict(orient='list')
    
    X = np.array([recent_games['MIN'], recent_games['FGA'], recent_games['FGM'], recent_games['FG3A'], recent_games['FG3M'], recent_games['FTA'], recent_games['FTM']]).T
    y = np.array(recent_games[stat])
    
    if len(X) < 10:
        raise ValueError(f"Not enough data for reliable prediction for {player_name}.")
    
    simulation_results = []
    for _ in range(simulations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)  # Random state removed for variability
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=None)  # Random state removed for variability
        model.fit(X_train, y_train)
        predicted_stat = model.predict([X.mean(axis=0)])[0]  # Use the mean of the last 10 games for prediction
        simulation_results.append(predicted_stat)
    
    return np.mean(simulation_results)  # Return average of simulations

# Predict Isaiah Hartenstein's rebounds (vs Grizzlies)
predicted_hartenstein_rebounds = predict_player_stat("Isaiah Hartenstein", "REB")
print(f"Predicted Rebounds for Isaiah Hartenstein vs. Grizzlies: {predicted_hartenstein_rebounds:.2f}")

# Predict Nikola Jokic's rebounds (vs Kings)
predicted_jokic_rebounds = predict_player_stat("Nikola Jokić", "REB")
print(f"Predicted Rebounds for Nikola Jokić vs. Kings: {predicted_jokic_rebounds:.2f}")

# Predict Ivica Zubac's rebounds (vs Pistons)
predicted_zubac_rebounds = predict_player_stat("Ivica Zubac", "REB")
print(f"Predicted Rebounds for Ivica Zubac vs. Pistons: {predicted_zubac_rebounds:.2f}")

# Predict Giannis Antetokounmpo's points (vs Mavericks)
predicted_giannis_points = predict_player_stat("Giannis Antetokounmpo", "PTS")
print(f"Predicted Points for Giannis Antetokounmpo vs. Mavericks: {predicted_giannis_points:.2f}")

# Predict Shai Gilgeous-Alexander's points (vs Grizzlies)
predicted_sga_points = predict_player_stat("Shai Gilgeous-Alexander", "PTS")
print(f"Predicted Points for Shai Gilgeous-Alexander vs. Grizzlies: {predicted_sga_points:.2f}")