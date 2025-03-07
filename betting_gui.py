import tkinter as tk
from tkinter import ttk
import betting  # Import the betting module

# Function to fetch prediction from betting.py
def get_prediction():
    player_name = player_entry.get()
    stat = stat_entry.get()
    opponent_team = team_entry.get()
    
    if not player_name or not stat or not opponent_team:
        result_label.config(text="Please fill in all fields.", fg="red")
        return
    
    try:
        prediction = betting.predict_weighted_player_stat(player_name, stat, opponent_team)
        result_list.insert(0, f"{player_name} ({stat}) vs. {opponent_team}: {prediction:.2f}")  # Insert at top
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="red")

# Function to initialize the GUI
def main():
    global root, player_entry, stat_entry, team_entry, result_list
    root = tk.Tk()
    root.title("NBA Player Stat Predictor")
    root.geometry("450x400")
    root.configure(bg="#2C3E50")  # Dark background

    # Styling
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=5)
    style.configure("TLabel", font=("Arial", 12), background="#2C3E50", foreground="white")
    style.configure("TEntry", font=("Arial", 12))

    # Labels and Input Fields
    ttk.Label(root, text="Player Name:").pack(pady=5)
    player_entry = ttk.Entry(root)
    player_entry.pack(pady=5)

    ttk.Label(root, text="Stat (PTS, REB, AST, etc.):").pack(pady=5)
    stat_entry = ttk.Entry(root)
    stat_entry.pack(pady=5)

    ttk.Label(root, text="Opponent Team:").pack(pady=5)
    team_entry = ttk.Entry(root)
    team_entry.pack(pady=5)

    # Predict Button
    predict_button = ttk.Button(root, text="Get Prediction", command=get_prediction)
    predict_button.pack(pady=10)

    # Result Section
    ttk.Label(root, text="Prediction History:").pack(pady=5)
    result_list = tk.Listbox(root, height=10, width=50, font=("Arial", 10))
    result_list.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
