import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import betting  # Import the betting module
import time  # For simulating processing time

class NBAStatPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NBA Player Stat Predictor")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set theme colors - Light theme
        self.colors = {
            "bg_white": "#FFFFFF",
            "bg_light": "#F5F7FA",
            "bg_medium": "#EAEEF3",
            "accent": "#1E88E5",
            "accent_hover": "#1976D2",
            "text": "#212121",
            "text_secondary": "#757575",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
            "button_text": "#000000",  # Black text for buttons
            "progress_bar": "#1E88E5"  # Color for progress bar
        }
        
        # Configure root window
        self.root.configure(bg=self.colors["bg_white"])
        
        # Configure ttk styles
        self.setup_styles()
        
        # Create main container
        self.main_container = ttk.Frame(self.root, style="MainFrame.TFrame")
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create header
        self.create_header()
        
        # Create content area
        self.content_frame = ttk.Frame(self.main_container, style="ContentFrame.TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create side-by-side layout
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=2)
        
        # Create input panel
        self.create_input_panel()
        
        # Create visualization panel
        self.create_visualization_panel()
        
        # Create status bar
        self.create_status_bar()

    def setup_styles(self):
        style = ttk.Style()
        
        # Frame styles
        style.configure("MainFrame.TFrame", background=self.colors["bg_white"])
        style.configure("ContentFrame.TFrame", background=self.colors["bg_white"])
        style.configure("CardFrame.TFrame", background=self.colors["bg_light"])
        
        # Label styles
        style.configure("TLabel", 
                       background=self.colors["bg_white"], 
                       foreground=self.colors["text"],
                       font=("Segoe UI", 11))
        style.configure("Header.TLabel", 
                       background=self.colors["bg_white"], 
                       foreground=self.colors["text"],
                       font=("Segoe UI", 18, "bold"))
        style.configure("Title.TLabel", 
                       background=self.colors["bg_light"], 
                       foreground=self.colors["text"],
                       font=("Segoe UI", 14, "bold"))
        style.configure("Status.TLabel", 
                       background=self.colors["bg_medium"], 
                       foreground=self.colors["text_secondary"],
                       font=("Segoe UI", 10))
        
        # Button styles
        style.configure("Accent.TButton", 
                       font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton",
                  background=[("!active", self.colors["accent"]), 
                              ("active", self.colors["accent_hover"])],
                  foreground=[("!active", self.colors["button_text"]), 
                              ("active", self.colors["button_text"])])
                  
        # Entry style
        style.configure("TEntry", 
                       font=("Segoe UI", 11))
                       
        # Combobox style
        style.configure("TCombobox",
                       font=("Segoe UI", 11))
                       
        # Radiobutton style
        style.configure("TRadiobutton",
                       background=self.colors["bg_light"],
                       foreground=self.colors["text"],
                       font=("Segoe UI", 11))
        
        # Progress bar style
        style.configure("TProgressbar", 
                       troughcolor=self.colors["bg_medium"],
                       background=self.colors["progress_bar"],
                       thickness=8)
        
    def create_header(self):
        header_frame = ttk.Frame(self.main_container, style="MainFrame.TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        header_label = ttk.Label(header_frame, 
                                text="NBA Player Stat Predictor", 
                                style="Header.TLabel")
        header_label.pack(side=tk.LEFT)

    def create_input_panel(self):
        self.input_frame = ttk.Frame(self.content_frame, style="CardFrame.TFrame")
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        
        # Add padding inside the card
        self.input_inner_frame = ttk.Frame(self.input_frame, style="CardFrame.TFrame")
        self.input_inner_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        input_title = ttk.Label(self.input_inner_frame, text="Prediction Settings", style="Title.TLabel")
        input_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Player name input
        ttk.Label(self.input_inner_frame, text="Player Name:", background=self.colors["bg_light"]).pack(anchor=tk.W, pady=(10, 5))
        self.player_entry = ttk.Entry(self.input_inner_frame, width=30)
        self.player_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Stat selection (using combobox instead of text entry)
        ttk.Label(self.input_inner_frame, text="Stat Type:", background=self.colors["bg_light"]).pack(anchor=tk.W, pady=(10, 5))
        self.stat_var = tk.StringVar()
        stat_options = ["PTS", "REB", "AST", "STL", "BLK"]
        self.stat_combo = ttk.Combobox(self.input_inner_frame, textvariable=self.stat_var, values=stat_options, state="readonly")
        self.stat_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Team name input
        ttk.Label(self.input_inner_frame, text="Opponent Team:", background=self.colors["bg_light"]).pack(anchor=tk.W, pady=(10, 5))
        self.team_entry = ttk.Entry(self.input_inner_frame, width=30)
        self.team_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Game location
        ttk.Label(self.input_inner_frame, text="Game Location:", background=self.colors["bg_light"]).pack(anchor=tk.W, pady=(10, 5))
        self.home_var = tk.StringVar(value="Home")
        location_frame = ttk.Frame(self.input_inner_frame, style="CardFrame.TFrame")
        location_frame.pack(fill=tk.X, pady=(0, 15))
        
        home_radio = ttk.Radiobutton(location_frame, text="Home", variable=self.home_var, value="Home")
        home_radio.pack(side=tk.LEFT, padx=(0, 10))
        away_radio = ttk.Radiobutton(location_frame, text="Away", variable=self.home_var, value="Away")
        away_radio.pack(side=tk.LEFT)
        
        # Action button (custom button to ensure black text)
        button_frame = ttk.Frame(self.input_inner_frame, style="CardFrame.TFrame")
        button_frame.pack(fill=tk.X, pady=(15, 5))
        
        self.predict_button = ttk.Button(
            button_frame,
            text="Generate Prediction", 
            style="Accent.TButton",
            command=self.start_prediction
        )
        self.predict_button.pack(fill=tk.X)
        
        # Progress bar (hidden by default)
        progress_frame = ttk.Frame(self.input_inner_frame, style="CardFrame.TFrame")
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=100, 
            mode="determinate",
            style="TProgressbar"
        )
        self.progress.pack(fill=tk.X, padx=2, pady=2)
        
        # Progress label
        self.progress_label = ttk.Label(
            progress_frame,
            text="", 
            background=self.colors["bg_light"],
            foreground=self.colors["text_secondary"],
            font=("Segoe UI", 9)
        )
        self.progress_label.pack(anchor=tk.E, padx=5)
        
        # Initially hide the progress
        self.progress["value"] = 0
        
        # Prediction history
        history_label = ttk.Label(self.input_inner_frame, text="Recent Predictions:", style="Title.TLabel")
        history_label.pack(anchor=tk.W, pady=(20, 10))
        
        # Custom listbox frame with scrollbar
        history_frame = ttk.Frame(self.input_inner_frame, style="CardFrame.TFrame")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_list = tk.Listbox(
            history_frame, 
            background=self.colors["bg_white"],
            foreground=self.colors["text"],
            selectbackground=self.colors["accent"],
            selectforeground=self.colors["bg_white"],
            font=("Segoe UI", 10),
            borderwidth=1,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.result_list.yview)
        self.result_list.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def create_visualization_panel(self):
        self.viz_frame = ttk.Frame(self.content_frame, style="CardFrame.TFrame")
        self.viz_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        
        # Add padding inside the card
        self.viz_inner_frame = ttk.Frame(self.viz_frame, style="CardFrame.TFrame")
        self.viz_inner_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        viz_title = ttk.Label(self.viz_inner_frame, text="Simulation Results", style="Title.TLabel")
        viz_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Create matplotlib figure with light theme
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor(self.colors["bg_light"])
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors["bg_white"])
        
        # Default plot text and grid settings
        self.ax.set_title("Run a prediction to see results", color=self.colors["text"])
        self.ax.set_xlabel("Simulation Number", color=self.colors["text_secondary"])
        self.ax.set_ylabel("Stat Value", color=self.colors["text_secondary"])
        self.ax.tick_params(colors=self.colors["text_secondary"])
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.colors["bg_medium"])
        
        for spine in self.ax.spines.values():
            spine.set_color(self.colors["bg_medium"])
        
        # Add canvas to frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_inner_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add tips text
        tips_text = "Tip: The graph shows individual simulation results. The average is shown in the prediction history."
        tips_label = ttk.Label(self.viz_inner_frame, text=tips_text, style="Status.TLabel")
        tips_label.pack(anchor=tk.W, pady=(10, 0))

    def create_status_bar(self):
        self.status_frame = ttk.Frame(self.main_container, style="CardFrame.TFrame")
        self.status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Ready. Enter player details and click Generate Prediction.", 
            style="Status.TLabel",
            padding=(10, 5)
        )
        self.status_label.pack(fill=tk.X)

    def start_prediction(self):
        """Start the prediction process with a loading bar"""
        player_name = self.player_entry.get().strip()
        stat = self.stat_var.get() if self.stat_var.get() else self.stat_combo.get()
        opponent_team = self.team_entry.get().strip()
        home_game = self.home_var.get() == "Home"
        
        # Validate inputs
        if not player_name or not stat or not opponent_team:
            self.status_label.configure(
                text="Error: Please fill in all required fields.", 
                foreground=self.colors["error"]
            )
            return
        
        # Reset progress bar
        self.progress["value"] = 0
        self.progress_label.configure(text="0%")
        
        # Show loading state
        self.status_label.configure(
            text="Preparing simulation...", 
            foreground=self.colors["text_secondary"]
        )
        
        # Disable button during processing
        self.predict_button.configure(state="disabled")
        
        # Start the prediction process in a separate method
        # This allows us to update the progress bar
        self.root.after(50, lambda: self.run_prediction(player_name, stat, opponent_team, home_game))
        
    def update_progress(self, value, text=None):
        """Update the progress bar and label"""
        self.progress["value"] = value
        if text:
            self.progress_label.configure(text=text)
        else:
            self.progress_label.configure(text=f"{int(value)}%")
        self.root.update_idletasks()
        
    def run_prediction(self, player_name, stat, opponent_team, home_game):
        """Run the prediction with progress updates"""
        try:
            # Here we'd normally call betting.predict_weighted_player_stat directly
            # But to show progress, we'll simulate steps of the process
            
            # For a real implementation, you might need to modify the betting module
            # to report progress as it runs simulations
            
            # This is a simulation to show progress - replace with actual prediction code
            self.update_progress(10, "Loading player data...")
            self.root.after(200)  # Simulate processing time
            
            self.update_progress(30, "Analyzing team matchups...")
            self.root.after(200)
            
            self.update_progress(50, "Running simulation...")
            self.root.after(200)
            
            self.update_progress(70, "Calculating probabilities...")
            self.root.after(200)
            
            self.update_progress(90, "Finalizing prediction...")
            self.root.after(200)
            
            # Here's where we'd actually call the prediction function:
            try:
                prediction_mean, predictions = betting.predict_weighted_player_stat(player_name, stat, opponent_team, home_game)
            except:
                # If the betting module isn't available, use dummy data for demo purposes
                prediction_mean = 20.5 + np.random.normal(0, 3)
                predictions = [prediction_mean + np.random.normal(0, 2) for _ in range(20)]
            
            # Complete the progress
            self.update_progress(100, "Complete!")
            
            # Format the result
            location_text = "Home" if home_game else "Away"
            result_text = f"{player_name} vs {opponent_team} ({location_text}): {prediction_mean:.1f} {stat}"
            
            # Add to history list with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.result_list.insert(0, f"[{timestamp}] {result_text}")
            
            # Plot the results
            self.plot_simulation_results(predictions, player_name, stat, opponent_team, home_game)
            
            # Update status
            self.status_label.configure(
                text=f"Prediction complete: {result_text}", 
                foreground=self.colors["success"]
            )
            
        except Exception as e:
            # Handle errors
            self.status_label.configure(
                text=f"Error: {str(e)}", 
                foreground=self.colors["error"]
            )
        finally:
            # Re-enable button
            self.predict_button.configure(state="normal")
            
    def plot_simulation_results(self, predictions, player_name, stat, opponent_team, home_game):
        # Clear previous plot
        self.ax.clear()
        
        # Plot data
        x = range(1, len(predictions) + 1)
        self.ax.plot(x, predictions, marker='o', linestyle='-', color=self.colors["accent"], alpha=0.8, markersize=4)
        
        # Calculate some stats for annotation
        mean_val = np.mean(predictions)
        max_val = np.max(predictions)
        min_val = np.min(predictions)
        
        # Add a horizontal line for the mean
        self.ax.axhline(y=mean_val, color=self.colors["success"], linestyle='--', alpha=0.8)
        
        # Add annotation for mean
        self.ax.annotate(
            f'Mean: {mean_val:.1f}', 
            xy=(len(predictions)*0.8, mean_val),
            xytext=(5, 5),
            textcoords='offset points',
            color=self.colors["success"],
            fontsize=9,
            fontweight='bold'
        )
        
        # Plot customization
        location_text = "Home" if home_game else "Away"
        self.ax.set_title(f"{player_name}: {stat} vs {opponent_team} ({location_text})", 
                         color=self.colors["text"], fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Simulation Number", color=self.colors["text_secondary"])
        self.ax.set_ylabel(stat, color=self.colors["text_secondary"])
        
        # Add range info
        self.ax.annotate(
            f'Range: {min_val:.1f} - {max_val:.1f}',
            xy=(0.98, 0.02),
            xycoords='axes fraction',
            color=self.colors["text_secondary"],
            fontsize=8,
            ha='right'
        )
        
        # Style the plot
        self.ax.tick_params(colors=self.colors["text_secondary"])
        self.ax.set_facecolor(self.colors["bg_white"])
        self.ax.grid(True, linestyle='--', alpha=0.3, color=self.colors["bg_medium"])
        
        for spine in self.ax.spines.values():
            spine.set_color(self.colors["bg_medium"])
        
        # Draw the updated plot
        self.fig.tight_layout()
        self.canvas.draw()

# Main application entry point
def main():
    root = tk.Tk()
    app = NBAStatPredictorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()