import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the JSON file
with open(r"P:\Group project files\gaze_actions_excluding_blinks.json", "r") as f:
    data = json.load(f)

# Convert to a DataFrame
df = pd.DataFrame(data)

# Encode each unique gaze action as a numeric ID for plotting
df['action_id'] = pd.factorize(df['action_type'])[0]
action_map = dict(enumerate(pd.factorize(df['action_type'])[1]))

# Plot the gaze actions over frames
plt.figure(figsize=(14, 6))
plt.plot(df['frame'], df['action_id'], linestyle='-', marker='o', markersize=2)

# Set y-ticks to show action labels
plt.yticks(ticks=list(action_map.keys()), labels=list(action_map.values()))
plt.xlabel("Frame")
plt.ylabel("Gaze Zone")
plt.title("Gaze Zone Transitions Over Time")
plt.grid(True)

# Save and show the plot
plt.tight_layout()
plt.savefig("gaze_zones_continuous_line.png")
plt.show()
