import matplotlib.pyplot as plt
import re
import os

# Log data as a string
log_data = ""
val_log_data = ""

with open("./logs/step_log.txt", "r") as f:
    log_data = f.read()

with open("./logs/loss_log.txt", "r") as f:
    val_log_data = f.read()
    
# Extract step and loss values using regular expressions
steps = []
losses = []

val_steps = []
val_losses = []

for line in log_data.strip().split("\n"):
    step_match = re.search(r"Step\s+(\d+)", line)
    loss_match = re.search(r"loss:\s+([\d\.]+)", line)
    if step_match and loss_match:
        steps.append(int(step_match.group(1)))
        losses.append(float(loss_match.group(1)))
        
for line in val_log_data.strip().split("\n"):
    step_match = re.search(r"Eval step\s+(\d+)", line)
    loss_match = re.search(r"val loss\s+([\d\.]+)", line)
    if step_match and loss_match:
        val_steps.append(int(step_match.group(1)))
        val_losses.append(float(loss_match.group(1)))

# Plot the loss vs. steps
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, linestyle='-', color='b')
plt.plot(val_steps, val_losses, linestyle='-', color='g')
plt.title("Training Loss vs. Steps")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim(0, max(losses) * 1.1)
plt.grid(True)

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
            
output_path = os.path.join(output_dir, f'loss.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
