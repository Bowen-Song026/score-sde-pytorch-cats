import re
with open('run_full_experiment.py', 'r') as f:
    content = f.read()
content = content.replace("'batch_size': 32,", "'batch_size': 8,")
with open('run_full_experiment.py', 'w') as f:
    f.write(content)
print('Done')
