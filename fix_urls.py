#!/usr/bin/env python3
import re

# Fix the URLs in .env file
with open('/Users/ibrooks/Documents/GitHub/Cloudera-Inference-With-Windsurf/.env', 'r') as f:
    content = f.read()

# Replace https:// with https://
content = re.sub(r'https:\/\//', 'https://', content)

with open('/Users/ibrooks/Documents/GitHub/Cloudera-Inference-With-Windsurf/.env', 'w') as f:
    f.write(content)

print("Fixed URLs in .env file - removed extra colon in https://")
