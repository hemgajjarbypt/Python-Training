import sys
import subprocess

print("Hello, World!")

print("Python Version:", sys.version)

result = subprocess.run(["pip", "list"], capture_output=True, text=True)
print(result.stdout)