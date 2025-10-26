# The English Compiler

## A two part python compiler system that can run any text file as if it were valid programming code.

### Requirements
- Python 3.x or Python in a virtual environment
- Install the required packages using pip:
  ```bash
  pip install requests watchdog
  ```
- Ollama or other local LLM backend

### Usage
1. Clone the repository:
```bash
git clone https://github.com/DebelToni/English-Compiler
```
2. Run the initilization script with a file to compile:
```bash
python English.py File.txt
# Works with any file extension
```
3. Compile the file 
```bash
python Compile.py File.txt
```

### Tips
- For speed choose smaller LLMs, for complex code choose larger LLMs.
- Alias the scripts so you can run them from anywhere:
```bash
alias english="python /path/to/English.py"
alias compile="python /path/to/Compile.py"
```

