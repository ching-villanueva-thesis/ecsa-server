## Cuckoo Search Enhancement Thesis

### Set up

1. Create a virtual environment `python -m venv venv`
2. Activate the virtual environment `source venv/Scripts/activate`
3. Install dependencies `pip install -r requirements.txt`

### Project Structure Introduction

```
functions/
├── test.py
└── ...
src/
├── cs/
│   ├── __init__.py
│   ├── __main__.py
│   └── algo.py
└── ecs/
    ├── sop/
    │   ├── sop1.py
    │   ├── sop2.py
    │   └── sop3.py
    ├── __init__.py
    ├── __main__.py
    └── algo.py
.gitignore
api.py
README.md
requirements.txt

```

### Algorithm Testing

To test the Cuckoo Search and Enhanced Cuckoo Search algorithms, simply run the following python commands:

```bash
python -m src.cs
# or
python -m src.ecs

```
