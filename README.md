# UZH: Advanced Topics in Artificial Intelligence (ATAI, Fall 2024)
Course assignment of building a chatbot that answers questions

## Team

- Melih Serin ([melih.serin@uzh.ch](melih.serin@uzh.ch))
- Claude Lehmann ([claudeolivier.lehmann@uzh.ch](claudeolivier.lehmann@uzh.ch))


## Setup

### Setup virtual environment and install dependencies

```
# Create python virtual environment
python3 -m venv env

# Activate environment (alternatively use the activate_env.sh bash script)
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download dataset

Copy the content of the provided dataset from OLAT to the `./data` directory.

### Set environment variables

In order to not have passwords in our github repository, please copy the `template.env` file and call the copy `.env`. Then fill in all the required variables in the `.env` file. We use `python-decouple` to access them in code like this:

```python
from decouple import Config

config = Config()

# Get UZH_SPEAKEASY_HOST value from .env
speakeasy_host = config('UZH_SPEAKEASY_HOST')

print(f"Speakeasy Host: {speakeasy_host}")
```

### Adding new libraries

When adding additional libraries, update the `requirements.txt` like this:

```
# Install the library you want
pip install <library>

# Add the installed libraries to requirements.txt
pip freeze > requirements.txt

# Commit and push the updated requirements.txt
git add requirements.txt
git commit -m "updated requirements.txt"
git push origin master
```

## Run Code

Assuming you have set up the environment already previously

```
# Activate environment (alternatively use the activate_env.sh bash script)
source env/bin/activate
```

