# Authenticating on HuggingFace

```
git config --global credential.helper store
huggingface-cli login
```

or

```
from huggingface_hub import HfApi, HfFolder

# Save the token for the session
token = os.getenv('HF_TOKEN')
HfFolder.save_token(token)

# Use the token for Hugging Face API interactions
api = HfApi()
api.whoami()

```