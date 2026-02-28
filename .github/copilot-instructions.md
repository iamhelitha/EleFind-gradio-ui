# EleFind Gradio UI - Deployment Instructions

## Project Links

- **GitHub**: https://github.com/iamhelitha/EleFind-gradio-ui
- **HuggingFace Space**: https://huggingface.co/spaces/iamhelitha/EleFind-gradio-ui
- **HuggingFace Model**: https://huggingface.co/iamhelitha/EleFind-yolo11-elephant

## Automatic Deployment (GitHub Actions)

Pushing to `main` automatically deploys to HuggingFace Spaces via the workflow at `.github/workflows/deploy-hf.yml`.

The workflow:
1. Checks out the repo
2. Installs `huggingface-hub`
3. Runs `upload_folder()` to push files to the HF Space (excluding dev-only folders)
4. Polls the build status for up to 8 minutes and reports RUNNING / BUILD_ERROR

### Required GitHub Secret

You must add a HuggingFace write token as a GitHub Secret before the workflow can run:

| Secret name | Where to get it |
|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens → New token → Role: **Write** |

Add it at: **GitHub → Settings → Secrets and variables → Actions → New repository secret**

---

## Manual Deployment (Fallback)

If you need to deploy outside of a GitHub push (e.g., for a hotfix or out-of-band update), use the `huggingface_hub` Python library directly.

> **Note:** `git push hf main` does NOT work — HuggingFace requires Xet storage for binary files (JPEGs in `examples/`). Always use `upload_folder()` instead.

### Upload Command

```python
from huggingface_hub import upload_folder

result = upload_folder(
    folder_path='.',
    repo_id='iamhelitha/EleFind-gradio-ui',
    repo_type='space',
    ignore_patterns=[
        '.git/*', '.git',
        'meeting_materials/*', 'meeting_materials',
        '.DS_Store',
        '__pycache__/*', '*.pyc',
        '.claude/*', '.claude',
        '.github/*', '.github',
    ],
    commit_message='Your commit message here',
)
print('Upload complete!')
print(result)
```

Run this from the project root directory.

### Prerequisites

- `huggingface_hub` must be installed (`pip install huggingface-hub`)
- You must be logged in via `huggingface-cli login` or have a token set

### Checking Build Status

```python
from huggingface_hub import HfApi

api = HfApi()
info = api.space_info('iamhelitha/EleFind-gradio-ui')
rt = info.runtime
print('Stage:', rt.stage)  # BUILDING, RUNNING, BUILD_ERROR, etc.
if rt.raw.get('errorMessage'):
    print('Error:', rt.raw['errorMessage'][:500])
```

---

## Gradio Version Handling

- **HF Spaces**: Gradio version is controlled by `sdk_version` in `README.md` frontmatter (currently `6.8.0`). Do NOT pin gradio in `requirements.txt` — it will conflict.
- **Local dev**: Install gradio separately (`pip install gradio>=4.44`). The local version (4.44.1) differs from HF Spaces (6.8.0) due to Python 3.9 compatibility.

## Typical Deploy Workflow

1. Make changes locally and test on `http://127.0.0.1:7860`
2. Commit and push to GitHub: `git add . && git commit -m "message" && git push origin main`
3. GitHub Actions automatically deploys to HF Space — check progress at: https://github.com/iamhelitha/EleFind-gradio-ui/actions
4. If `BUILD_ERROR`, check build logs:

```python
from huggingface_hub import HfFolder
import urllib.request

token = HfFolder.get_token()
url = 'https://huggingface.co/api/spaces/iamhelitha/EleFind-gradio-ui/logs/build'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
resp = urllib.request.urlopen(req)
print(resp.read().decode('utf-8')[-3000:])
```
