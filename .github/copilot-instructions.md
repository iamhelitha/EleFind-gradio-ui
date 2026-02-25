# EleFind Gradio UI - Deployment Instructions

## Project Links

- **GitHub**: https://github.com/iamhelitha/EleFind-gradio-ui
- **HuggingFace Space**: https://huggingface.co/spaces/iamhelitha/EleFind-gradio-ui
- **HuggingFace Model**: https://huggingface.co/iamhelitha/EleFind-yolo11-elephant

## Pushing to HuggingFace Space

The HF Space is **not** synced with GitHub automatically. You must upload manually using the `huggingface_hub` Python library.

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

## Gradio Version Handling

- **HF Spaces**: Gradio version is controlled by `sdk_version` in `README.md` frontmatter (currently `5.50.0`). Do NOT pin gradio in `requirements.txt` — it will conflict.
- **Local dev**: Install gradio separately (`pip install gradio>=4.44`). The local version (4.44.1) differs from HF Spaces (5.50.0) due to Python 3.9 compatibility.

## Typical Deploy Workflow

1. Make changes locally and test on `http://127.0.0.1:7860`
2. Commit and push to GitHub: `git add . && git commit -m "message" && git push origin main`
3. Upload to HF Space using the `upload_folder` command above
4. Check build status — wait for `RUNNING` stage
5. If `BUILD_ERROR`, check logs:

```python
from huggingface_hub import HfFolder
import urllib.request

token = HfFolder.get_token()
url = 'https://huggingface.co/api/spaces/iamhelitha/EleFind-gradio-ui/logs/build'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
resp = urllib.request.urlopen(req)
print(resp.read().decode('utf-8')[-3000:])
```
