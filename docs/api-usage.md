# EleFind API Usage

EleFind exposes a detection endpoint through Gradio's built-in API. The Space is publicly accessible — no authentication is required.

**Base URL:** `https://iamhelitha-elefind-gradio-ui.hf.space`

---

## Endpoint

### `POST /call/detect`

Submits an image for elephant detection. Returns an `event_id` used to retrieve results.

**Request**

```http
POST /call/detect
Content-Type: application/json

{
  "data": [
    { "path": "https://example.com/aerial-image.jpg" },
    0.30,
    1024,
    0.30,
    0.40
  ]
}
```

**Parameters (positional, in `data` array)**

| Index | Name | Type | Default | Range | Description |
|-------|------|------|---------|-------|-------------|
| 0 | `image` | URL or file path | required | — | Aerial image to analyse |
| 1 | `conf_threshold` | float | `0.30` | 0.05 – 0.95 | Minimum detection confidence |
| 2 | `slice_size` | int | `1024` | 256 – 2048 | SAHI tile size in pixels |
| 3 | `overlap_ratio` | float | `0.30` | 0.05 – 0.50 | Tile overlap fraction |
| 4 | `iou_threshold` | float | `0.40` | 0.10 – 0.80 | NMS IoU threshold |

**Response**

```json
{ "event_id": "abc123xyz" }
```

---

### `GET /call/detect/{event_id}`

Streams results for a submitted job using Server-Sent Events (SSE).

```http
GET /call/detect/abc123xyz
```

The stream emits events until a `complete` event is received:

```
event: generating
data: null

event: complete
data: [<detection_image>, <count>, <avg_confidence>, <max_confidence>, <min_confidence>, <params_text>, <conf_chart>, <det_table>]
```

**Output fields (positional)**

| Index | Field | Type | Description |
|-------|-------|------|-------------|
| 0 | `detection_image` | object | Annotated image `{ path, url, size, orig_name }` |
| 1 | `count` | int | Number of elephants detected |
| 2 | `avg_confidence` | float | Average detection confidence (0.0 – 1.0) |
| 3 | `max_confidence` | float | Highest single detection confidence |
| 4 | `min_confidence` | float | Lowest single detection confidence |
| 5 | `params_text` | string | Markdown summary of inference parameters |
| 6 | `conf_chart` | object / null | Per-elephant confidence data (pandas DataFrame as JSON) |
| 7 | `det_table` | object / null | Full detection table with bounding boxes |

---

## Using the JavaScript client (recommended for React / Next.js)

Install the official Gradio client:

```bash
npm install @gradio/client
```

### Basic usage

```javascript
import { Client, handle_file } from "@gradio/client";

async function detectElephants(imageFile, options = {}) {
  const client = await Client.connect("iamhelitha/EleFind-gradio-ui");

  const result = await client.predict("/detect", {
    image:          handle_file(imageFile),       // File or Blob object
    conf_threshold: options.conf    ?? 0.30,
    slice_size:     options.slice   ?? 1024,
    overlap_ratio:  options.overlap ?? 0.30,
    iou_threshold:  options.iou     ?? 0.40,
  });

  const [
    detectionImage,
    count,
    avgConfidence,
    maxConfidence,
    minConfidence,
    paramsText,
    confChart,
    detTable,
  ] = result.data;

  return { detectionImage, count, avgConfidence, maxConfidence, minConfidence };
}
```

### React component example

```jsx
import { useState } from "react";
import { Client, handle_file } from "@gradio/client";

export default function ElephantDetector() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    const file = e.target.image.files[0];
    if (!file) return;

    setLoading(true);
    try {
      const client = await Client.connect("iamhelitha/EleFind-gradio-ui");
      const { data } = await client.predict("/detect", {
        image:          handle_file(file),
        conf_threshold: 0.30,
        slice_size:     1024,
        overlap_ratio:  0.30,
        iou_threshold:  0.40,
      });

      setResult({
        imageUrl:  data[0].url,
        count:     data[1],
        avgConf:   data[2],
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input type="file" name="image" accept="image/*" />
      <button type="submit" disabled={loading}>
        {loading ? "Detecting..." : "Detect Elephants"}
      </button>
      {result && (
        <div>
          <p>Elephants found: {result.count}</p>
          <p>Avg confidence: {(result.avgConf * 100).toFixed(1)}%</p>
          <img src={result.imageUrl} alt="Detection result" />
        </div>
      )}
    </form>
  );
}
```

---

## Using curl (testing / server-side)

### Step 1 — Submit the job

```bash
curl -X POST https://iamhelitha-elefind-gradio-ui.hf.space/call/detect \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      { "path": "https://example.com/aerial-image.jpg" },
      0.30,
      1024,
      0.30,
      0.40
    ]
  }'
```

Response:
```json
{ "event_id": "abc123xyz" }
```

### Step 2 — Stream the result

```bash
curl -N https://iamhelitha-elefind-gradio-ui.hf.space/call/detect/abc123xyz
```

---

## Using the Python client

```bash
pip install gradio_client
```

```python
from gradio_client import Client, handle_file

client = Client("iamhelitha/EleFind-gradio-ui")

result = client.predict(
    image=handle_file("/path/to/aerial-image.jpg"),
    conf_threshold=0.30,
    slice_size=1024,
    overlap_ratio=0.30,
    iou_threshold=0.40,
    api_name="/detect",
)

detection_image, count, avg_conf, max_conf, min_conf, params, chart, table = result
print(f"Elephants detected: {count}")
print(f"Average confidence: {avg_conf:.1%}")
```

---

## Notes

- **CORS:** Gradio Spaces allow requests from any origin, including browser-side JavaScript on Vercel or other external domains.
- **Rate limits:** The Space runs on CPU (free tier). Inference on a large image can take 30–120 seconds. Set appropriate timeouts in your client.
- **Concurrency:** The Space processes one request at a time (`concurrency_limit=1`, queue `max_size=10`). Requests beyond the queue limit will be rejected.
- **Image size:** Images larger than 6000 px on the longest edge are automatically downscaled before inference.
- **Interactive docs:** Visit `https://iamhelitha-elefind-gradio-ui.hf.space` and click "Use via API" in the footer to see the live API reference.
