#!/usr/bin/env python3
"""Retest GPT-5.4, Kimi-K2.6, and Gemini-3.6 Flash using the recorded study settings.

Inference settings follow the reproducibility appendix (Table S1):

  GPT-5.4            Azure OpenAI-compatible Chat Completions
                     temperature 0.2, 4096 completion tokens, JSON object
  Kimi-K2.6          Azure OpenAI-compatible Chat Completions
                     temperature 0.6, 4096 completion tokens, prompt-enforced JSON
  Gemini-3.6 Flash   Vertex AI generateContent REST
                     temperature 0.2, 800 output tokens, application/json

Each call uses the same English prompt and returns JSON with
primary_diagnosis, secondary_diagnosis, and basis. Up to 4 images. Three
attempts (initial plus 2 retries), waiting 5, 10 seconds between failures.

Environment:

  OPENAI_API_KEY / OPENAI_BASE_URL     GPT-5.4 Azure deployment
  OPENAI_MODEL                         default gpt-5.4
  KIMI_API_KEY / KIMI_BASE_URL         Kimi-K2.6 Azure deployment
                                       fall back to OPENAI_API_KEY / OPENAI_BASE_URL
  KIMI_MODEL                           default kimi-k2.6
  VERTEX_PROJECT / VERTEX_LOCATION     Gemini on Vertex AI
  VERTEX_ACCESS_TOKEN                  gcloud auth print-access-token
  GEMINI_MODEL                         default gemini-3.6-flash
  GEMINI_API_KEY                       optional Google AI Studio fallback
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "results" / "api_suggestions.csv"

PROMPT = (
    "You are an expert radiologist. Use both the clinical history and uploaded "
    "radiographs; do not rely on either source alone. Return JSON only with "
    "the keys primary_diagnosis, secondary_diagnosis, and basis. Use concise "
    "diagnostic terms; use 'None' when there is no secondary diagnosis. In no "
    "more than 120 words, explain the key imaging findings and how they relate "
    "to the clinical history. "
)

EXAM = {
    "chest": "Chest radiograph - frontal view",
    "abdomen": "Abdominal radiograph - frontal view",
}


def png_bytes(path: Path) -> bytes:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".dicom"} or _looks_dicom(path):
        import pydicom

        ds = pydicom.dcmread(str(path), force=True)
        pixels = ds.pixel_array
        if pixels.ndim == 3:
            pixels = pixels[..., :3] if pixels.shape[-1] in (3, 4) else pixels[0]
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            pixels = pixels.max() - pixels
        x = pixels.astype(np.float32)
        x -= x.min()
        if x.max() > 0:
            x = x / x.max() * 255
        image = Image.fromarray(x.astype(np.uint8)).convert("RGB")
    else:
        image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _looks_dicom(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(png_bytes(path)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_prompt(history: str, examination: str) -> str:
    return (
        PROMPT
        + f"Clinical history: {history or 'not provided'}. Examination: {examination}."
    )


def parse_result(payload) -> dict:
    if isinstance(payload, str):
        match = re.search(r"\{.*\}", payload, flags=re.S)
        payload = json.loads(match.group(0)) if match else {"primary_diagnosis": payload}
    primary = str(payload.get("primary_diagnosis") or "").strip()
    secondary = str(payload.get("secondary_diagnosis") or "").strip()
    basis = str(payload.get("basis") or payload.get("rationale") or "").strip()
    if secondary.lower() == "none":
        secondary = ""
    return {
        "primary_diagnosis_zh": primary,
        "secondary_diagnosis_zh": secondary,
        "rationale_zh": basis,
        "primary_diagnosis": primary,
        "secondary_diagnosis": secondary,
        "basis": basis,
    }


def with_retries(fn, attempts: int = 3):
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == attempts - 1:
                raise RuntimeError("Inference failed after 3 attempts") from exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Inference failed after 3 attempts") from last


def infer_openai(
    client: OpenAI,
    model: str,
    image_paths: list[Path],
    history: str,
    examination: str,
    temperature: float,
    json_object: bool,
    max_tokens: int,
) -> dict:
    content = [{"type": "text", "text": build_prompt(history, examination)}]
    content += [{"type": "image_url", "image_url": {"url": image_data_url(p)}} for p in image_paths[:4]]
    request = dict(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )
    if json_object:
        request["response_format"] = {"type": "json_object"}

    def _call() -> dict:
        try:
            response = client.chat.completions.create(**request)
        except TypeError:
            request.pop("max_completion_tokens", None)
            request["max_tokens"] = max_tokens
            response = client.chat.completions.create(**request)
        return parse_result(response.choices[0].message.content)

    return with_retries(_call)


def infer_gemini(image_paths: list[Path], history: str, examination: str) -> dict:
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    parts = [{"text": build_prompt(history, examination)}]
    for path in image_paths[:4]:
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": base64.b64encode(png_bytes(path)).decode("ascii"),
                }
            }
        )
    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json",
        },
    }
    project = os.environ.get("VERTEX_PROJECT", "").strip()
    location = os.environ.get("VERTEX_LOCATION", "us-central1").strip()
    token = os.environ.get("VERTEX_ACCESS_TOKEN", "").strip()
    studio_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if project and token:
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    elif studio_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={studio_key}"
        headers = {"Content-Type": "application/json"}
    else:
        raise RuntimeError("Set VERTEX_PROJECT and VERTEX_ACCESS_TOKEN, or GEMINI_API_KEY")

    def _call() -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        text = payload["candidates"][0]["content"]["parts"][0].get("text", "")
        return parse_result(text)

    return with_retries(_call)


def azure_client(api_key_var: str, base_url_var: str, fallback_key_var: str = "", fallback_url_var: str = "") -> OpenAI:
    api_key = os.environ.get(api_key_var) or (os.environ.get(fallback_key_var) if fallback_key_var else "")
    base_url = os.environ.get(base_url_var) or (os.environ.get(fallback_url_var) if fallback_url_var else "")
    if not api_key or not base_url:
        raise RuntimeError(f"Set {api_key_var} and {base_url_var} for the Azure OpenAI deployment")
    return OpenAI(api_key=api_key, base_url=base_url)


def infer_named(name: str, image_paths: list[Path], history: str, examination: str) -> dict:
    if name == "GPT-5.4":
        client = azure_client("OPENAI_API_KEY", "OPENAI_BASE_URL")
        model = os.environ.get("OPENAI_MODEL", "gpt-5.4")
        return infer_openai(client, model, image_paths, history, examination, 0.2, True, 4096)
    if name == "Kimi-K2.6":
        client = azure_client("KIMI_API_KEY", "KIMI_BASE_URL", "OPENAI_API_KEY", "OPENAI_BASE_URL")
        model = os.environ.get("KIMI_MODEL", "kimi-k2.6")
        return infer_openai(client, model, image_paths, history, examination, 0.6, False, 4096)
    if name == "Gemini-3.6 Flash":
        return infer_gemini(image_paths, history, examination)
    raise ValueError(name)


def available_models() -> list[str]:
    ready = []
    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL"):
        ready.append("GPT-5.4")
    if (os.environ.get("KIMI_API_KEY") and os.environ.get("KIMI_BASE_URL")) or (
        os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_BASE_URL")
    ):
        ready.append("Kimi-K2.6")
    if os.environ.get("VERTEX_PROJECT") or os.environ.get("GEMINI_API_KEY"):
        ready.append("Gemini-3.6 Flash")
    return ready


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    existing = pd.read_csv(path)
    if existing.empty:
        return set()
    return set(zip(existing["case_id"].astype(str), existing["model"].astype(str)))


def run_one(args: argparse.Namespace) -> None:
    result = infer_named(
        args.model,
        [Path(p) for p in args.image],
        args.history,
        args.examination,
    )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_batch(args: argparse.Namespace) -> None:
    cases = pd.read_csv(DATA / "cases.csv")
    if args.cases:
        cases = cases[cases["case_id"].isin(args.cases)]
    names = args.models or available_models()
    if not names:
        raise SystemExit("No API credentials found.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    rows = pd.read_csv(args.out).to_dict("records") if args.out.exists() else []

    for name in names:
        for _, case in cases.iterrows():
            pair = (str(case["case_id"]), name)
            if pair in done:
                continue
            image = DATA / case["image_file"]
            history = str(case.get("clinical_history_en") or case.get("clinical_history_zh") or "")
            examination = EXAM.get(str(case["body_region"]), str(case["body_region"]))
            try:
                parsed = infer_named(name, [image], history, examination)
                error = ""
            except Exception as exc:  # noqa: BLE001
                parsed = parse_result({})
                parsed["rationale_zh"] = f"ERROR: {exc}"
                parsed["basis"] = parsed["rationale_zh"]
                error = str(exc)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "model": name,
                    "clinical_history_zh": case.get("clinical_history_zh", ""),
                    **parsed,
                }
            )
            pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
            print(f"{case['case_id']}  {name}  {'ok' if not error else 'fail:' + error}")
    print(f"Wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", help="Image or DICOM path; repeat up to 4 times")
    parser.add_argument("--history", help="Clinical history for a single-case call")
    parser.add_argument("--examination", help="Examination label for a single-case call")
    parser.add_argument("--model", default="GPT-5.4", help="Model name for a single-case call")
    parser.add_argument("--output", default="gpt_case_result.json")
    parser.add_argument("--models", nargs="*", help="Batch: model names to run")
    parser.add_argument("--cases", nargs="*", help="Batch: Case_01 … subset")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.image and args.history and args.examination:
        run_one(args)
        return
    run_batch(args)


if __name__ == "__main__":
    main()
