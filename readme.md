# 🚪🪟 Simple Structure Segmentation API

FastAPI-based image segmentation API using YOLOv8 for detecting **windows** and **doors**, with model artifacts hosted on Hugging Face Hub.

## Features

- Upload image and get polygon-based segmentation results
- Overlay masks on original image
- Upload processed result to Hugging Face Hub
- Rate limited (10 requests per minute)
- API key authentication via `X-API-KEY` header

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
