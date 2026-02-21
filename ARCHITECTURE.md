# Aerivon Live Architecture

This diagram shows the full system architecture of Aerivon Live.

![Aerivon Live Architecture](docs/images/aerivon-live-architecture.png)

## Overview

Aerivon Live is an autonomous Gemini-powered AI agent deployed on Google Cloud Run.

It integrates:

- Vertex AI Gemini models
- Autonomous agent runtime
- Tool execution firewall
- Secure browsing via Playwright
- Lead extraction and outreach automation

The system automatically falls back to Gemini Flash if Gemini Live is unavailable.
