<div align="center">

# ⚽ FootballIQ

### AI-Powered Football Intelligence Platform

Transform football match footage into tactical intelligence, player analytics, and AI-powered coaching insights.

<img src="assets/footballiq-banner.png" alt="FootballIQ Banner" width="100%">

</div>

---

> **Real-time football intelligence platform powered by computer vision, deterministic analytics, and Fireworks AI.**

---

## 🌌 Overview

**FootballIQ** bridges the gap between raw physical movement on the pitch and high-level strategic reasoning. By ingesting video feeds, tracking positional coordinates, and running deterministic sports-science analytics, the platform generates a live game snapshot. This data is orchestrated and dispatched directly to elite open-weight LLMs hosted via Fireworks AI to deliver instant, actionable tactical insights straight to the technical bench.

---

## ⚡ Key Features

- **🤖 Multi-Agent Tactical Dispatcher** — Automatically routes match contexts into dedicated analytical streams (`CoachChat`, `Strategy`, `Opponent`, `Reports`).
- **💬 CoachChat Service** — A live, conversational interface allowing staff to query real-time match telemetry using natural language.
- **🎯 Automated Alert Engine** — Constantly monitors data thresholds to instantly push notifications regarding tactical shifts, positional errors, or player fatigue.
- **🛠️ Robust JSON Repair Pipeline** — Built-in fault tolerance ensures that unstructured LLM outputs are repaired on-the-fly to maintain UI uptime.
- **📊 Cost & Performance Audit** — Built-in simulation scripts to evaluate latency, token efficiency, and operational costs per match.

---

## 🏗️ Architecture

```
Vision ──> Tracking ──> Events ──> Analytics ──> Player Intelligence ──> Match Intelligence
                                                                                │
                                                                                ▼
                                                                      Live Match Session
                                                                                │
                                                                                ▼
                                                                     PipelineContext Snapshot
                                                                                │
                                                                                ▼
                                                                       Live AI Dispatcher
                                                                                │
                                                              ┌─────────────────┴─────────────────┐
                                                              ▼                                   ▼
                                                          CoachChat                        Tactical Alerts
                                                              │                                   │
                                                              └─────────────────┬─────────────────┘
                                                                                ▼
                                                                          AIOrchestrator
                                                                                │
                                                                                ▼
                                                                        FireworksProvider
                                                                                │
                                                                                ▼
                                                                         Fireworks AI API
```

---

## ⚙️ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/footballiq.git
cd footballiq
cp .env.example .env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Your Keys

Populate your `.env` file with your Fireworks credentials:

```env
FIREWORKS_API_KEY=your_secret_fireworks_api_key_here
FIREWORKS_MODEL=accounts/fireworks/models/gpt-oss-120b
```

---

## 🧠 Supported Intelligence Models

Hot-swap models by updating `.env` — no code changes needed.

| Model Reference | Primary Target Domain | Strength |
|---|---|---|
| `accounts/fireworks/models/gpt-oss-120b` | General Coaching & Complex Tactics | High contextual comprehension |
| `accounts/fireworks/models/deepseek-r1` | Reasoning-heavy Tasks & Math | Advanced logic & chain-of-thought |

> 💡 Override parameters directly within `config/ai.yaml` for fine-grained control.

---

## 🧪 Testing Suite

387+ automated checks to guarantee precision during live matches.

```bash
# Run the entire test suite
python -m pytest tests/ -q

# Test live integration endpoints
python -m pytest tests/test_live_ai.py -v

# Run tactical alert threshold engine tests
python -m pytest tests/test_tactical_alerts.py -v
```

---

## 📈 Production Verification & Cost Estimation

```bash
# Verify end-to-end API connectivity
python scripts/test_fireworks.py

# Simulate seasonal token costs
python scripts/cost_estimator.py --matches 10 --questions 20 --strategy 5 --reports 3
```

---

## 📁 Project Structure

```
footballiq/
├── assets/              # Banner and static media
├── config/              # AI provider & live threshold configurations
├── ai/                  # Core AIOrchestrator, router, and parsing engines
├── live/                # Match sessions, dispatchers, and alert rules
├── schema/              # Data schemas for PipelineContext definitions
├── scripts/             # Cost estimation and live testing utilities
└── tests/               # 387+ automated test validation scripts
```

---

## 🛡️ Troubleshooting

- **`FIREWORKS_API_KEY is not set`** — Ensure `.env` exists in the root directory and `python-dotenv` is initializing correctly.
- **`Confidence = 0.0`** — The model returned coordinates or player IDs that failed grounding validation. Ensure your mock telemetry is accurately populated.
- **`HTTP 429 Too Many Requests`** — You've hit the Fireworks concurrency limit. Scale down request intervals or increase retry parameters in `config/ai.yaml`.

---

<div align="center">

⚡ *Powered by FootballIQ Intelligence Engines*

🛡️ Built for Coaches &nbsp;|&nbsp; 👥 Trusted by Analysts &nbsp;|&nbsp; ⚡ Powered by AI &nbsp;|&nbsp; 🎯 Designed for Performance

</div>
