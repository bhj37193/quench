"""Locust load test for Quench.

Run:
    uvicorn src.proxy:app --port 4141 &
    locust -f load_test/locustfile.py --headless -u 50 -r 5 --run-time 3m \
           --host http://localhost:4141
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

# Repeated/paraphrased prompts — designed to generate cache hits after warmup
_PROMPTS = [
    # capital-of-france group
    ("What is the capital of France?", "You are a geography expert."),
    ("Which city is the capital of France?", "You are a geography expert."),
    ("Name the capital city of France.", "You are a geography expert."),
    ("What's France's capital city?", "You are a geography expert."),
    # http-404 group
    ("What does HTTP 404 mean?", "You are a web developer."),
    ("Explain the HTTP 404 error code.", "You are a web developer."),
    ("What is a 404 error?", "You are a web developer."),
    # reverse-string group
    ("How do I reverse a string in Python?", "You are a Python expert."),
    ("What's the Python way to flip a string?", "You are a Python expert."),
    ("How to reverse str in Python?", "You are a Python expert."),
    # what-is-ml group
    ("What is machine learning?", "You are an AI expert."),
    ("Define machine learning.", "You are an AI expert."),
    ("Explain ML to me.", "You are an AI expert."),
]


class QuenchUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        # Warm the cache by seeding common questions first
        for prompt, system in _PROMPTS[:4]:
            self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                },
                name="/v1/chat/completions [warmup]",
            )

    @task
    def chat(self) -> None:
        prompt, system = random.choice(_PROMPTS)
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-3-5-haiku-20241022",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            },
            name="/v1/chat/completions",
        )
