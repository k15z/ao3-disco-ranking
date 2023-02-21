# flake8: noqa
import os
import time
from random import choice

import requests

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
    "Mozilla/5.0 (Linux; Android 11; SM-G960U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.72 Mobile Safari/537.36",
]

url = "http://archiveofourown.org/works"

use_proxy = False
if "AO3_DISCO_PROXY_USER" in os.environ:
    use_proxy = True
    username = os.environ["AO3_DISCO_PROXY_USER"]
    password = os.environ["AO3_DISCO_PROXY_PASS"]
    proxy = f"http://{username}:{password}@dc.smartproxy.com:10000"


def get(url: str) -> bytes:
    headers = {"User-Agent": choice(user_agents)}
    proxies = {"http": proxy, "https": proxy} if use_proxy else {}
    try:
        response = requests.get(url, headers=headers, timeout=30, proxies=proxies)
    except requests.exceptions.ReadTimeout as e:
        time.sleep(10)
        raise e
    if response.status_code == 429:
        time.sleep(10.0)
        raise ValueError("Too many requests!")
    if response.status_code >= 404:
        raise KeyError("This work doesn't exist.")
    if response.status_code >= 400:
        raise ValueError("Try again later!", response, url)
    return response.content
