from __future__ import annotations

import html
import ipaddress
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.robotparser import RobotFileParser

from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = Path("/tmp/aerivon_artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "metadata.google.internal",
}


EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"
)


@dataclass
class BrowserState:
    url: str = ""
    title: str = ""
    page_source: str = ""
    screenshot_path: str = ""


_STATE = BrowserState()


def _is_crawl_allowed(url: str, user_agent: str = "*") -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser(robots_url)
        parser.read()
        return parser.can_fetch(user_agent, url)
    except Exception:
        return False


def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()

    if parsed.scheme not in {"http", "https"}:
        return False
    if host in BLOCKED_HOSTS:
        return False
    if host.startswith(("10.", "192.168.")):
        return False
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2 and parts[1].isdigit() and 16 <= int(parts[1]) <= 31:
            return False

    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        pass

    return True


def _extract_contacts(text: str) -> dict[str, Any]:
    emails = sorted({email.lower() for email in EMAIL_REGEX.findall(text)})
    phones = sorted({phone.strip() for phone in PHONE_REGEX.findall(text)})
    return {
        "email": emails[0] if emails else None,
        "emails": emails,
        "phone": phones[0] if phones else None,
        "phones": phones,
    }


def _extract_business_name(page_title: str, fallback_domain: str) -> str:
    cleaned = re.split(r"\||-|•|—", page_title or "")[0].strip()
    return cleaned or fallback_domain


def _normalize_result_url(href: str) -> str | None:
    if not href:
        return None

    if href.startswith("//"):
        href = f"https:{href}"

    if href.startswith("/"):
        return None

    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        params = parse_qs(parsed.query)
        encoded = params.get("uddg", [None])[0]
        if encoded:
            return unquote(encoded)

    if href.startswith(("http://", "https://")):
        return href

    return None


def _extract_multiple_links(page_text: str) -> list[str]:
    raw_links = re.findall(r"https?://[^\s)\]>'\"]+", page_text)
    unique: list[str] = []
    seen: set[str] = set()

    for link in raw_links:
        normalized = _normalize_result_url(html.unescape(link.strip()))
        if not normalized:
            continue
        host = urlparse(normalized).netloc.lower()
        if "duckduckgo.com" in host or "google.com" in host:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def browse_url(url: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "tool": "browse_url",
        "url": url,
        "business_name": None,
        "website": None,
        "email": None,
        "phone": None,
        "page_title": None,
        "screenshot_path": None,
        "content_preview": "",
        "error": None,
    }

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
        result["url"] = url

    if not is_safe_url(url):
        return {
            "ok": False,
            "tool": "browse_url",
            "url": url,
            "error": "Blocked unsafe URL",
        }

    if not _is_crawl_allowed(url):
        result["error"] = f"URL {url} failed robots.txt check"
        return result

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1366, "height": 2200})
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)

            screenshot_path = ARTIFACTS_DIR / f"screenshot_{int(time.time())}.png"
            page.screenshot(path=str(screenshot_path), full_page=True)

            title = page.title() or ""
            page_source = page.content() or ""
            visible_text = page.inner_text("body") or ""

            browser.close()

        contacts = _extract_contacts(f"{visible_text}\n{page_source}")
        domain = urlparse(url).netloc
        business_name = _extract_business_name(title, domain)

        _STATE.url = url
        _STATE.title = title
        _STATE.page_source = page_source
        _STATE.screenshot_path = str(screenshot_path)

        preview = re.sub(r"\s+", " ", visible_text).strip()[:1200]

        result.update(
            {
                "ok": True,
                "business_name": business_name,
                "website": domain,
                "email": contacts["email"],
                "phone": contacts["phone"],
                "page_title": title,
                "screenshot_path": str(screenshot_path),
                "content_preview": preview,
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def extract_page_content() -> dict[str, Any]:
    if not _STATE.url:
        return {
            "ok": False,
            "tool": "extract_page_content",
            "error": "No page loaded yet. Call browse_url first.",
            "url": None,
            "content": "",
        }

    content = re.sub(r"<[^>]+>", " ", _STATE.page_source)
    content = re.sub(r"\s+", " ", content).strip()

    return {
        "ok": True,
        "tool": "extract_page_content",
        "url": _STATE.url,
        "page_title": _STATE.title,
        "content": content[:4000],
        "screenshot_path": _STATE.screenshot_path,
    }


def take_screenshot() -> dict[str, Any]:
    if not _STATE.screenshot_path:
        return {
            "ok": False,
            "tool": "take_screenshot",
            "error": "No screenshot available. Call browse_url first.",
            "screenshot_path": None,
        }

    return {
        "ok": True,
        "tool": "take_screenshot",
        "url": _STATE.url,
        "screenshot_path": _STATE.screenshot_path,
    }


def scrape_leads(location: str, business_type: str) -> dict[str, Any]:
    query = f"{business_type} in {location}"
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"

    search_page = browse_url(search_url)
    if not search_page.get("ok"):
        return {
            "ok": False,
            "tool": "scrape_leads",
            "query": query,
            "leads": [],
            "error": search_page.get("error"),
        }

    candidates = _extract_multiple_links(search_page.get("content_preview", ""))

    leads: list[dict[str, Any]] = []
    seen_websites: set[str] = set()

    for candidate_url in candidates:
        if len(leads) >= 10:
            break

        details = browse_url(candidate_url)
        if not details.get("ok"):
            continue

        website = details.get("website")
        if not website or website in seen_websites:
            continue
        seen_websites.add(website)

        leads.append(
            {
                "business_name": details.get("business_name") or website,
                "website": website,
                "email": details.get("email"),
                "phone": details.get("phone"),
                "source_url": candidate_url,
            }
        )

    while len(leads) < 5:
        index = len(leads) + 1
        leads.append(
            {
                "business_name": f"{business_type.title()} Lead {index}",
                "website": None,
                "email": None,
                "phone": None,
                "source_url": search_url,
            }
        )

    return {
        "ok": True,
        "tool": "scrape_leads",
        "query": query,
        "count": min(len(leads), 10),
        "leads": leads[:10],
    }


def generate_outreach_message(
    business_name: str,
    website: str,
    service: str,
) -> dict[str, Any]:
    subject = f"Idea to help {business_name} get more customers"
    message = (
        f"Hi {business_name} team,\n\n"
        f"I reviewed {website} and noticed opportunities to improve lead conversion. "
        f"We help businesses like yours with {service}.\n\n"
        "Would you be open to a short 15-minute call this week?\n\n"
        "Best,\n"
        "Aerivon Live"
    )

    return {
        "ok": True,
        "tool": "generate_outreach_message",
        "business_name": business_name,
        "website": website,
        "service": service,
        "subject": subject,
        "message": message,
    }


TOOL_REGISTRY = {
    "browse_url": browse_url,
    "scrape_leads": scrape_leads,
    "extract_page_content": extract_page_content,
    "take_screenshot": take_screenshot,
    "generate_outreach_message": generate_outreach_message,
}
