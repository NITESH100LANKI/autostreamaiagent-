"""
agent/tools.py
--------------
Tool wrappers used by the ToolExecutor node.

mock_lead_capture  : Simulates writing a lead to a CRM / database.
validate_email     : Basic regex email validator.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Email validation ────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def validate_email(email: str) -> bool:
    """Return True if *email* matches a basic RFC-style pattern."""
    return bool(_EMAIL_RE.match(email.strip()))


# ── Lead capture ────────────────────────────────────────────────────────────
def mock_lead_capture(name: str, email: str, platform: str) -> bool:
    """
    Mock CRM submission.

    In production, replace with a real API call (HubSpot, Salesforce, etc.).

    Parameters
    ----------
    name     : Full name of the lead.
    email    : Verified email address.
    platform : Primary video platform (YouTube, Instagram, …).

    Returns
    -------
    bool : True on success, False on failure.
    """
    try:
        if not all([name.strip(), email.strip(), platform.strip()]):
            logger.warning("mock_lead_capture called with incomplete data.")
            return False

        logger.info("Lead captured → name=%s | email=%s | platform=%s", name, email, platform)
        print(f"\n✅ [CRM] Lead captured → {name} | {email} | {platform}")
        return True

    except Exception as exc:
        logger.error("mock_lead_capture failed: %s", exc)
        return False
