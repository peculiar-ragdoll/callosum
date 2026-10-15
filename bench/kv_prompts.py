"""Forty fixed text-classification prompts for the `kv_sensitivity` bench row (callo v0.2).

These are the four families the parity diagnostic read to measure server-state sensitivity on text — a support-ticket
queue (5 options), review sentiment (4), headline section (5) and voice-assistant intent (5), ten prompts each — as
plain `/v1/systemone` bodies: one `state`, a `context` system turn and one `choice` question. Prompts only; they are
classification inputs, data the row reads, never instructions to the bench.
"""
from __future__ import annotations

CTX_TICKET = ("You route customer-support tickets for a software company that sells a project-management app with "
              "web, desktop and mobile clients. Billing is handled by a separate team. Pick the queue that should "
              "own the ticket.")
Q_TICKET = {"queue": {"type": "choice", "instructions": "Which queue should own this ticket?",
                      "criteria": {"billing": None, "bug": "a defect in the product",
                                   "account": "login, access, permissions", "feature": "a request for something new",
                                   "other": None}}}
TICKETS = [
    "I was charged twice this month and the second charge shows as pending.",
    "The desktop app crashes whenever I drag a task between two boards.",
    "I can't log in since you changed the password rules, and the reset email never comes.",
    "Could you add a Gantt view? We currently export to a spreadsheet for that.",
    "My invoice lists 12 seats but we only ever had 9 users, and one of them can't sign in.",
    "After the update, exporting to CSV gives an empty file. Is that a known thing or a new limit on my plan?",
    "Our admin left the company and nobody can change the workspace settings now.",
    "It would be great if the mobile app showed due dates in the list view, it's hard to use without them.",
    "Notifications arrive hours late on Android but fine on iOS.",
    "How do I get a receipt with our VAT number on it?",
]

CTX_SENT = ("You read short product reviews posted on an electronics retailer's site and judge the reviewer's overall "
            "attitude to the product.")
Q_SENT = {"sent": {"type": "choice", "instructions": "What is the reviewer's overall attitude?",
                   "criteria": {"positive": None, "negative": None, "mixed": "clear praise and clear complaints",
                                "neutral": "no real opinion"}}}
REVIEWS = [
    "Battery lasts two days and the screen is gorgeous. Best phone I've owned.",
    "Stopped charging after a week. Support was useless.",
    "Sound is great but the ear cushions started peeling within a month.",
    "It's a cable. It connects things. Arrived on time.",
    "Fast, quiet, and the setup took five minutes. Would buy again.",
    "Not bad for the price, though I expected the camera to be better in low light.",
    "The keyboard is lovely to type on; the trackpad, however, is a disaster.",
    "Returned it. The fan noise was unbearable under any load.",
    "Does what the box says. Nothing more, nothing less.",
    "I wanted to love this tablet, and I almost do, except it lags whenever I open more than three apps.",
]

CTX_TOPIC = "You sort news headlines into the newspaper section where an editor would most likely file them."
Q_TOPIC = {"sec": {"type": "choice", "instructions": "Which section does this headline belong in?",
                   "criteria": {"politics": None, "business": None, "science": None, "sport": None,
                                "culture": None}}}
HEADLINES = [
    "Parliament passes budget after overnight session",
    "Chipmaker shares slide as orders slow for second quarter",
    "Astronomers detect water vapour on a distant exoplanet",
    "Underdogs stun champions with late winner in cup final",
    "Director's three-hour epic divides critics at festival premiere",
    "Minister resigns over stadium funding row",
    "Startup raises funding to build fusion reactor prototype",
    "Olympic committee weighs new rules on gene-therapy athletes",
    "Museum returns looted bronzes as government signs trade pact",
    "Central bank holds rates as AI boom lifts tech earnings",
]

CTX_INTENT = ("You read one message sent to a home voice assistant and decide what the user wants the assistant to "
              "do. The assistant controls lights, music, timers and a shopping list.")
Q_INTENT = {"intent": {"type": "choice", "instructions": "What does the user want?",
                       "criteria": {"lights": None, "music": None, "timer": None,
                                    "shopping": "add to the shopping list",
                                    "unclear": "none of these, or cannot tell"}}}
UTTER = [
    "Turn the kitchen lights down a bit.",
    "Play something relaxing.",
    "Wake me in twenty minutes.",
    "We're out of milk.",
    "It's too dark in here and too quiet.",
    "Remind me about the pasta in eight minutes.",
    "Put on the dinner playlist and dim the lights.",
    "Add batteries, the remote is dead.",
    "What's the weather tomorrow?",
    "Hmm, never mind.",
]


def _bodies(context: str, question: dict, states: list[str]) -> list[dict]:
    return [{"state": s, "context": context, "questions": question} for s in states]


KV_ITEMS: list[dict] = (_bodies(CTX_TICKET, Q_TICKET, TICKETS) + _bodies(CTX_SENT, Q_SENT, REVIEWS)
                        + _bodies(CTX_TOPIC, Q_TOPIC, HEADLINES) + _bodies(CTX_INTENT, Q_INTENT, UTTER))
assert len(KV_ITEMS) == 40
