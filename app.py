import base64
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from cryptography import (
    sign, verify, make_qr, split_shares,
    QR_FILE, SHARE_A, SHARE_B, RECON_FILE,
)

LOG_FILE = Path(__file__).parent / "transactions.csv"
LOG_FIELDS = ["txid", "timestamp", "sender", "receiver", "amount", "signature", "verified"]


def _b64(path):
    return base64.b64encode(path.read_bytes()).decode()


def log_transaction(tx, signature_hex, verified):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "txid": tx.txid,
            "timestamp": tx.timestamp.isoformat(),
            "sender": tx.sender,
            "receiver": tx.receiver,
            "amount": tx.amount,
            "signature": signature_hex,
            "verified": verified,
        })

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class Transaction(BaseModel):
    sender: str
    receiver: str
    amount: float
    txid: str = Field(default_factory=lambda: "TX" + uuid4().hex[:6].upper())
    timestamp: datetime = Field(default_factory=datetime.now)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"title": "Send Money"})


@app.post("/send")
def send(
    request: Request,
    sender: str = Form(min_length=1),
    receiver: str = Form(min_length=1),
    amount: float = Form(gt=0),
):
    tx = Transaction(sender=sender, receiver=receiver, amount=amount)
    message = tx.model_dump_json().encode()
    signature = sign(message)
    verified = verify(message, signature)
    qr_matrix = make_qr(tx, signature)
    split_shares(qr_matrix)
    log_transaction(tx, signature.hex(), verified)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "Send Money",
            "tx": tx,
            "signature": signature.hex(),
            "verified": verified,
            "qr_b64": _b64(QR_FILE),
            "share_a": _b64(SHARE_A),
            "share_b": _b64(SHARE_B),
            "recon": _b64(RECON_FILE),
        },
    )
