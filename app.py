from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from cryptography import sign, verify

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
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "title": "Send Money",
            "tx": tx,
            "signature": signature.hex(),
            "verified": verify(message, signature),
        },
    )
