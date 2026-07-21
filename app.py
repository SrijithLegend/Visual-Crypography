from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")


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
    result = f"Sent ${amount:.2f} from {sender} to {receiver}."
    return templates.TemplateResponse(
        request, "index.html", {"title": "Send Money", "result": result}
    )
