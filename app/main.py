from fastapi import FastAPI
from app.routes import item_routes

app = FastAPI()

app.include_router(item_routes.router)