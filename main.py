from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
app = FastAPI()
DB_URL = "postgresql://postgres:sanjay@localhost:5433/phsd"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
@app.get("/products")
def get_products():
    session = SessionLocal()
    try:
        result = session.execute(text("SELECT * FROM products"))
        products = [dict(row) for row in result.mappings()]
        return {"products": products}
    finally:
        session.close()