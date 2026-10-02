import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = "sqlite:///./agriforecast.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="farmer") # "farmer" or "mandi_officer"
    full_name = Column(String(100), nullable=True)
    state = Column(String(50), nullable=True)
    preferred_mandi = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    forecasts = relationship("SavedForecast", back_populates="owner")
    reports = relationship("MandiReport", back_populates="officer")

class SavedForecast(Base):
    __tablename__ = "saved_forecasts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    market_name = Column(String(100))
    commodity = Column(String(100))
    predicted_modal_price = Column(Float)
    predicted_kg_price = Column(Float)
    perishability_tier = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="forecasts")

class MandiReport(Base):
    __tablename__ = "mandi_reports"

    id = Column(Integer, primary_key=True, index=True)
    officer_id = Column(Integer, ForeignKey("users.id"))
    market_name = Column(String(100))
    commodity = Column(String(100))
    reported_price = Column(Float)
    arrival_tonnes = Column(Float)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    officer = relationship("User", back_populates="reports")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
