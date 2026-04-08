"""
Database service – PostgreSQL via SQLAlchemy.
Falls back gracefully if DATABASE_URL is not set.
"""
import os
import logging
from typing import Optional
from datetime import datetime
import json

log = logging.getLogger("moteur.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# SQLAlchemy optional
engine = None
SessionLocal = None
Base = None

if DATABASE_URL:
    try:
        from sqlalchemy import (
            create_engine, Column, String, Float, Integer,
            DateTime, Text, JSON
        )
        from sqlalchemy.orm import declarative_base, sessionmaker

        # Railway provides postgres:// but SQLAlchemy needs postgresql://
        db_url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        engine = create_engine(db_url, pool_pre_ping=True, pool_size=5)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base = declarative_base()

        class ForecastJob(Base):
            __tablename__ = "forecast_jobs"
            id = Column(String, primary_key=True)
            created_at = Column(DateTime, default=datetime.utcnow)
            status = Column(String, default="pending")       # pending|running|done|error
            frequency = Column(String, nullable=True)
            horizon = Column(Integer, nullable=True)
            data_points = Column(Integer, nullable=True)
            best_model = Column(String, nullable=True)
            best_mape = Column(Float, nullable=True)
            result = Column(JSON, nullable=True)
            error = Column(Text, nullable=True)

        class ForecastRanking(Base):
            __tablename__ = "forecast_rankings"
            id = Column(Integer, primary_key=True, autoincrement=True)
            job_id = Column(String, nullable=False)
            model = Column(String, nullable=False)
            mape = Column(Float, nullable=True)
            smape = Column(Float, nullable=True)
            mae = Column(Float, nullable=True)
            rmse = Column(Float, nullable=True)
            r2 = Column(Float, nullable=True)
            wape = Column(Float, nullable=True)
            bias = Column(Float, nullable=True)
            rank = Column(Integer, nullable=True)
            created_at = Column(DateTime, default=datetime.utcnow)

        Base.metadata.create_all(bind=engine)
        log.info("PostgreSQL connected – tables created")

    except Exception as exc:
        log.warning("PostgreSQL unavailable: %s – running without DB", exc)
        engine = None
else:
    log.info("No DATABASE_URL – running without persistence")


def get_db():
    """FastAPI dependency – yields DB session."""
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_job(db, job_id: str, status: str, result: dict = None, error: str = None):
    """Persist a forecast job result."""
    if db is None:
        return
    try:
        from sqlalchemy import text
        job = db.query(ForecastJob).filter(ForecastJob.id == job_id).first()
        if job is None:
            job = ForecastJob(id=job_id)
            db.add(job)
        job.status = status
        if result:
            job.best_model = result.get("best_model")
            job.frequency = result.get("frequency")
            job.horizon = result.get("horizon")
            job.data_points = result.get("data_points")
            rankings = result.get("backtesting", {}).get("rankings", [])
            if rankings:
                job.best_mape = rankings[0].get("mape")
            job.result = result
        if error:
            job.error = error
        db.commit()
    except Exception as exc:
        log.error("DB save_job error: %s", exc)
        db.rollback()


def list_jobs(db, limit: int = 50) -> list:
    """List recent forecast jobs."""
    if db is None:
        return []
    try:
        jobs = (
            db.query(ForecastJob)
            .order_by(ForecastJob.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": j.id,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "status": j.status,
                "best_model": j.best_model,
                "best_mape": j.best_mape,
                "frequency": j.frequency,
                "horizon": j.horizon,
                "data_points": j.data_points,
            }
            for j in jobs
        ]
    except Exception as exc:
        log.error("DB list_jobs error: %s", exc)
        return []


def get_job(db, job_id: str) -> Optional[dict]:
    """Get a specific forecast job."""
    if db is None:
        return None
    try:
        job = db.query(ForecastJob).filter(ForecastJob.id == job_id).first()
        if not job:
            return None
        return {
            "id": job.id,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "status": job.status,
            "best_model": job.best_model,
            "best_mape": job.best_mape,
            "frequency": job.frequency,
            "horizon": job.horizon,
            "data_points": job.data_points,
            "result": job.result,
            "error": job.error,
        }
    except Exception as exc:
        log.error("DB get_job error: %s", exc)
        return None
