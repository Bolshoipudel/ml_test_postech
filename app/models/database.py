"""SQLAlchemy модели базы данных."""
from datetime import datetime, date
from typing import List
from sqlalchemy import (
    Column, Integer, String, Text, Date, Boolean,
    ForeignKey, TIMESTAMP, ARRAY
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    head_name = Column(String(100))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    products = relationship("Product", back_populates="department")
    team_members = relationship("TeamMember", back_populates="department")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    short_name = Column(String(50))
    description = Column(Text)
    department_id = Column(Integer, ForeignKey("departments.id"))
    status = Column(String(50), default="active")
    version = Column(String(20))
    release_date = Column(Date)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    department = relationship("Department", back_populates="products")
    features = relationship("Feature", back_populates="product")
    incidents = relationship("Incident", back_populates="product")


class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    position = Column(String(100))
    department_id = Column(Integer, ForeignKey("departments.id"))
    skills = Column(ARRAY(Text))
    experience_years = Column(Integer)
    join_date = Column(Date)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    department = relationship("Department", back_populates="team_members")
    assigned_features = relationship("Feature", back_populates="assigned_member")
    reported_incidents = relationship(
        "Incident",
        foreign_keys="Incident.reported_by",
        back_populates="reporter"
    )
    assigned_incidents = relationship(
        "Incident",
        foreign_keys="Incident.assigned_to",
        back_populates="assignee"
    )


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    product_id = Column(Integer, ForeignKey("products.id"))
    status = Column(String(50), default="in_development")
    priority = Column(String(20))
    assigned_to = Column(Integer, ForeignKey("team_members.id"))
    estimated_hours = Column(Integer)
    completed_hours = Column(Integer, default=0)
    start_date = Column(Date)
    target_date = Column(Date)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product", back_populates="features")
    assigned_member = relationship("TeamMember", back_populates="assigned_features")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    product_id = Column(Integer, ForeignKey("products.id"))
    severity = Column(String(20))
    status = Column(String(50), default="open")
    reported_by = Column(Integer, ForeignKey("team_members.id"))
    assigned_to = Column(Integer, ForeignKey("team_members.id"))
    reported_date = Column(Date, default=date.today)
    resolved_date = Column(Date)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product", back_populates="incidents")
    reporter = relationship(
        "TeamMember",
        foreign_keys=[reported_by],
        back_populates="reported_incidents"
    )
    assignee = relationship(
        "TeamMember",
        foreign_keys=[assigned_to],
        back_populates="assigned_incidents"
    )
