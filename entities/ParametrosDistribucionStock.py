from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ParametrosDistribucionStock(Base):
    __tablename__ = 'ParametrosDistribucionStock'

    Id = Column(Integer, primary_key=True)
    FechaInicio = Column(DateTime, nullable=False)
    FechaFin = Column(DateTime, nullable=False)

    # Relación con ParametrosTemporada
    temporadas = relationship('ParametrosTemporada', backref='parametros', cascade='all, delete-orphan')

class ParametrosDistribucionStock_TipoHP(Base):
    __tablename__ = 'ParametrosDistribucionStock_TipoHP'

    Id = Column(Integer, primary_key=True)
    TipoHP = Column(String(255), nullable=False)

class ParametrosTemporada(Base):
    __tablename__ = 'ParametrosTemporada'

    Id = Column(Integer, primary_key=True)
    ParametrosId = Column(Integer, ForeignKey('ParametrosDistribucionStock.Id', ondelete='CASCADE'), nullable=False)
    Temporada = Column(String(100), nullable=False)
