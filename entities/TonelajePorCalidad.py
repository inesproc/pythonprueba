from sqlalchemy import Column, Integer, String, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TonelajePorCalidad(Base):
    __tablename__ = "TonelajePorCalidad"
    
    Id = Column(Integer, primary_key=True, autoincrement=True, name="Id")
    TonUpgrade = Column(DECIMAL(18, 2), nullable=False, name="TonUpgrade")
    TonOriginal = Column(DECIMAL(18, 2), nullable=False, name="TonOriginal")
    Calidad = Column(String(255), nullable=False, name="Calidad")
    Proteina = Column(DECIMAL(18, 2), nullable=False, name="Proteina")
    TVN = Column(DECIMAL(18, 2), nullable=False, name="TVN")
    Histaminappm = Column(Integer, nullable=False, name="Histaminappm")
    Humedad = Column(DECIMAL(18, 2), nullable=False, name="Humedad")
    Grasas = Column(DECIMAL(18, 2), nullable=False, name="Grasas")
    Cenizas = Column(DECIMAL(18, 2), nullable=False, name="Cenizas")
    Cloruros = Column(DECIMAL(18, 2), nullable=False, name="Cloruros")
    Arena = Column(DECIMAL(18, 2), nullable=False, name="Arena")
    Acidez = Column(DECIMAL(18, 2), nullable=False, name="Acidez")
    UpgradePorc = Column(DECIMAL(18, 2), nullable=False, name="UpgradePorc")
    UpgradeFormula = Column(String(1000), nullable=False, name="UpgradeFormula")
    Fecha = Column(DateTime, nullable=False, name="Fecha")
    IdBatch = Column(String(50), nullable=False, name="IdBatch")
    IdEjecucion = Column(Integer, nullable=False, name="IdEjecucion")
    Feccrea = Column(DateTime, nullable=False, name="Feccrea")
