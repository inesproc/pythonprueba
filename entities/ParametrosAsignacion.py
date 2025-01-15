from datetime import datetime
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ParametrosAsignacion(Base):
    __tablename__ = 'ParametrosAsignacion'  # Nombre de la tabla en la base de datos

    # Definición de columnas
    Id = Column(Integer, primary_key=True, autoincrement=True, name='Id')
    NroPlanta = Column(Integer, nullable=False, name='NroPlanta')
    MinSacosRuma = Column(Integer, nullable=False, name='MinSacosRuma')
    ToneladasPorRuma = Column(Integer, nullable=False, name='ToneladasPorRuma')
    FechaFabricacion = Column(DateTime, nullable=True, name='FechaFabricacion')
    Feccrea = Column(DateTime, default=datetime.now, nullable=True, name='Feccrea')

