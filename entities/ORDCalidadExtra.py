from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ORDCalidadExtra(Base):
    __tablename__ = 'ORDCalidadExtra'  # Nombre de la tabla en la base de datos

    id = Column(Integer, primary_key=True, autoincrement=True,name='Id')
    Calidad = Column(String, nullable=False,name='Calidad')
    Orden = Column(Integer, nullable=False,name='Orden')

    def __repr__(self):
        return f"<ORDCalidadExtra(id={self.id}, calidad='{self.calidad}', orden={self.orden})>"
