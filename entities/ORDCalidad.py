from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ORDCalidad(Base):
    __tablename__ = 'ORDCalidad'  # Nombre de la tabla en la base de datos

    id = Column(Integer, primary_key=True, autoincrement=True)
    CALIDAD = Column(String,name='Calidad', nullable=False)
    ORDEN = Column(Integer, name='Orden',nullable=False)

    def __repr__(self):
        return f"<ORDCalidad(id={self.id}, calidad='{self.CALIDAD}', orden={self.ORDEN})>"
