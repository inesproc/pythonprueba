from sqlalchemy import Column, Integer, String, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TonelajePorCalidadPorContrato(Base):
    __tablename__ = 'TonelajePorCalidadPorContrato'  # Nombre de la tabla en la base de datos
    
    # Definición de columnas con el parámetro name para coincidir con los nombres de las columnas en SQL
    Id = Column('Id', Integer, primary_key=True, autoincrement=True)
    NroPedido = Column('NroPedido', String(255), nullable=False)
    Calidad = Column('Calidad', String(255), nullable=False)
    Tipo = Column('Tipo', String(255), nullable=False)
    PrecioUnitario = Column('PrecioUnitario', DECIMAL(18, 2), nullable=False)
    TM = Column('TM', DECIMAL(18, 2), nullable=False)
    TM_Asignar = Column('TM_Asignar', DECIMAL(18, 2), nullable=False)
    FechaEjecucion = Column('FechaEjecucion', DateTime, nullable=False)
    FechaStock = Column('FechaStock', DateTime, nullable=False)
    idBatch = Column('idBatch', String(255), nullable=False)
    feccrea = Column('feccrea', DateTime, default='getdate()', nullable=False)
    Material=Column('Material', String(255), nullable=True)
    Destino=Column('Destino', String(255), nullable=True)
    Comprador=Column('Comprador', String(255), nullable=True)
    Tonelaje=Column('Tonelaje',DECIMAL(18,2),nullable=True)
    EstadoId=Column('EstadoId',Integer,nullable=True)
    
   