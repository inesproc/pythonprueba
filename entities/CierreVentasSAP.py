from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.types import DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CierreVentasSAP(Base):
    __tablename__ = 'CierreVentasSAP'  # Nombre de la tabla en la base de datos

    # Definición de columnas con nombres para identificarlos desde la base de datos
    Id = Column(Integer, primary_key=True, autoincrement=True, name='Id')
    # Indicador = Column(String(255), name='Indicador')
    NroPedido = Column(String(255), name='NroPedido')
    Documento = Column(String(255), name='Documento')
    CondPago = Column(String(255), name='CondPago')
    FechaCierre = Column(DateTime, name='FechaCierre')
    Posicion = Column(Integer, name='Posicion')
    Cdis = Column(String(255), name='Cdis')
    Comprador = Column(String(255), name='Comprador')
    Producto = Column(String(255), name='Producto')
    TMTotal = Column(DECIMAL(18, 2), name='TMTotal')
    Atendido = Column(DECIMAL(18, 2), name='Atendido')
    Pendiente = Column(DECIMAL(18, 2), name='Pendiente')
    ValorFOB = Column(DECIMAL(18, 2), name='ValorFOB')
    # ValorPendiente = Column(DECIMAL(18, 2), name='ValorPendiente')
    PrecioUnitario = Column(DECIMAL(18, 2), nullable=True, name='PrecioUnitario')
    # Tolerancia = Column(DECIMAL(18, 2), name='Tolerancia')
    Temporada = Column(String(255), name='Temporada')
    ValidoDe = Column(DateTime, name='ValidoDe')
    ValidoA = Column(DateTime, name='ValidoA')
    FechaProgramada = Column(DateTime, name='FechaProgramada')
    IdBatch = Column(String(255), name='IdBatch')
    FecCrea = Column(DateTime, name='FecCrea')
    Observacion = Column(String, name='Observacion')
    def __repr__(self):
        return f"<CierreVentasSAP(Id={self.Id}, NroPedido={self.NroPedido}, Documento={self.Documento})>"
