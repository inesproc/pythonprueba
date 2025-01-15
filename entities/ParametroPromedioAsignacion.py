from sqlalchemy import Column, Integer, String, Numeric, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ParametroPromedioAsignacion(Base):
    __tablename__ = 'ParametroPromedioAsignacion'

    Id = Column(Integer, primary_key=True, autoincrement=True)  # Llave primaria, incremental
    calidad = Column(String,name='Calidad', nullable=False, default='')  # Obligatorio
    proteina_min = Column(Numeric,name='ProteinaMin', nullable=False)  # Decimal
    tvn_max = Column(Numeric, name='TVNMax',nullable=False)  # Decimal
    histamina_max = Column(Numeric,name='HistaminaMax', nullable=False)  # Decimal
    humedad_max = Column(Numeric,name='HumedadMax', nullable=False)  # Decimal
    humedad_min = Column(Numeric,name='HumedadMin', nullable=False)  # Decimal
    grasas_max = Column(Numeric,name='GrasasMax', nullable=False)  # Decimal
    cenizas_max = Column(Numeric,name='CenizasMax', nullable=False)  # Decimal
    cloruros_max = Column(Numeric,name='ClorurosMax', nullable=False)  # Decimal
    acidez_max = Column(Numeric,name='AcidezMax', nullable=False)  # Decimal
    precio = Column(Integer,name='Precio',  nullable=False)  # Entero

    def __repr__(self):
        return (
            f"<ParametroPromedio(id={self.Id}, calidad='{self.calidad}', proteina_min={self.proteina_min}, "
            f"tvn_max={self.tvn_max}, histamina_max={self.histamina_max}, humedad_max={self.humedad_max}, "
            f"humedad_min={self.humedad_min}, grasas_max={self.grasas_max}, cenizas_max={self.cenizas_max}, "
            f"cloruros_max={self.cloruros_max}, acidez_max={self.acidez_max},  "
            f" precio={self.precio})>"
        )
