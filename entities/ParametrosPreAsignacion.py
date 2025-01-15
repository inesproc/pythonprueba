from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ParametrosPreAsignacion(Base):
    __tablename__ = 'ParametrosPreAsignacion'  # Nombre de la tabla en la base de datos

    Id = Column(Integer, primary_key=True, autoincrement=True)
    PorcentajeUpgrade = Column(Float, nullable=False)
    NumeroSacos = Column(Integer, nullable=False)
    PermitirDowngrade = Column(Boolean, nullable=False)
    UserCrea = Column(String, nullable=False)
    FecCrea = Column(DateTime, nullable=False)
    FechaFabricacion = Column(DateTime, nullable=False)
    ToneladasPorRuma = Column(Integer, nullable=True)

    def __repr__(self):
        return (f"<ParametrosPreAsignacion("
                f"Id={self.Id}, PorcentajeUpgrade={self.PorcentajeUpgrade}, "
                f"NumeroSacos={self.NumeroSacos}, PermitirDowngrade={self.PermitirDowngrade}, "
                f"UserCrea='{self.UserCrea}', ToneladasPorRuma={self.ToneladasPorRuma}, "
                f"FechaFabricacion={self.FechaFabricacion}, FecCrea={self.FecCrea})>")
