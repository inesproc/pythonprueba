from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from entities.RumaSAP import RumaSAP  # Asegúrate de importar tu clase Config
from entities.ParametrosPreAsignacion import ParametrosPreAsignacion
from entities.ORDCalidad import ORDCalidad
from entities.ORDCalidadExtra import ORDCalidadExtra
from entities.ParametroPromedio import ParametroPromedio
from entities.ParametroRuma import ParametroRuma
from entities.CierreVentasSAP import CierreVentasSAP
from entities.TonelajePorCalidad import TonelajePorCalidad
from entities.TonelajePorCalidadPorContrato import TonelajePorCalidadPorContrato
from entities.ParametrosDistribucionStock import ParametrosDistribucionStock
from entities.ParametrosAsignacion import ParametrosAsignacion
from entities.Ruma import Ruma
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import joinedload
import logging
from configDB import Config
from entities.ParametroPromedioAsignacion import ParametroPromedioAsignacion
from entities.ParametroPromedioPreAsignacion import ParametroPromedioPreAsignacion
from entities.ParametroRumaAsignacion import ParametroRumaAsignacion
from entities.ParametroRumaPreAsignacion import ParametroRumaPreAsignacion
from entities.ParametrosDistribucionStock import ParametrosTemporada

class DatabaseManager:
    def __init__(self):
        # Crear motor y sesión
        print("###"*20)
        print(Config.SQLALCHEMY_DATABASE_URI)
        self.engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        self.Session = sessionmaker(bind=self.engine)
        logging.info("DatabaseManager inicializado con éxito.")
    
    def get_filtered_and_sorted_ruma_sap(self, id_batch_value):
        """
        Devuelve los registros de la tabla RumaSAP filtrados por idBatch
        y ordenados por un campo específico.input pre-asignaciones
        """
        session = self.Session()
        try:
            # Filtrar por idBatch y ordenar por un campo, como 'id' o 'fecha_pre_asignacion'.
            registros = (
                session.query(RumaSAP)
                .filter(RumaSAP.IdBatch == id_batch_value)
                .order_by(RumaSAP.Id)  # Cambia `RumaSAP.id` por el campo que prefieras.
                .all()
            )
             # Convertir registros a una lista de diccionarios para Pandas
            registros_dict = [
                {
                    "Ruma Nro":r.RumaNro,
                    "CALIDAD":r.CALIDAD,
                    "Cantidad": r.Cantidad,
                    "proteina": r.ProteinaPorc,
                    "tvn": r.TVNmg100g,
                    "histamina": r.Histaminappm,
                    "humedad": r.HumedadPorc,
                    "grasa": r.GrasasPorc,
                    "cenizas": r.CenizasPorc,
                    "cloruros": r.ClorurosPorc,
                    "Arena             %": r.ArenaPorc,
                    "acidez": r.AcidezPorc,
                    "A/O             ppm": r.AOppm,
                    "Cadmio": r.Cadmio,
                    "Hierro": r.Hierro,
                    "Proximal": r.Proximal,
                    "Granulometría %": r.GranulometriaPorc,
                    "Dens.Apar. g/ml": r.DensAparGml,
                    "Dens.Comp.  g/ml": r.DensCompGml,
                    "Nro.Flujo cm.": r.NroFlujoCm,
                    "U.M.": r.UM,
                    "Código": r.Codigo,
                    "Descripción del Material": r.DescripcionMaterial,
                    "Centro Ubicación": r.CentroUbicacion,
                    "Almacen Ubicación": r.AlmacenUbicacion,
                    "Tipo Producción": r.TipoProduccion,
                    "Centro Producción": r.CentroProduccion,
                    "Calidad Planta": r.CalidadPlanta,
                    "Cierre Vta": r.CierreVta,
                    "Posición": r.Posicion,
                    "Material": r.Material,
                    "Cant. Pre Asignado": r.CantPreAsignado,
                    "Cant. Transito": r.CantTransito,
                    "Cant. Lote": r.CantLote,
                    "Lote Exp.": r.LoteExp,
                    "Ubicación en almacen": r.UbicacionAlmacen,
                    "Fecha de Contabilización": r.FechaContabilizacion,
                    "Fecha de Fabricación": r.FechaFabricacion,
                    "Certificadora": r.Certificadora,
                    "F. Anal  Fco Qco": r.FAnalFcoQco,
                    "F. Anal  Micobiol": r.FAnalMicobiol,
                    "F.V. Anal  Fco Qco": r.FVAnalFcoQco,
                    "F.V. Anal  Micobiol": r.FVAnalMicobiol,
                    "Salmonella": r.Salmonella,
                    "Shiguella": r.Shiguella,
                    "Enterobact ufc/g": r.EnterobactUfcG,
                    "Intervalo": r.Intervalo,
                    "Nro.": r.Nro,
                    "Clase Insp.": r.ClaseInsp,
                    "Temp. Elevada °C": r.TempElevadaC,
                    "Color Harina": r.ColorHarina,
                    "Presencia Mat.Extraña": r.PresenciaMatExtrana,
                    "% Camaroncillo": r.CamaroncilloPorc,
                    "% Camaroncillo Cualit": r.CamaroncilloCualit,
                    "UPGRADE":r.Upgrade,
                    "TM":r.TM,
                    "KG":r.KG,
                    "prod":r.Prod,
                    "f_fab":r.FFab,
                    "f_fab_final":r.FFabFinal,
                    "f_fab_ultima":r.FFabUltima,
                    "descripcion_material":r.DescripcionMaterial
                }
                for r in registros
            ]


            # Crear un DataFrame de Pandas
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()
    
    def get_first_parametro_pre_asignacion(self):
        """
        Devuelve el primer registro de la tabla ParametrosPreAsignacion.
        """
        session = self.Session()
        try:
            first_record = session.query(ParametrosPreAsignacion).first()
            
            return first_record
        finally:
            session.close()

    def get_all_ord_calidad(self):
        """
        Devuelve todos los registros de la tabla ORDCalidad.
        """
        session = self.Session()
        try:
            results = session.query(ORDCalidad).all()
            registros_dict = [
                {
                    "CALIDAD": r.CALIDAD,
                    "ORDEN": r.ORDEN,                    
                }
                for r in results
            ]
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()

    def get_all_ord_calidad_extra(self):
        """
        Devuelve todos los registros de la tabla ORDCalidad.
        """
        session = self.Session()
        try:
            results = session.query(ORDCalidadExtra).all()
            registros_dict = [
                {
                    "CALIDAD": r.Calidad,
                    "ORDEN": r.Orden,                    
                }
                for r in results
            ]
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()
        
    def get_all_parametro_promedio_preasignacion(self):
        """Devuelve todos los registros de la tabla ParametroPromedio."""
        session = self.Session()
        try:
            results = session.query(ParametroPromedioPreAsignacion).all()  # Trae todos los registros
            registros_dict = [
                {
                    "CALIDAD": r.calidad,
                    "PROTEINA_MIN": r.proteina_min,
                    "TVN_MAX": r.tvn_max,
                    "HISTAMINA_MAX": r.histamina_max,
                    "HUMEDAD_MAX": r.humedad_max,
                    "HUMEDAD_MIN": r.humedad_min,
                    "GRASAS_MAX": r.grasas_max,
                    "CENIZAS_MAX": r.cenizas_max,
                    "CLORUROS_MAX": r.cloruros_max,
                    "ACIDEZ_MAX": r.acidez_max,
                    "PRECIO": r.precio,
                        }
                        for r in results
                    ]
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()

    def get_all_parametro_promedio_asignacion(self):
        """Devuelve todos los registros de la tabla ParametroPromedio."""
        session = self.Session()
        try:
            results = session.query(ParametroPromedioAsignacion).all()  # Trae todos los registros
            registros_dict = [
                {
                    "CALIDAD": r.calidad,
                    "PROTEINA_MIN": r.proteina_min,
                    "TVN_MAX": r.tvn_max,
                    "HISTAMINA_MAX": r.histamina_max,
                    "HUMEDAD_MAX": r.humedad_max,
                    "HUMEDAD_MIN": r.humedad_min,
                    "GRASAS_MAX": r.grasas_max,
                    "CENIZAS_MAX": r.cenizas_max,
                    "CLORUROS_MAX": r.cloruros_max,
                    "ACIDEZ_MAX": r.acidez_max,
                    "PRECIO": r.precio,
                        }
                        for r in results
                    ]
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()

    def get_all_parametro_ruma_asignacion(self):
        """Devuelve todos los registros de la tabla ParametroRuma."""
        session = self.Session()
        try:
            results = session.query(ParametroRumaAsignacion).all()  # Trae todos los registros
            registros_dict = [
                {
                    "CALIDAD": r.calidad,
                    "PROTEINA_MIN": r.proteina_min,
                    "TVN_MAX": r.tvn_max,
                    "HISTAMINA_MAX": r.histamina_max,
                    "HUMEDAD_MAX": r.humedad_max,
                    "HUMEDAD_MIN": r.humedad_min,
                    "GRASAS_MAX": r.grasas_max,
                    "CENIZAS_MAX": r.cenizas_max,
                    "CLORUROS_MAX": r.cloruros_max,
                    "ACIDEZ_MAX": r.acidez_max,
                    "PRECIO": r.precio,
                        }
                        for r in results
                    ]
            df = pd.DataFrame(registros_dict)
            return df
        finally:
            session.close()
    
    def get_last_tonelaje_por_calidad(self):
        """
        Obtiene el último id_batch (basado en la fecha más reciente) y devuelve 
        un DataFrame con todos los registros de la tabla TonelajePorCalidad para ese id_batch.
        """
        session = self.Session()
        try:
            # Obtener el último id_batch ordenado por fecha descendente
            ultimo_id_batch = (
                session.query(TonelajePorCalidad.IdBatch)
                .order_by(TonelajePorCalidad.Feccrea.desc())
                .first()
            )

            if not ultimo_id_batch:
                return pd.DataFrame()  # Retorna un DataFrame vacío si no hay registros

            # Extraer el valor del último id_batch
            ultimo_id_batch = ultimo_id_batch[0]

            # Filtrar registros con el último id_batch
            registros = (
                session.query(TonelajePorCalidad)
                .filter(TonelajePorCalidad.IdBatch == ultimo_id_batch)
                .all()
            )

            # Convertir registros a una lista de diccionarios para Pandas
            registros_dict = [
                {
                    "id": r.Id,
                    "Ton Upgrade": r.TonUpgrade,
                    "CALIDAD": r.Calidad,
                    "Fecha": r.Fecha,
                }
                for r in registros
            ]

            df = pd.DataFrame(registros_dict)
            return df

        finally:
            session.close()

    #PREASIGNACION

    def save_rumas_preasignadas(self, df):
        """
        Guarda cada fila del DataFrame en la base de datos.
        """
        session = self.Session()
        try:
            # Definir feccrea_aux si no está definido
            feccrea_aux = datetime.now()  # O ajusta según sea necesario

            # Convertir cada fila del DataFrame en objetos Ruma
            nuevas_rumas = [
                Ruma(
                    RumaNro=row["Ruma Nro"],
                    Cantidad=row["Cantidad"],
                    UM=row["U.M."],
                    Codigo=row["Código"],
                    DescripcionMaterial=row["Descripción del Material"],  # Corregir aquí si es necesario
                    CentroUbicacion=row["Centro Ubicación"],
                    AlmacenUbicacion=row["Almacen Ubicación"],
                    TipoProduccion=row["Tipo Producción"],
                    CentroProduccion=row["Centro Producción"],
                    CalidadPlanta=row["Calidad Planta"],
                    CierreVta=row["Cierre Vta"],
                    Posicion=row["Posición"],
                    Material=row["Material"],
                    CantPreAsignado=row["Cant. Pre Asignado"],
                    CantTransito=row["Cant. Transito"],
                    CantLote=row["Cant. Lote"],
                    LoteExp=row["Lote Exp."],
                    UbicacionAlmacen=row["Ubicación en almacen"],
                    FechaContabilizacion=row["Fecha de Contabilización"],  # Ajustar formato
                    FechaFabricacion=row["Fecha de Fabricación"],  # Ajustar formato
                    Certificadora=row["Certificadora"],
                    FAnalFcoQco=row["F. Anal  Fco Qco"] if isinstance(row["F. Anal  Fco Qco"], datetime) else None,
                    FAnalMicobiol=row["F. Anal  Micobiol"] if isinstance(row["F. Anal  Micobiol"], datetime) else None,
                    FVAnalFcoQco=row["F.V. Anal  Fco Qco"] if isinstance(row["F.V. Anal  Fco Qco"], datetime) else None,
                    FVAnalMicobiol=row["F.V. Anal  Micobiol"] if isinstance(row["F.V. Anal  Micobiol"], datetime) else None,
                    ProteinaPorc=row["proteina"],
                    TVNmg100g=row["tvn"],
                    Histaminappm=row["histamina"],
                    HumedadPorc=row["humedad"],
                    GrasasPorc=row["grasa"],
                    CenizasPorc=row["cenizas"],
                    ClorurosPorc=row["cloruros"],
                    ArenaPorc=row["Arena             %"],
                    AcidezPorc=row["acidez"],
                    AOppm=row["A/O             ppm"],
                    Cadmio=row["Cadmio"],
                    Hierro=row["Hierro"],
                    Proximal=row["Proximal"],
                    GranulometriaPorc=row["Granulometría %"],
                    DensAparGml=row["Dens.Apar. g/ml"],
                    DensCompGml=row["Dens.Comp.  g/ml"],
                    NroFlujoCm=row["Nro.Flujo cm."],                                  
                    Salmonella=row["Salmonella"],
                    Shiguella=row["Shiguella"],
                    EnterobactUfcG=row["Enterobact ufc/g"],
                    Intervalo=row["Intervalo"],
                    Nro=row["Nro."],
                    ClaseInsp=row["Clase Insp."],
                    TempElevadaC=row["Temp. Elevada °C"],
                    ColorHarina=row["Color Harina"],
                    PresenciaMatExtrana=row["Presencia Mat.Extraña"],
                    CamaroncilloPorc=row["% Camaroncillo"],
                    CamaroncilloCualit=row["% Camaroncillo Cualit"],
                    UpgradeFinal=row["UPGRADE"],
                    feccrea=feccrea_aux,
                    IdBatch=row["idBatch"],
                    Estado="PREASIGNADO",
                    IdEjecucion=1,
                    FechaPreAsignacion=feccrea_aux
                )
                for _, row in df.iterrows()
            ]
            # Agregar nuevos objetos a la sesión y guardar
            session.add_all(nuevas_rumas)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
        
    # DISTRIBUCION STOCK
    def get_parametros_distribucion_stock(self):
        session = self.Session()
        try:
            temporadas = session.query(ParametrosTemporada.Temporada).all()
        
            # Crear una lista de solo las temporadas
            array_temporadas = [temporada[0] for temporada in temporadas]
            return array_temporadas
        finally:
            session.close()



    def get_cierres_by_idbatch(self, id_batch_value):
        """
        Devuelve los registros de la tabla CierreVentasSAP filtrados por idBatch
        """
        session = self.Session()
        try:
            # Filtrar por idBatch y ordenar por un campo, como 'id' o 'fecha_pre_asignacion'.
            registros = (
                session.query(CierreVentasSAP)
                .filter(CierreVentasSAP.IdBatch == id_batch_value)
                .order_by(CierreVentasSAP.Id)  # Cambia `RumaSAP.id` por el campo que prefieras.
                .all()
            )
            # Convertir registros a una lista de diccionarios para Pandas
            registros_dict = [
                {
                    "N° Pedido": r.NroPedido,
                    "Documento": r.Documento,
                    "Cond.Pago": r.CondPago,
                    "Fecha Cierre": r.FechaCierre,
                    "Posición": r.Posicion,
                    "Cdis": r.Cdis,
                    "Comprador": r.Comprador,                    
                    "P.Unitario": r.PrecioUnitario,
                    "Producto": r.Producto,
                    "TM Total": r.TMTotal,
                    "Atendido": r.Atendido,
                    "Pendiente": r.Pendiente,
                    "Valor FOB": r.ValorFOB,
                    "Temporada": r.Temporada,
                    "Välido de": r.ValidoDe,
                    "Válido a": r.ValidoA,
                    "Fecha Programada": r.FechaProgramada,
                    "Destino": "r.Destino",
                    "comentarios 2": r.Observacion
                }
                for r in registros
            ]
            
            # Crear un DataFrame de Pandas
            df = pd.DataFrame(registros_dict)
            
            return df
        finally:
            session.close()    
        #def get_parametros_distribucion_stock(self):


        def close(self):
            """Cierra el motor de la base de datos."""
            self.engine.dispose()

    def save_tonelaje_por_calidad_contrato(self, df):
        """
        Guarda cada fila del DataFrame en la base de datos.
        """
        session = self.Session()
        try:
            # Definir feccrea_aux si no está definido
            feccrea_aux = datetime.now()  # O ajusta según sea necesario

            # Convertir cada fila del DataFrame en objetos Ruma
            nuevo_row = [
                TonelajePorCalidadPorContrato(
                    NroPedido=row["Cierre"],
                    Calidad=row["Calidad"],       
                    Tipo=row["Tipo"],
                    PrecioUnitario=row["Precio_unitario"],
                    TM=row["TM"],
                    TM_Asignar=None,
                    FechaEjecucion=row["Fecha_ejecucion"],
                    FechaStock=row["Fecha_stock"],
                    idBatch=row["idBatch"],
                    feccrea=feccrea_aux,
                    Material=row["Producto"],
                    Destino=row["Destino"],
                    Comprador=row["Comprador"],
                    EstadoId=2
                )
                for _, row in df.iterrows()
            ]
            # Agregar nuevos objetos a la sesión y guardar
            session.add_all(nuevo_row)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


    # ASIGNACION
    def get_first_parametro_asignacion(self):
        """
        Devuelve el primer registro de la tabla ParametrosAsignacion.
        """
        session = self.Session()
        try:
            first_record = session.query(ParametrosAsignacion).first()
            
            return first_record
        finally:
            session.close()

    def get_filtered_and_sorted_ruma(self):
        """
        Devuelve los registros de la tabla RumaSAP filtrados por idBatch
        """
        session = self.Session()
        try:
            ##si cambia van a ser mediante ids
            # Obtener el último id_batch ordenado por fecha descendente
            ultimo_id_batch = (
                session.query(Ruma.IdBatch)
                .order_by(Ruma.feccrea.desc())
                .first()
            )

            if not ultimo_id_batch:
                return pd.DataFrame()  # Retorna un DataFrame vacío si no hay registros

            # # Extraer el valor del último id_batch
            ultimo_id_batch = ultimo_id_batch[0]
            
            # Filtrar por idBatch y ordenar por un campo, como 'id' o 'fecha_pre_asignacion'.
            registros = (
                session.query(Ruma)
                .filter((Ruma.Estado == "PREASIGNADO") & (Ruma.IdBatch == ultimo_id_batch))
                .order_by(Ruma.Id)  # Cambia `RumaSAP.id` por el campo que prefieras.
                .all()
            )
             # Convertir registros a una lista de diccionarios para Pandas
            registros_dict = [
                {
                    "id":r.Id,
                    "Ruma Nro": r.RumaNro,
                    "Cantidad": r.Cantidad,
                    "U.M.": r.UM,
                    "Código": r.Codigo,
                    "Descripción del Material": r.DescripcionMaterial,
                    "Centro Ubicación": r.CentroUbicacion,
                    "Almacen Ubicación": r.AlmacenUbicacion,
                    "UpgradeFinal": r.UpgradeFinal,
                    "Tipo Producción": r.TipoProduccion,
                    "Centro Producción": r.CentroProduccion,
                    "Calidad Planta": r.CalidadPlanta,
                    "Cierre Vta": r.CierreVta,
                    "Posición": r.Posicion,
                    "Material": r.Material,
                    "Cant. Pre Asignado": r.CantPreAsignado,
                    "Cant. Transito": r.CantTransito,
                    "Cant. Lote": r.CantLote,
                    "Lote Exp.": r.LoteExp,
                    "Ubicación en almacen": r.UbicacionAlmacen,
                    "Fecha de Contabilización": r.FechaContabilizacion,
                    "Fecha de Fabricación": r.FechaFabricacion,
                    "Certificadora": r.Certificadora,
                    "F. Anal  Fco Qco": r.FAnalFcoQco,
                    "F. Anal  Micobiol": r.FAnalMicobiol,
                    "F.V. Anal  Fco Qco": r.FVAnalFcoQco,
                    "F.V. Anal  Micobiol": r.FVAnalMicobiol,
                    "proteina": r.ProteinaPorc,
                    "tvn": r.TVNmg100g,
                    "histamina": r.Histaminappm,
                    "humedad": r.HumedadPorc,
                    "grasa": r.GrasasPorc,
                    "cenizas": r.CenizasPorc,
                    "cloruros": r.ClorurosPorc,
                    "Arena             %": r.ArenaPorc,
                    "acidez": r.AcidezPorc,
                    "A/O             ppm": r.AOppm,
                    "Cadmio": r.Cadmio,
                    "Hierro": r.Hierro,
                    "Proximal": r.Proximal,
                    "Granulometría %": r.GranulometriaPorc,
                    "Dens.Apar. g/ml": r.DensAparGml,
                    "Dens.Comp.  g/ml": r.DensCompGml,
                    "Nro.Flujo cm.": r.NroFlujoCm,
                    "Salmonella": r.Salmonella,
                    "Shiguella": r.Shiguella,
                    "Enterobact ufc/g": r.EnterobactUfcG,
                    "Intervalo": r.Intervalo,
                    "Nro.": r.Nro,
                    "Clase Insp.": r.ClaseInsp,
                    "Temp. Elevada °C": r.TempElevadaC,
                    "Color Harina": r.ColorHarina,
                    "Presencia Mat.Extraña": r.PresenciaMatExtrana,
                    "% Camaroncillo": r.CamaroncilloPorc,
                    "% Camaroncillo Cualit": r.CamaroncilloCualit,"IdBatch":r.IdBatch
                }
                for r in registros
            ]
    
            # Crear un DataFrame de Pandas
            df = pd.DataFrame(registros_dict)
            
            return df
        finally:
            session.close()       

    def get_tonelaje_por_calidad_por_contrato(self):

        """
        Devuelve los registros de la tabla RumaSAP filtrados por idBatch
        y ordenados por un campo específico.input pre-asignaciones
        """
        session = self.Session()
        try:
            # Obtener el último id_batch ordenado por fecha descendente
            ultimo_id_batch = (
                session.query(TonelajePorCalidadPorContrato.idBatch)
                .order_by(TonelajePorCalidadPorContrato.feccrea.desc())
                .first()
            )

            if not ultimo_id_batch:
                return pd.DataFrame()  # Retorna un DataFrame vacío si no hay registros

            # Extraer el valor del último id_batch
            ultimo_id_batch = ultimo_id_batch[0]

            # Filtrar por idBatch y ordenar por un campo, como 'id' o 'fecha_pre_asignacion'.
            registros = (
                session.query(TonelajePorCalidadPorContrato)
                .filter(TonelajePorCalidadPorContrato.EstadoId == 7)
                .order_by(TonelajePorCalidadPorContrato.Id)  # Cambia `RumaSAP.id` por el campo que prefieras.
                .all()
            )
             # Convertir registros a una lista de diccionarios para Pandas
            registros_dict = [
                    {
                        "pedido": r.NroPedido,
                        "Material": r.Calidad,
                        "Tipo ": r.Tipo,
                        "PrecioUnitario": r.PrecioUnitario,
                        "TM Total  ": r.Tonelaje,
                        "TM_objetivo": r.TM_Asignar,
                        "destino": r.Destino            
                    }
                    for r in registros
                ]
            
            # Crear un DataFrame de Pandas
            df = pd.DataFrame(registros_dict)
            print('....'*15)
            print(df.head())
            return df
        finally:
            session.close()    

    def update_ruma_asignada(self,id:int, RumaNro: str, nroPedido: str):
        
        
        try:
            # Buscar la fila específica por ID
            session = self.Session()
            id = int(id)
            ruma = session.query(Ruma).filter(Ruma.Id == id).first()
            
            if ruma:
                
                # Actualizar el valor de Cantidad
                # ruma.Cantidad = nueva_cantidad
                ruma.Estado='asignado'
                ruma.NroPedido=nroPedido
                # Confirmar los cambios
                session.commit()
                print(f"Estado se actualizo para el ID {RumaNro}-{nroPedido}")
            else:
                print(f"No se encontró ninguna fila con ID {RumaNro}")
        except Exception as e:
            session.rollback()  # Revertir los cambios en caso de error
            print(f"Error al actualizar {RumaNro}-{nroPedido} : {e} ")

    def save_rumaspromedio(self,df):
        """
        Guarda cada fila del DataFrame en la base de datos.
        """
        session = self.Session()
        try:
            # Definir feccrea_aux si no está definido
            feccrea_aux = datetime.now()  # O ajusta según sea necesario

            # Convertir cada fila del DataFrame en objetos Ruma
            nuevas_rumas = [
                Ruma(
                    RumaNro="PROMEDIO",
                    Cantidad=0,
                    UM="SAC",
                    Codigo=0,
                    DescripcionMaterial=".",  # Corregir aquí si es necesario
                    CentroUbicacion=".",
                    AlmacenUbicacion=".",
                    TipoProduccion=".",
                    CentroProduccion=".",
                    CalidadPlanta=".",
                    CierreVta=".",
                    Posicion=0,
                    Material=0,
                    CantPreAsignado=0,
                    CantTransito=0,
                    CantLote=0,
                    LoteExp=".",
                    UbicacionAlmacen=".",
                    FechaContabilizacion=".",  # Ajustar formato
                    FechaFabricacion=".",  # Ajustar formato
                    Certificadora=".",
                    FAnalFcoQco=None,
                    FAnalMicobiol=None,
                    FVAnalFcoQco=None,
                    FVAnalMicobiol=None,
                    ProteinaPorc = row["proteina"] if row["proteina"] is not None else 0,
                    TVNmg100g = row["tvn"] if row["tvn"] is not None else 0,
                    Histaminappm = row["histamina"] if row["histamina"] is not None else 0,
                    HumedadPorc = row["humedad"] if row["humedad"] is not None else 0,
                    GrasasPorc = row["grasa"] if row["grasa"] is not None else 0,
                    CenizasPorc = row["cenizas"] if row["cenizas"] is not None else 0,
                    ClorurosPorc = row["cloruros"] if row["cloruros"] is not None else 0,
                    ArenaPorc=0,
                    AcidezPorc=row["acidez"] if row["acidez"] is not None else 0,
                    AOppm=0,
                    Cadmio=0,
                    Hierro=0,
                    Proximal=0,
                    GranulometriaPorc=0,
                    DensAparGml=0,
                    DensCompGml=0,
                    NroFlujoCm=0,                                  
                    Salmonella="",
                    Shiguella="",
                    EnterobactUfcG="",
                    Intervalo=0,
                    Nro="",
                    ClaseInsp=0,
                    TempElevadaC=0,
                    ColorHarina="",
                    PresenciaMatExtrana="",
                    CamaroncilloPorc=0,
                    CamaroncilloCualit="",
                    BHTNIR=0,
                    HierroNIR=0,
                    UpgradeFinal="",
                    feccrea=feccrea_aux,
                    IdBatch=row["IdBatch"],
                    Estado="PROMEDIO",
                    IdEjecucion=1,
                    FechaPreAsignacion=feccrea_aux,
                    Calidad="",
                    NroPedido=row['Pedido']
                )
                for _, row in df.iterrows()
            ]
            # Agregar nuevos objetos a la sesión y guardar
            session.add_all(nuevas_rumas)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()