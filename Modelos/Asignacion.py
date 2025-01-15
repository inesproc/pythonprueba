import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pulp import *
import unidecode
import re
from datetime import datetime
from Database.DatabaseManager import DatabaseManager
## Paquetes para lectura de datos del SQL
import urllib
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine

def process_date_info(row):
    if pd.isna(row['f_fab']):
        return pd.NaT  # Return NaT if data is missing

    # Extraer el año de 'id_ruma' y formar el año completo
    year = "20" + row['Ruma Nro'][:5][-2:]

    # Limpiar y seleccionar la parte más reciente de la fecha
    date_parts = [x.strip() for x in str(row['f_fab']).split(',')]
    max_date_part = max(date_parts, key=lambda x: x[:5])

    # Extraer día y mes
    day = max_date_part[:2]
    month = max_date_part[3:5]

    # Formular el string de fecha y convertirlo a fecha
    date_str = f"{year}-{month}-{day}"
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return pd.NaT  # Return NaT in case of formatting error
    
def calcular_fila_fab(asignaciones_reglas):
    # Obtener la fecha de hoy
    fecha_hoy = pd.Timestamp(datetime.now().date())
    
    fecha_hoy = fecha_hoy - pd.Timedelta(days=630)
    
    # Calcular la diferencia de días entre 'f_fab_ultima' y la fecha de hoy
    asignaciones_reglas['FIFO'] = (fecha_hoy - pd.to_datetime(asignaciones_reglas['f_fab_ultima'])).dt.days
    
    return asignaciones_reglas
    
## Funcion para catalogar la calidad de una ruma a partir de sus parametros
def dar_formato_al_stock_de_rumas(rumas_disponibles, column, param_prom_df):
    '''
    Funcion para generar el formato de los datos de las rumas para que entren a la funcion de optimizacion
    
    Args:
        rumas_disponibles (df): dataframe que se utilizaran en la funcion de optimizacion
        column (str): variable que indica la calidad en la que estaremos buscando la demanda
        demanda_cantidad (int): variable que indica la cantidad de kilos que buscamos generar
        param_prom_df (df): dataframe que tiene los parametros para dar formato

    Returns:
        (parametros): te retorna todos los parametros que ingresaran al modelo
    '''

    # Datos de la oferta disponible
    oferta_tn = dict(zip(rumas_disponibles['indice'], rumas_disponibles['KG']))
    oferta_proteina = dict(zip(rumas_disponibles['indice'], rumas_disponibles['proteina']))
    oferta_tvn = dict(zip(rumas_disponibles['indice'], rumas_disponibles['tvn']))
    oferta_histamina = dict(zip(rumas_disponibles['indice'], rumas_disponibles['histamina']))
    oferta_ceniza = dict(zip(rumas_disponibles['indice'], rumas_disponibles['cenizas']))
    oferta_grasa = dict(zip(rumas_disponibles['indice'], rumas_disponibles['grasa']))
    oferta_humedad = dict(zip(rumas_disponibles['indice'], rumas_disponibles['humedad']))
    oferta_cloruro = dict(zip(rumas_disponibles['indice'], rumas_disponibles['cloruros']))
    oferta_acidez = dict(zip(rumas_disponibles['indice'], rumas_disponibles['acidez']))
    oferta_fifo = dict(zip(rumas_disponibles['indice'], rumas_disponibles['FIFO']))

    # Datos de precios y costos
    precios = dict(zip(rumas_disponibles['indice'], rumas_disponibles['PRECIO']))
    # costos = dict(zip(rumas_disponibles['indice'], rumas_disponibles['costo']))
    
    # Datos de la demanda
    # demanda_tn = demanda_cantidad

    # demanda_proteina_min = df_requerimientos.loc['proteina'][df_requerimientos["CALIDAD"]==column]
    demanda_proteina_min = param_prom_df['PROTEINA_MIN'][param_prom_df["CALIDAD"]==column].iloc[0]
    demanda_proteina_max = 10**5

    demanda_tvn_min = 0
    # demanda_tvn_max = df_requerimientos.loc['tvn'][df_requerimientos["CALIDAD"]==column]
    demanda_tvn_max = param_prom_df['TVN_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_histamina_min = 0
    demanda_histamina_max = param_prom_df['HISTAMINA_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_ceniza_min = 0
    demanda_ceniza_max = param_prom_df['CENIZAS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_grasa_min = 0
    demanda_grasa_max = param_prom_df['GRASAS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_humedad_min = param_prom_df['HUMEDAD_MIN'][param_prom_df["CALIDAD"]==column].iloc[0]
    demanda_humedad_max = param_prom_df['HUMEDAD_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_cloruro_min = 0
    demanda_cloruro_max = param_prom_df['CLORUROS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    demanda_acidez_min = 0
    demanda_acidez_max = param_prom_df['ACIDEZ_MAX'][param_prom_df["CALIDAD"]==column].iloc[0]

    indice_oferta = list(rumas_disponibles['indice'])

    return (oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn, oferta_fifo,
            demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
            demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max,
            precios, #costos, 
            indice_oferta)
    
def algoritmo_preasignaciones(oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn, oferta_fifo,
    demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
    demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max, demanda_tn, indice_oferta, precios):
                
    kilos_minimos_utilizar = 2000
    # Cast all inputs to float
    oferta_proteina = {k: float(v) for k, v in oferta_proteina.items()}
    oferta_tvn = {k: float(v) for k, v in oferta_tvn.items()}
    oferta_histamina = {k: float(v) for k, v in oferta_histamina.items()}
    oferta_grasa = {k: float(v) for k, v in oferta_grasa.items()}
    oferta_humedad = {k: float(v) for k, v in oferta_humedad.items()}
    oferta_cloruro = {k: float(v) for k, v in oferta_cloruro.items()}
    oferta_acidez = {k: float(v) for k, v in oferta_acidez.items()}
    oferta_tn = {k: float(v) for k, v in oferta_tn.items()}

    demanda_proteina_min = float(demanda_proteina_min)
    demanda_proteina_max = float(demanda_proteina_max)
    demanda_tvn_min = float(demanda_tvn_min)
    demanda_tvn_max = float(demanda_tvn_max)
    demanda_histamina_min = float(demanda_histamina_min)
    demanda_histamina_max = float(demanda_histamina_max)
    demanda_ceniza_min = float(demanda_ceniza_min)
    demanda_ceniza_max = float(demanda_ceniza_max)
    demanda_grasa_min = float(demanda_grasa_min)
    demanda_grasa_max = float(demanda_grasa_max)
    demanda_humedad_min = float(demanda_humedad_min)
    demanda_humedad_max = float(demanda_humedad_max)
    demanda_cloruro_min = float(demanda_cloruro_min)
    demanda_cloruro_max = float(demanda_cloruro_max)
    demanda_acidez_min = float(demanda_acidez_min)
    demanda_acidez_max = float(demanda_acidez_max)
    precios = {k: float(v) for k, v in precios.items()}
    # Modelo
    opt_model = LpProblem('Model', LpMinimize)
    peso_fifo = 20

    # Variables de decision
    # x: número de toneladas a utilizar para la mezcla
    # y: decisión de utilizar la ruma 
    x_vars  = {(i): LpVariable(cat = LpInteger, lowBound = 0, name='x_{0}'.format(i)) for i in indice_oferta}
    y_vars = {(i): LpVariable(cat = LpBinary, name='y_{0}'.format(i)) for i in indice_oferta}

    # Objetivo
    opt_model += lpSum(precios[i] * x_vars[i]* kilos_minimos_utilizar for i in indice_oferta) - lpSum(oferta_fifo[i]* x_vars[i] * kilos_minimos_utilizar * peso_fifo for i in indice_oferta)

    # Mínimo se deerian usar 2 rumas para que sea una mezcla
    opt_model += (lpSum(y_vars[i] for i in indice_oferta) >= 2)
    
    for i in indice_oferta:
        opt_model += x_vars[i] * kilos_minimos_utilizar <= y_vars[i] * oferta_tn[i]

    opt_model += lpSum(sum(x_vars[i] * kilos_minimos_utilizar for i in indice_oferta)) == demanda_tn

    # Restricciones
    # Proteina
    opt_model += (lpSum((oferta_proteina[i] - demanda_proteina_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_proteina[i] - demanda_proteina_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # tvn
    opt_model += (lpSum((oferta_tvn[i] - demanda_tvn_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_tvn[i] - demanda_tvn_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # histamina
    opt_model += (lpSum((oferta_histamina[i] - demanda_histamina_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_histamina[i] - demanda_histamina_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # Ceniza no se considera por el usuario
    # # ceniza
    # opt_model += (lpSum((oferta_ceniza[i] - demanda_ceniza_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    # opt_model += (lpSum((oferta_ceniza[i] - demanda_ceniza_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # grasa
    opt_model += (lpSum((oferta_grasa[i] - demanda_grasa_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_grasa[i] - demanda_grasa_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # humedad
    opt_model += (lpSum((oferta_humedad[i] - demanda_humedad_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_humedad[i] - demanda_humedad_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # cloruro
    opt_model += (lpSum((oferta_cloruro[i] - demanda_cloruro_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_cloruro[i] - demanda_cloruro_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # acidez
    opt_model += (lpSum((oferta_acidez[i] - demanda_acidez_min) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) >= 0)
    opt_model += (lpSum((oferta_acidez[i] - demanda_acidez_max) * x_vars[i] * kilos_minimos_utilizar for i in indice_oferta) <= 0)

    # Minimizacion
    opt_model.solve()

    # print(opt_model)

    if LpStatus[opt_model.status] != 'Infeasible':
        optimization_solution = True
        optimalSolution = pd.DataFrame(columns=['indice', 'seleccionar'])

        for var in opt_model.variables():
            if var.name[0] == 'x':
                opt_df = pd.DataFrame({'indice': [str(var.name[(var.name.find('_') + 1):])], 'seleccionar': [var.varValue * kilos_minimos_utilizar]})
                # optimalSolution = optimalSolution.append(opt_df, ignore_index=True)
                optimalSolution = pd.concat([optimalSolution, opt_df])
    else:
        optimization_solution = False
        optimalSolution = pd.DataFrame()
    
    return optimization_solution, optimalSolution

def promedio_ponderado(df, columnas_ponderadas):
    # Crear un diccionario para almacenar los promedios ponderados
    promedios = {}
    
    for col in columnas_ponderadas:
        # Calcular el promedio ponderado para cada columna
        weighted_avg = (df[col] * df['Cantidad']).sum() / df['Cantidad'].sum()
        promedios[col] = weighted_avg
    
    return promedios

def ejecutar_asignacion():
    db_manager = DatabaseManager()
    ## Parametros iniciales
    # num_sacos = 0
    parametrosAsignacion=db_manager.get_first_parametro_asignacion()
    toneladas_por_ruma = parametrosAsignacion.ToneladasPorRuma
    sacos_por_ruma = parametrosAsignacion.MinSacosRuma
    # flagDowngrade = False
    # upgrade_maximo = 0.3#PorcentajeUpgrade

    ## Consultas a las bases de datos
    # Parametros Ruma de entrada para clasificar
    param_ruma_usuario=db_manager.get_all_parametro_ruma_asignacion()
    # Parametros Promedio que deberia salir la calidad
    param_prom_usuario=db_manager.get_all_parametro_promedio_asignacion()
    
    # Orden de las calidades de harina
    qualities_orig=db_manager.get_all_ord_calidad()

    # Orden de las calidades de harina extra
    qualities_extra=db_manager.get_all_ord_calidad_extra()
    
    # param_prom_usuario["HUMEDAD_MIN"] = [6,6,6,6,6,6,6,6,6]##????

    # Ejemplo de uso
    # nombre_archivo = 'SEPARACION_OUTPUT_EJEMPLO.xlsx'
    # asignaciones_df = leer_excel_con_upgrade_final(nombre_archivo)
    
    asignaciones_df=db_manager.get_filtered_and_sorted_ruma()
    
    asignaciones_reglas = asignaciones_df.copy()
    asignaciones_reglas = asignaciones_reglas[asignaciones_reglas["Tipo Producción"]!= "C/LODO + BHT"]

    # Rellenar los valores NaN con 0 en varias columnas**done
    # asignaciones_reglas['proteina'].fillna(0, inplace=True)
    # asignaciones_reglas['tvn'].fillna(0, inplace=True)
    # asignaciones_reglas['histamina'].fillna(0, inplace=True)
    # asignaciones_reglas['humedad'].fillna(0, inplace=True)
    # asignaciones_reglas['grasa'].fillna(0, inplace=True)
    # asignaciones_reglas['cenizas'].fillna(0, inplace=True)
    # asignaciones_reglas['cloruros'].fillna(0, inplace=True)
    # asignaciones_reglas['acidez'].fillna(0, inplace=True)

    # Generamos la columna con las toneladas
    asignaciones_reglas['TM'] = (asignaciones_reglas['Cantidad'] * toneladas_por_ruma) / sacos_por_ruma
    asignaciones_reglas['KG'] = asignaciones_reglas['TM'] * 1000

    # Generamos la columna con el tipo de produccion
    asignaciones_reglas['prod'] = np.where(asignaciones_reglas['Tipo Producción'].str.contains('ETOXIQ', na=False),
                                    'ETOXIQ',
                                    np.where(asignaciones_reglas['Tipo Producción'].str.contains('ETOX', na=False),
                                                'ETOXIQ', 'BHT'))

    asignaciones_reglas = asignaciones_reglas.reset_index()
    asignaciones_reglas["indice"] = asignaciones_reglas["index"]+1

    # Tratamiento Datos 5:  Creación de columna “f_fab_ultima”
    # Determinamos la f_fab
    asignaciones_reglas['f_fab'] = np.where(pd.isna(asignaciones_reglas['Fecha de Fabricación']),
                                                    asignaciones_reglas['Fecha de Fabricación'],  # Si es NaN, asignar la misma 'Fecha de Fabricación'
                                                    asignaciones_reglas['Fecha de Contabilización'])  # Si no es NaN, asignar 'Fecha de Contabilización'

    # Establecer 'f_fab_final' con una fecha constante usando pd.to_datetime para asegurar el formato correcto
    asignaciones_reglas['f_fab_final'] = pd.to_datetime("1982-01-13")       

    # Tomar la ultima fecha de fabricacion
    asignaciones_reglas['f_fab_ultima'] = asignaciones_reglas.apply(process_date_info, axis=1)

    # Ejemplo de uso????

    asignaciones_reglas = calcular_fila_fab(asignaciones_reglas)
    
    asignaciones_reglas = pd.merge(asignaciones_reglas, param_prom_usuario[["CALIDAD", "PRECIO"]], left_on="Descripción del Material", right_on=["CALIDAD"])

    
    
    pedidos_df = db_manager.get_tonelaje_por_calidad_por_contrato()
    
    pedidos_reglas = pedidos_df.copy()
    ## Limpieza de la.COLUMNSumnas
    #pedidos_reglas['Etapa'] = pedidos_reglas["\xa0Etapa\xa0"].str.strip()
    # pedidos_reglas['Material'] = pedidos_reglas["Material\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"].str.strip()
    # pedidos_reglas['TM_objetivo'] = pedidos_reglas['Tm\xa0Asignar']
    # pedidos_reglas['destino'] = pedidos_reglas['Destino\xa0\xa0\xa0'].str.strip()
    # pedidos_reglas['pedido'] = pedidos_reglas['N°\xa0de\xa0pedido\xa0\xa0\xa0'].str.strip()

    ## Regla Negocio 1: Etapa igual a 10VT
    # pedidos_reglas = pedidos_reglas[pedidos_reglas['Etapa']=="10VT"]

    ## Tratamiento de Datos 1: La calidad objetivo se estandariza
    
    pedidos_reglas['calidad_objetivo'] = pedidos_reglas['Material'].str.replace(".*DRIED", "", regex=True)   
    pedidos_reglas['calidad_objetivo'] = pedidos_reglas['calidad_objetivo'].str.strip()
    pedidos_reglas["tipo_produccion"] = pedidos_reglas["Material"].apply(lambda x: "BHT" if "BHT" in x else "ETOXIQ")

    # pedidos_reglas = pedidos_reglas[["Etapa", "TM_objetivo", "destino", "calidad_objetivo","pedido", "tipo_produccion"]]
    pedidos_reglas = pedidos_reglas[["TM_objetivo", "destino", "calidad_objetivo","pedido", "tipo_produccion"]]
    pedidos_reglas = pd.merge(pedidos_reglas, qualities_orig, left_on=["calidad_objetivo"], right_on=["CALIDAD"], how="left")
    pedidos_reglas = pedidos_reglas.sort_values(by=["ORDEN", "TM_objetivo"], ascending=[True, True])

    pedidos_asignaciones = pedidos_reglas.copy()

    lista_produccion = list(pedidos_asignaciones["tipo_produccion"].unique())
    lista_pedidos = list(pedidos_asignaciones["pedido"].unique())

    df_resultados_asignaciones = pd.DataFrame()
    rumas_disponibles_asignaciones = asignaciones_reglas.copy()

    rumas_disponibles_asignaciones_etox = asignaciones_reglas[asignaciones_reglas["prod"]=="ETOX"]
    rumas_disponibles_asignaciones_bht = asignaciones_reglas[asignaciones_reglas["prod"]=="BHT"]

    for tipo_prod in lista_produccion:
        print("#############################################################################################")
        print("#####################  Asignamos para la harina de tipo {} ######################".format(tipo_prod))
        print("#############################################################################################")

        if tipo_prod=="BHT":
            rumas_disponibles_asignaciones = rumas_disponibles_asignaciones_bht.copy()
        elif tipo_prod=="ETOX":
            rumas_disponibles_asignaciones = rumas_disponibles_asignaciones_etox.copy()
        
        if rumas_disponibles_asignaciones.shape[0]>1:

            for pedido in lista_pedidos:
                print("######################################################################")
                print("Estamos en el pedido {}".format(pedido))
                
                # Filtramos los datos de este pedido
                pedido_asignar = pedidos_asignaciones[pedidos_asignaciones["pedido"]==pedido]
                
                # Obtenemos el objetivo de este pedido
                ton_objetivo = pedido_asignar["TM_objetivo"].iloc[0]
                columns_to_convert = ['TM_objetivo', 'ORDEN']
                # Convertir las columnas específicas a float
                pedido_asignar[columns_to_convert] = pedido_asignar[columns_to_convert].astype(float)
                
                ton_objetivo = float(ton_objetivo*1000)
                calidad_objetivo = pedido_asignar["calidad_objetivo"].iloc[0]

                print("La cantidad objetivo es {}".format(ton_objetivo))
                print("La calidad objetivo es {}".format(calidad_objetivo))                
                # rumas_disponibles_asignaciones = rumas_disponibles_asignaciones.apply(pd.to_numeric, errors='coerce')                
                
                # Filtramos el dataset de entrada a la calidad objetivo
                print(rumas_disponibles_asignaciones["UpgradeFinal"]==calidad_objetivo)
                asignaciones_entrada = rumas_disponibles_asignaciones[rumas_disponibles_asignaciones["UpgradeFinal"]==calidad_objetivo]
                print("La cantidad de rumas de entrada es {}".format(asignaciones_entrada.shape[0]))
                print("La cantidad de kilos disponibles es {}".format(asignaciones_entrada["KG"].sum()))
                
                ## Obtenemos los diccionarios necesarios para el algoritmo de optimizacion
                (oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn, oferta_fifo,
                demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
                demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max, precios, 
                indice_oferta) = dar_formato_al_stock_de_rumas(asignaciones_entrada, calidad_objetivo, param_prom_usuario)
                
                (optimization_solution, optimalSolution) = algoritmo_preasignaciones(oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn, oferta_fifo,
                demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
                demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max, ton_objetivo, indice_oferta, precios)
                        
                print(optimization_solution)
                
                ## Si encontramos solucion debemos extraer las rumas escogidas para homogenizado de las rumas disponibles
                if optimization_solution:

                    optimalSolution = optimalSolution[optimalSolution['seleccionar'] > 0].reset_index(drop = True)
                    optimalSolution['indice'] = optimalSolution['indice'].astype(int)

                    seleccion = asignaciones_entrada[asignaciones_entrada['indice'].isin(optimalSolution['indice'])].reset_index(drop = True)

                    # Eliminar la columna de kilos iniciales y reemplazarlos con los kilos que se utilizarán
                    seleccion = seleccion.drop(columns=['KG'])
                    seleccion = pd.merge(seleccion, optimalSolution.rename(columns={'seleccionar':'KG'}), how='left', on='indice')

                    valor_inicial = sum(seleccion['PRECIO'] * seleccion['KG']/1000)
                    valor_proceso_en_planta = param_prom_usuario['PRECIO'][param_prom_usuario["CALIDAD"]==calidad_objetivo].iloc[0] * seleccion['KG'].sum()/1000
                    
                    # seleccion["Iteracion"] = calidad
                    seleccion["UpgradeFinal"] = calidad_objetivo
                    seleccion["Pedido"] = pedido
                    
                    rumas_disponibles_asignaciones = pd.merge(rumas_disponibles_asignaciones, optimalSolution.rename(columns={'seleccionar':'kilos_utilizados'}), how='left', on='indice').fillna(0)
                    rumas_disponibles_asignaciones['KG'] = rumas_disponibles_asignaciones['KG'] - rumas_disponibles_asignaciones['kilos_utilizados']

                    rumas_disponibles_asignaciones = rumas_disponibles_asignaciones.drop(columns=['kilos_utilizados'])
                    rumas_disponibles_asignaciones = rumas_disponibles_asignaciones[rumas_disponibles_asignaciones['KG'] > 0].reset_index(drop = True)
                    
                    
                    df_resultados_asignaciones = pd.concat([df_resultados_asignaciones, seleccion])
                    print(df_resultados_asignaciones)
                else:
                    seleccion = pd.DataFrame()
               
    print("se completo la asignacion")
    print(df_resultados_asignaciones)
    for index, row in df_resultados_asignaciones.iterrows():
        print(df_resultados_asignaciones.at[index, 'Ruma Nro'])
        db_manager.update_ruma_asignada(df_resultados_asignaciones.at[index, 'id'],df_resultados_asignaciones.at[index, 'Ruma Nro'],df_resultados_asignaciones.at[index, 'Pedido'])
    
    
    # # Columnas para las cuales queremos calcular el promedio ponderado
    # columnas_ponderadas = ['proteina', 'tvn', 'histamina', 'humedad', 'grasa', 'cenizas', 'cloruros', 'acidez']

    # # Agrupar por Pedido y aplicar la función para obtener los promedios ponderados
    # resultado = (
    #     df_resultados_asignaciones
    #     .groupby('Pedido')
    #     .apply(lambda x: pd.Series(promedio_ponderado(x, columnas_ponderadas)))
    #     .reset_index()  # Restablecer el índice para que 'Pedido' sea una columna
    # )
    # if not asignaciones_df.empty:
    #     resultado['IdBatch'] = asignaciones_df['IdBatch'].iloc[0]
    # else:
    #     resultado['IdBatch']= None 
    # db_manager.save_rumaspromedio(resultado)


    
