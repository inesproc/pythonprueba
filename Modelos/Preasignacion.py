# archivo preasignacion.py
from decimal import Decimal
import pandas as pd
import numpy as np
from Database.DatabaseManager import DatabaseManager
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pulp import *
import unidecode
import re
from datetime import datetime

## Paquetes para lectura de datos del SQL
import urllib
import pyodbc
import sqlalchemy
import logging

# Configuración del logger
logging.basicConfig(level=logging.INFO)

# Función para calcular el promedio ponderado para múltiples columnas
def promedio_ponderado(grupo):
    result = {}
    total_tm = grupo['TM'].sum()
    parametros_ponderado = ['proteina','tvn', 'histamina', 'humedad', 'grasa', 'cenizas', 'cloruros', 'acidez','PRECIO']
    
    for column in parametros_ponderado:
        weighted_average = (grupo[column] * grupo['TM']).sum() / total_tm
        result[column] = weighted_average
        
    result["TM"] = total_tm
    return pd.Series(result)

def dar_formato_al_stock_de_rumas(rumas_disponibles, column, param_prom_df):
    '''
    Funcion para generar el formato de los datos de las rumas para que entren a la funcion de optimizacion
    
    Args:
        rumas_disponibles (df): dataframe que se utilizaran en la funcion de optimizacion
        column (str): variable que indica la calidad en la que estaremos buscando la demanda
        param_prom_df (df): dataframe que tiene los parametros para dar formato

    Returns:
        (parametros): te retorna todos los parametros que ingresaran al modelo
    '''

    # Datos de la oferta disponible
    

    oferta_tn = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['KG']]))   
    oferta_proteina = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['proteina']]))
    oferta_tvn = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['tvn']]))
    oferta_histamina = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['histamina']]))
    oferta_ceniza = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['cenizas']]))
    oferta_grasa = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['grasa']]))
    oferta_humedad = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['humedad']]))
    oferta_cloruro = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['cloruros']]))
    oferta_acidez = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['acidez']]))
    
    # Datos de precios y costos
    precios = dict(zip(rumas_disponibles['indice'], [Decimal(x) for x in rumas_disponibles['PRECIO']]))

    # Datos de la demanda
    demanda_proteina_min = Decimal(param_prom_df['PROTEINA_MIN'][param_prom_df["CALIDAD"]==column].iloc[0])
    demanda_proteina_max = Decimal(10**5)
    
    demanda_tvn_min = Decimal(0)
    demanda_tvn_max = Decimal(param_prom_df['TVN_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_histamina_min = Decimal(0)
    demanda_histamina_max = Decimal(param_prom_df['HISTAMINA_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_ceniza_min = Decimal(0)
    demanda_ceniza_max = Decimal(param_prom_df['CENIZAS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_grasa_min = Decimal(0)
    demanda_grasa_max = Decimal(param_prom_df['GRASAS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_humedad_min = Decimal(param_prom_df['HUMEDAD_MIN'][param_prom_df["CALIDAD"]==column].iloc[0])
    demanda_humedad_max = Decimal(param_prom_df['HUMEDAD_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_cloruro_min = Decimal(0)
    demanda_cloruro_max = Decimal(param_prom_df['CLORUROS_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    demanda_acidez_min = Decimal(0)
    demanda_acidez_max = Decimal(param_prom_df['ACIDEZ_MAX'][param_prom_df["CALIDAD"]==column].iloc[0])

    indice_oferta = list(rumas_disponibles['indice'])
    
    return (oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn,
            demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
            demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max,
            precios, 
            indice_oferta)


def algoritmo_preasignaciones(oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn,
    demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
    demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max,
    precios, #costos,
    demanda_maxima, demanda_minima, precio_calidad_objetivo,
    indice_oferta):
                
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
    demanda_maxima = float(demanda_maxima)
    demanda_minima = float(demanda_minima)
    precio_calidad_objetivo = float(precio_calidad_objetivo)

    # Modelo
    opt_model = LpProblem('Model', LpMaximize)

    # Variables de decision
    x_vars = {(i): LpVariable(cat=LpContinuous if i == 999999 else LpInteger, lowBound=0, name='x_{0}'.format(i)) for i in indice_oferta}
    y_vars = {(i): LpVariable(cat = LpBinary, name='y_{0}'.format(i)) for i in indice_oferta}
    
    # Objetivo
    opt_model += lpSum(x_vars[i]* kilos_minimos_utilizar for i in indice_oferta)

    # Mínimo y máximo número de rumas que se utilizarán
    opt_model += (lpSum(y_vars[i] for i in indice_oferta) >= 2)
    
    for i in indice_oferta:
        if i != 999999:
            opt_model += x_vars[i] * kilos_minimos_utilizar <= y_vars[i] * oferta_tn[i]
        else:
            y_vars[i] = 1
            opt_model += x_vars[i] * kilos_minimos_utilizar == y_vars[i] * oferta_tn[i]

    opt_model += lpSum(sum(x_vars[i] * kilos_minimos_utilizar for i in indice_oferta)) <= demanda_maxima
    opt_model += lpSum(sum(x_vars[i] * kilos_minimos_utilizar for i in indice_oferta)) >= demanda_minima

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

    # Maximizacion
    print("&&&&&"*15)
    opt_model.solve()
    print(opt_model)
    print("&&&__&&&"*15)
    if LpStatus[opt_model.status] != 'Infeasible':        
        optimization_solution = True
        optimalSolution = pd.DataFrame(columns=['indice', 'seleccionar'])

        for var in opt_model.variables():
            if var.name[0] == 'x':
                opt_df = pd.DataFrame({'indice': [str(var.name[(var.name.find('_') + 1):])], 'seleccionar': [var.varValue * kilos_minimos_utilizar]})
                optimalSolution = pd.concat([optimalSolution, opt_df])
    else:
        optimization_solution = False
        optimalSolution = pd.DataFrame()
    return optimization_solution, optimalSolution


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

def ejecutar_preasignacion(idBatch):
    # Instancia de DatabaseManager
    db_manager = DatabaseManager()
    
    # Obtener registros filtrados y ordenados
    preasignaciones_reglas_df=db_manager.get_filtered_and_sorted_ruma_sap(idBatch)
    # print(preasignaciones_reglas_df.head())
    parametrosPreAsignacion=db_manager.get_first_parametro_pre_asignacion()
    # print(parametrosPreAsignacion)
    flagDowngrade=parametrosPreAsignacion.PermitirDowngrade
    upgrade_maximo=parametrosPreAsignacion.PorcentajeUpgrade
    # param_ruma_usuario=db_manager.get_all_parametro_ruma()
    param_prom_usuario=db_manager.get_all_parametro_promedio_preasignacion()
    
    qualities_orig=db_manager.get_all_ord_calidad()
    #qualities_extra=db_manager.get_all_ord_calidad_extra()
    # Todo el código existente va aquí
    # Por ejemplo:
    
    preasignaciones_reglas_df['indice'] = range(1, len(preasignaciones_reglas_df) + 1)
    preasignaciones_reglas_df['f_fab_ultima'] = preasignaciones_reglas_df.apply(process_date_info, axis=1)
    print(preasignaciones_reglas_df['CALIDAD'])
    if not flagDowngrade:
        print("No hay Downgrade")
        # Unir las tablas con la calidad original y la propuesta
        available_stock = pd.merge(preasignaciones_reglas_df, qualities_orig, left_on="CALIDAD", right_on="CALIDAD", how="left")
        available_stock = available_stock.rename(columns={"ORDEN":"OrdenOriginal"})
        available_stock = pd.merge(available_stock, qualities_orig, left_on="UPGRADE", right_on="CALIDAD", how="left")
        available_stock = available_stock.rename(columns={"ORDEN":"OrdenPropuesto", "CALIDAD_x":"CALIDAD"})
        available_stock = available_stock.drop(columns={"CALIDAD_y"})
        
        # Condicional para actualizar la columna UPGRADE
        available_stock['UPGRADE'] = np.where(
            available_stock['OrdenOriginal'] < available_stock['OrdenPropuesto'],
            available_stock['CALIDAD'], 
            available_stock['UPGRADE']
        )
        
        # Eliminar las últimas dos columnas que fueron creadas
        available_stock = available_stock.iloc[:, :-2]
        
        # Unir nuevamente con el quality_order para obtener la columna OrdenUpgrade
        available_stock = pd.merge(available_stock, qualities_orig, left_on="UPGRADE", right_on="CALIDAD", how="left")
        available_stock.columns.values[-1] = "OrdenUpgrade"
        available_stock = available_stock.rename(columns={"CALIDAD_x":"CALIDAD"})
        available_stock = available_stock.drop(columns={"CALIDAD_y"})
        
        # Unir nuevamente con el quality_order para obtener la columna OrdenOriginal
        available_stock = pd.merge(available_stock, qualities_orig, left_on="CALIDAD", right_on="CALIDAD", how="left")
        available_stock.columns.values[-1] = "OrdenOriginal"
        
        # Asignar OrdenDowngrade y Downgrade
        available_stock['Downgrade'] = available_stock['CALIDAD']
        available_stock['OrdenDowngrade'] = available_stock['OrdenOriginal']    
    else:
        print("Downgrade!")
        # Unir las tablas con la calidad original y la propuesta
        available_stock = pd.merge(preasignaciones_reglas_df, qualities_orig, left_on="CALIDAD", right_on="CALIDAD", how="left")
        available_stock = available_stock.rename(columns={"ORDEN":"OrdenOriginal"})
        available_stock = pd.merge(available_stock, qualities_orig, left_on="UPGRADE", right_on="CALIDAD", how="left")
        available_stock = available_stock.rename(columns={"ORDEN":"OrdenUpgrade", "CALIDAD_x":"CALIDAD"})
        available_stock = available_stock.drop(columns={"CALIDAD_y"})
        
        # Actualizar OrdenUpgrade usando np.where
        available_stock['OrdenUpgrade'] = np.where(
            available_stock['OrdenUpgrade'] > available_stock['OrdenOriginal'],
            available_stock['OrdenOriginal'] + 1,
            available_stock['OrdenUpgrade']
        )
        
        # Unir con quality_order usando OrdenUpgrade
        upgrade_helper = pd.merge(available_stock, qualities_orig, left_on="OrdenUpgrade", right_on="ORDEN", how="left")
        available_stock['UPGRADE'] = upgrade_helper['CALIDAD_y']
    
    available_stock = pd.merge(available_stock, param_prom_usuario[["CALIDAD", "PRECIO"]], left_on="CALIDAD", right_on=["CALIDAD"])
    #bht_stock_df = available_stock[available_stock["Tipo Producción"]=='C/LODO + BHT']
    #etox_stock_df = available_stock[available_stock["Tipo Producción"]=='C/LODO + ETOXIQ']

    lista_orden_calidades = list(qualities_orig["ORDEN"])

    summary = available_stock.copy()
    df_entrada_original = pd.DataFrame()

    tipo_produccion = list(available_stock["Tipo Producción"].unique())
    df_resultados_preasignaciones_final = pd.DataFrame()
    # available_stock.to_excel('C:/Users/admin/Desktop/asignaciones/Documentacion Asignaciones/3. Carpeta/respuestasdos/available_stock.xlsx', sheet_name='MiHoja', index=False)
    
    for tipo_prod in tipo_produccion:
        print("#############################################################################################")
        print("#####################  Asignamos para la harina de tipo {} ######################".format(tipo_prod))
        print("#############################################################################################")
        
        stock_original = available_stock[available_stock["Tipo Producción"]==tipo_prod]
        
        print("Tenemos {} rumas disponibles para asignar".format(stock_original.shape[0]))
        
        if stock_original.shape[0]>1:
            print("Es posible asignar!")
            
        else:
            print("No es posible asignar!")
            df_resultados_preasignaciones = pd.DataFrame()
            
        ## Iteramos por cada calidad
        for calidad in lista_orden_calidades:
            print("#############################################################")
            print("Buscamos preasignar hacia la calidad: {}".format(calidad))
            
            # Entrada de preasignaciones
            df_entrada_preasignaciones = pd.DataFrame()
            
            # Si queremos hacer upgrade entonces solo apuntamos a subir de calidad
                
            # El upgrade se dara solo si la Calidad Original es igual o mejor que la que buscamos
            # El upgrade se dara hacia la calidad igual o mejor a la buscada en OrdenUpgrade
            evaluate_upgrade = stock_original[(stock_original['OrdenUpgrade'] <= calidad) & (stock_original['OrdenOriginal'] >= calidad)]
            etox_current_TM = evaluate_upgrade.groupby('OrdenOriginal').agg({'Cantidad': 'sum'}).reset_index()
            etox_current_TM.columns = ['OrdenOriginal', 'Current_TM']
            quality_tons = etox_current_TM[etox_current_TM['OrdenOriginal'] == calidad][['Current_TM']]
            evaluate_upgrade["Iteracion"] = calidad
            
            # La calidad por preasignar
            calidad_preasignaciones = qualities_orig["CALIDAD"][qualities_orig["ORDEN"]==calidad].iloc[0]
            print(calidad_preasignaciones)
            
            # Dataframe para PreAsignar
            df_preasignaciones = evaluate_upgrade.copy()
            print("Total de rumas {}".format(df_preasignaciones.shape[0]))
            
            ## Las calidades originales no se deben tocar, es decir, se quedan con la calidad original 
            df_preasignaciones_original = df_preasignaciones[(df_preasignaciones["OrdenOriginal"] == calidad) & (df_preasignaciones["OrdenUpgrade"] == calidad)]
            print("Total de rumas de la calidad original {}".format((df_preasignaciones_original.shape[0])))   
            
            if (df_preasignaciones_original.shape[0]>0):
                # Agrupar por 'calidad_entrada' y calcular el promedio ponderado
                df_entrada_original = df_preasignaciones_original.groupby('CALIDAD').apply(promedio_ponderado).reset_index()
                
                df_entrada_original["KG"] = df_entrada_original["TM"].sum() * 1000
                df_entrada_original["indice"] = 999999
                df_entrada_original["Iteracion"] = calidad
                
                df_entrada_preasignaciones = pd.concat([df_entrada_preasignaciones, df_entrada_original])
            
            ## Las calidades por debajo de la buscada son las que pueden subir por el modelo
            df_preasignaciones_upgrade = df_preasignaciones[~(df_preasignaciones["OrdenOriginal"] == calidad) & (df_preasignaciones["OrdenUpgrade"] <= calidad)]
            print("Total de rumas de una calidad inferior {}".format(df_preasignaciones_upgrade.shape[0]))
            # Confirmamos si existen datos para Upgrade
            if (df_preasignaciones_upgrade.shape[0]>0):
                df_entrada_upgrade = df_preasignaciones_upgrade[['CALIDAD', 'proteina', 'tvn', 'histamina', 'humedad', 'grasa','cenizas', 'cloruros', 'acidez', 'KG', 'indice','PRECIO']]
                df_entrada_upgrade["Iteracion"] = calidad
                df_entrada_preasignaciones = pd.concat([df_entrada_preasignaciones, df_entrada_upgrade]) 
                
            if (df_entrada_preasignaciones.shape[0]>= 2):
                
                ## Obtenemos los diccionarios necesarios para el algoritmo de optimizacion
                (oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn,
                demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
                demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max,
                precios, indice_oferta) = dar_formato_al_stock_de_rumas(df_entrada_preasignaciones, calidad_preasignaciones, param_prom_usuario)
                #todo aqui sucede el error porque se valida df_entrada_preasignaciones, si se usa df_entrada_original
                if (df_entrada_preasignaciones.shape[0] > 0):
                    ## El limite minimo deberia ser lo que suman los kilos de la calidad original
                    if df_entrada_original.empty:
                        logging.error("El DataFrame 'df_entrada_original' está vacío.")        
                        raise ValueError("El DataFrame 'df_entrada_original' está vacío.")

                    demanda_minima = Decimal(df_entrada_original["KG"].iloc[0])
                    ## El limite maximo a llegar serian las toneladas originales sumada al upgrade permitido
                    # demanda_maxima = Decimal(df_entrada_original["KG"].iloc[0]) * (1 + upgrade_maximo)
                    demanda_maxima = Decimal(df_entrada_original["KG"].iloc[0]) * (1 + Decimal(upgrade_maximo))

                ## El precio objetivo a obtener en esta iteracion
                #precio_calidad_objetivo = Decimal(param_prom_usuario["PRECIO"][param_prom_usuario["CALIDAD"] == calidad_preasignaciones].iloc[0])
                precio_calidad_objetivo = Decimal(float(param_prom_usuario["PRECIO"][param_prom_usuario["CALIDAD"] == calidad_preasignaciones].iloc[0]))
                
                ## Aplicamos el algoritmo de preasignaciones para Upgrade
                (optimization_solution, optimalSolution) = algoritmo_preasignaciones(oferta_proteina, oferta_tvn, oferta_histamina, oferta_ceniza, oferta_grasa, oferta_humedad, oferta_cloruro, oferta_acidez, oferta_tn,
                demanda_proteina_min, demanda_proteina_max, demanda_tvn_min, demanda_tvn_max, demanda_histamina_min, demanda_histamina_max, demanda_ceniza_min, demanda_ceniza_max,
                demanda_grasa_min, demanda_grasa_max, demanda_humedad_min, demanda_humedad_max, demanda_cloruro_min, demanda_cloruro_max, demanda_acidez_min, demanda_acidez_max,
                precios,demanda_maxima, demanda_minima, precio_calidad_objetivo,indice_oferta)
                
                print(df_entrada_preasignaciones.shape[0])

                print("Demanda minima: {}, Demanda maxima: {}".format(demanda_minima, demanda_maxima))
                print(optimization_solution)
                
                ## Si encontramos solucion debemos extraer las rumas escogidas para homogenizado de las rumas disponibles
                if optimization_solution:

                    optimalSolution = optimalSolution[optimalSolution['seleccionar'] > 0].reset_index(drop = True)
                    optimalSolution['indice'] = optimalSolution['indice'].astype(int)

                    seleccion = evaluate_upgrade[evaluate_upgrade['indice'].isin(optimalSolution['indice'])].reset_index(drop = True)

                    # Eliminar la columna de kilos iniciales y reemplazarlos con los kilos que se utilizarán
                    seleccion = seleccion.drop(columns=['KG'])
                    seleccion = pd.merge(seleccion, optimalSolution.rename(columns={'seleccionar':'KG'}), how='left', on='indice')

                    valor_inicial = sum(seleccion['PRECIO'] * seleccion['KG']/1000)
                    valor_proceso_en_planta = param_prom_usuario['PRECIO'][param_prom_usuario["CALIDAD"]==calidad_preasignaciones].iloc[0] * seleccion['KG'].sum()/1000
                    
                    seleccion["Iteracion"] = calidad
                    seleccion["UpgradeFinal"] = calidad_preasignaciones
                    
                    df_resultados_preasignaciones_final = pd.concat([df_resultados_preasignaciones_final, seleccion])
                
                else:
                    seleccion = pd.DataFrame()
                
            else:
                print("No hay suficientes rumas para asignar!")  
    
    rumas_seleccionadas_preasignaciones = df_resultados_preasignaciones_final.copy()
    preasignaciones_output = preasignaciones_reglas_df.copy()
    preasignaciones_output = pd.merge(preasignaciones_output, rumas_seleccionadas_preasignaciones[["Ruma Nro", "UpgradeFinal"]], how="left")
    preasignaciones_output["Descripción del Material"] = preasignaciones_output["CALIDAD"]
    preasignaciones_output = preasignaciones_output.drop(columns={"CALIDAD", "descripcion_material"})
    preasignaciones_output["UpgradeFinal"] = np.where(preasignaciones_output["UpgradeFinal"].isnull(), preasignaciones_output["Descripción del Material"], preasignaciones_output["UpgradeFinal"]) 
    preasignaciones_output['idBatch']=idBatch
    db_manager.save_rumas_preasignadas(preasignaciones_output)
    print(preasignaciones_output)
    print(idBatch)
    # Devuelve los resultados principales como dataframes o valores
    return df_resultados_preasignaciones_final, summary
