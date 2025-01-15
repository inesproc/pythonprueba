import pandas as pd
import numpy as np
from Database.DatabaseManager import DatabaseManager
from sqlalchemy import create_engine
import unicodedata
from datetime import datetime
import datetime
from itertools import product
from pulp import *
from Database.DatabaseManager import DatabaseManager
# =============================================================================
# Funciones
# =============================================================================

def limpieza_cierres_calidades(df_cierres, temp_validas):
    
    """Funcion para definir tipos de datos, agregar calidades limpias y tipo HP
    inputs: 
        df_cierres: dataframe
        temp_validas: lista
    outputs: 
        df_cierres: dataframe
    """
    
    df_cierres['Fecha Cierre'] = pd.to_datetime(df_cierres['Fecha Cierre'], dayfirst=True)
    df_cierres['TM Total'] = df_cierres['TM Total'].replace(r'[\$,]', '', regex=True).astype(float)
    df_cierres['Atendido'] = df_cierres['Atendido'].replace(r'[\$,]', '', regex=True).astype(float)
    df_cierres['P.Unitario'] = df_cierres['P.Unitario'].map(float)
    df_cierres['Välido de'] = pd.to_datetime(df_cierres['Välido de'], dayfirst=True)
    df_cierres['Válido a'] = pd.to_datetime(df_cierres['Válido a'], dayfirst=True)
    df_cierres['Fecha Programada'] = pd.to_datetime(df_cierres['Fecha Programada'], dayfirst=True)
    
    # Solo seleccionar la data de la temporada 21-2
    # temp_validas = [temp_validas] #['2021-II','2021-II o Producción']
    df_cierres = df_cierres[df_cierres['Temporada'].isin(temp_validas)].reset_index(drop=True)
    
    # Limpieza especifica para columna de calidad
    df_cierres['Producto'] = [unicodedata.normalize("NFKD", word) for word in df_cierres['Producto']]
    
    # Calidad limpia
    df_cierres['Calidad'] = df_cierres['Producto']
    df_cierres['Calidad'] = np.where(df_cierres['Producto'].str.contains('SUPER'),'SUPER PRIME', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('PRIME')) & (df_cierres['Calidad']!='SUPER PRIME'),'PRIME', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('TAILANDIA')) | (df_cierres['Producto'].str.contains('THAILAND')),'THAILAND', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where(df_cierres['Producto'].str.contains('TAIWAN 67'),'TAIWAN 67', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where(df_cierres['Producto'].str.contains('TAIWAN 66'),'TAIWAN 66', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('STANDARD 67')) | (df_cierres['Producto'].str.contains('STANDAR 67')),'STANDARD 67', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('STANDARD 66')) | (df_cierres['Producto'].str.contains('STANDAR 66')),'STANDARD 66', df_cierres['Calidad'])
    #df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('STANDARD 65/120')) | (df_cierres['Producto'].str.contains('STANDAR 65/120')),'STANDARD 65/120', df_cierres['Calidad'])
    #df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('STANDARD 65')) | (df_cierres['Producto'].str.contains('STANDAR 65')),'STANDARD 65', df_cierres['Calidad'])
    df_cierres['Calidad'] = np.where(df_cierres['Producto'].str.contains(r'\bSTANDARD 65/120\b', case=False),'STANDARD 65/120',
                                     np.where(df_cierres['Producto'].str.contains(r'\bSTANDARD 65\b', case=False),'STANDARD 65',df_cierres['Calidad']))
    df_cierres['Calidad'] = np.where((df_cierres['Producto'].str.contains('STANDARD <65')) | (df_cierres['Producto'].str.contains('STANDAR <65')),'STANDARD <65', df_cierres['Calidad'])
    
    # Indicar BHT y ETOX
    # Separar dependiendo del contenido en 'Producto'
    df_extra = df_cierres['Producto'].apply(
        lambda x: x.split(' CON ') if ' CON ' in x else x.split('C/') if 'C/' in x else [x]
    )

    # Expande el resultado en dos columnas
    df_extra = pd.DataFrame(df_extra.tolist(), columns=[0, 1])

    # Limpieza y asignación
    df_extra[0] = df_extra[0].str.replace('\xa0', ' ', regex=True).str.strip()
    df_extra[1] = df_extra[1].fillna('ETOX')

    # Asignar el valor transformado a la columna 'Tipo'
    df_cierres['Tipo'] = df_extra[1]
    print("222"*20)
    print(df_cierres)
    print(df_extra)
    #Solo ETOX
    #df_cierres['Tipo'] = 'ETOX'
    
    return df_cierres

def tratamiento_cierres_fijos(df_cierres):
    # Separar contratos con calidad fija y abierta, tomar como referencia columna TM TOTAL
    df_calidad_preliminar = df_cierres.copy()
    # TODO: Validar que para la ejecucion 21-2 no genere una diferencia en los resultados
    # Filtrar los casos con precio 0
    # df_calidad_preliminar = df_calidad_preliminar[df_calidad_preliminar['TM Total']>0].reset_index(drop=True)
    df_calidad_preliminar = df_calidad_preliminar[df_calidad_preliminar['P.Unitario']>0].reset_index(drop=True)
    df_calidad_preliminar['Split_calidad'] = df_calidad_preliminar.groupby(['N° Pedido'])['Calidad'].transform('nunique')
    df_calidad_preliminar = df_calidad_preliminar[['N° Pedido','Split_calidad']].drop_duplicates()
    # df_cierres['Split_calidad'] = df_cierres.groupby(['N° Pedido'])['Calidad'].transform('nunique')
    df_cierres = df_cierres.merge(df_calidad_preliminar, how='left', on='N° Pedido')
    df_cierres['Tipo_split'] = np.where(df_cierres['Split_calidad']==1, 'Fija', 'Split')
    
    # Crear codigo unico por cierre-tipo
    df_cierres['Pedido_unico'] = df_cierres['N° Pedido'] + '-' + df_cierres['Tipo']
    
    # Calcular total TM por cada cierre original y atendido
    df_cierres['TM_cierre'] = df_cierres.groupby(['Pedido_unico'])['TM Total'].transform('sum')
    df_cierres['TM_atendido'] = df_cierres.groupby(['Pedido_unico'])['Atendido'].transform('sum')
    df_cierres['diferencia'] = df_cierres['TM_cierre'] - df_cierres['TM_atendido']
    
    return df_cierres

def limpieza_cierres_columnas(df_cierres):
    # Separar contratos con calidad fija y abierta, tomar como referencia columna TM TOTAL
    df_calidad_preliminar = df_cierres.copy()
    # Filtrar los casos con precio 0
    df_calidad_preliminar = df_calidad_preliminar[df_calidad_preliminar['P.Unitario']>0].reset_index(drop=True)
    df_calidad_preliminar['Split_calidad'] = df_calidad_preliminar.groupby(['N° Pedido'])['Calidad'].transform('nunique')
    df_calidad_preliminar = df_calidad_preliminar[['N° Pedido','Split_calidad']].drop_duplicates()
    # df_cierres['Split_calidad'] = df_cierres.groupby(['N° Pedido'])['Calidad'].transform('nunique')
    df_cierres = df_cierres.merge(df_calidad_preliminar, how='left', on='N° Pedido')
    df_cierres['Tipo_split'] = np.where(df_cierres['Split_calidad']==1, 'Fija', 'Split')
    
    # Crear codigo unico por cierre-tipo
    df_cierres['Pedido_unico'] = df_cierres['N° Pedido'] + '-' + df_cierres['Tipo']
    
    # Calcular total TM por cada cierre original y atendido
    df_cierres['TM_cierre'] = df_cierres.groupby(['Pedido_unico'])['TM Total'].transform('sum')
    #df_cierres['TM_atendido'] = df_cierres.groupby(['Pedido_unico'])['Atendido'].transform('sum')
    #df_cierres['diferencia'] = df_cierres['TM_cierre'] - df_cierres['TM_atendido']
    
    ##Agregar lineas de funcion limpieza cierres columnas
    df_cierres = df_cierres[(df_cierres['P.Unitario']>0)].reset_index(drop=True)
    df_cierres['Fecha_tope'] = df_cierres.groupby(['N° Pedido'])['Fecha Programada'].transform('first')
    
    # Sumarizar por calidad
    # cols_group = ['N° Pedido','Fecha_tope','Calidad','Tipo','Split_calidad','Tipo_split','TM_cierre','TM_atendido']
    cols_group = ['N° Pedido','Fecha_tope','Calidad','Tipo','Split_calidad','Tipo_split','TM_cierre']
    df_cierres_cleaned = df_cierres.groupby(cols_group).agg(TM_split_cierre=('TM Total','sum'),
                                                            TM_split_atendido=('Atendido','sum'),
                                                            Precio_unitario=('P.Unitario','min'),
                                                            Comentario=('comentarios 2','first'))
    df_cierres_cleaned.reset_index(inplace=True)

    return df_cierres_cleaned

def dividir_pedidos_generales(df):
    # Lista de calidades consideradas como altas
    calidades_altas = ['SUPER PRIME','PRIME']

    # Reemplaza los valores nulos en 'Comentario' por una cadena vacía para evitar errores
    df['Comentario'] = df['Comentario'].fillna('')
    
    # Filtra las filas con comentarios que contengan "altas" y "bajas"
    df_filtrado = df[df['Comentario'].str.contains('altas') & df['Comentario'].str.contains('bajas')].copy()

    # Lista para almacenar los nuevos pedidos
    nuevos_pedidos = []

    for _, row in df_filtrado.iterrows():
        # Extrae el porcentaje de "altas" y "bajas" del comentario
        match = re.search(r'(\d+)\s*altas\s*(\d+)\s*bajas', row['Comentario'])
        if match:
            porcentaje_altas = int(match.group(1)) / 100
            porcentaje_bajas = int(match.group(2)) / 100
        else:
            # Si el formato no es el esperado, salta la fila
            continue

        # Calcula el TM_cierre para altas y bajas según los porcentajes
        tm_cierre_altas = row['TM_cierre'] * porcentaje_altas
        tm_cierre_bajas = row['TM_cierre'] * porcentaje_bajas

        # Encuentra la posición del segundo guion en el número de pedido
        segundo_guion_pos = row['N° Pedido'].find('-', row['N° Pedido'].find('-') + 1) + 1

        # Modifica el número de pedido para altas o bajas, según la calidad
        if row['Calidad'] in calidades_altas:
            row_alta = row.copy()
            # Cambia el carácter después del segundo guion a 'A' para calidades altas
            row_alta['N° Pedido'] = row_alta['N° Pedido'][:segundo_guion_pos] + 'A' + row_alta['N° Pedido'][segundo_guion_pos + 1:]
            row_alta['TM_cierre'] = tm_cierre_altas
            nuevos_pedidos.append(row_alta)
        else:
            row_baja = row.copy()
            # Cambia el carácter después del segundo guion a 'B' para calidades bajas
            row_baja['N° Pedido'] = row_baja['N° Pedido'][:segundo_guion_pos] + 'B' + row_baja['N° Pedido'][segundo_guion_pos + 1:]
            row_baja['TM_cierre'] = tm_cierre_bajas
            nuevos_pedidos.append(row_baja)

    # Crear un DataFrame con los nuevos pedidos
    df_nuevos_pedidos = pd.DataFrame(nuevos_pedidos)
    
    # Combinar el DataFrame original sin los pedidos con comentarios de altas/bajas con el nuevo DataFrame de pedidos divididos
    df_resultado = pd.concat([df[~df['Comentario'].str.contains('altas') & ~df['Comentario'].str.contains('bajas')], df_nuevos_pedidos], ignore_index=True)
    
    return df_resultado

def dividir_pedidos_fijo_open_altas_bajas(df): 
    # Reemplaza los valores nulos en 'Comentario' por una cadena vacía para evitar errores
    df['Comentario'] = df['Comentario'].fillna('')

    # Función para identificar si el comentario cumple con el patrón 'X altas X bajas'
    def es_patron_altas_bajas(comentario):
        return bool(re.match(r'^\d+\s*altas\s*\d+\s*bajas$', comentario.strip(), re.IGNORECASE))

    # Identificar pedidos que tienen "Fijo" y "Open" o "Fijo" y "X altas X bajas"
    pedidos_con_fijo_y_otros = []
    for pedido in df['N° Pedido'].unique():
        comentarios_pedido = set(df[df['N° Pedido'] == pedido]['Comentario'].unique())
        if 'Fijo' in comentarios_pedido and ('Open' in comentarios_pedido or any(es_patron_altas_bajas(c) for c in comentarios_pedido)):
            pedidos_con_fijo_y_otros.append(pedido)

    # Filtrar solo las filas correspondientes a esos pedidos
    df_filtrado = df[df['N° Pedido'].isin(pedidos_con_fijo_y_otros)].copy()

    # Lista para almacenar los nuevos pedidos
    nuevos_pedidos = []

    # Procesar los pedidos con "Fijo", "Open" y patrones "X altas X bajas"
    for pedido in pedidos_con_fijo_y_otros:
        # Calcular TM_cierre acumulado para cada grupo de comentarios
        tm_cierre_fijo = df_filtrado[(df_filtrado['N° Pedido'] == pedido) & 
                                     (df_filtrado['Comentario'] == 'Fijo')]['TM_split_cierre'].sum()
        tm_cierre_open = df_filtrado[(df_filtrado['N° Pedido'] == pedido) & 
                                     (df_filtrado['Comentario'] == 'Open')]['TM_split_cierre'].sum()
        tm_cierre_altas_bajas = df_filtrado[(df_filtrado['N° Pedido'] == pedido) & 
                                            (df_filtrado['Comentario'].apply(es_patron_altas_bajas))]['TM_split_cierre'].sum()
        
        # Generar nuevas filas modificadas
        for _, row in df_filtrado[df_filtrado['N° Pedido'] == pedido].iterrows():
            nuevo_pedido = row.copy()
            segundo_guion_pos = row['N° Pedido'].find('-', row['N° Pedido'].find('-') + 1) + 1

            if row['Comentario'] == 'Fijo':
                nuevo_pedido['N° Pedido'] = row['N° Pedido'][:segundo_guion_pos] + 'F' + row['N° Pedido'][segundo_guion_pos + 1:]
                nuevo_pedido['TM_cierre'] = tm_cierre_fijo
            elif row['Comentario'] == 'Open':
                nuevo_pedido['N° Pedido'] = row['N° Pedido'][:segundo_guion_pos] + 'O' + row['N° Pedido'][segundo_guion_pos + 1:]
                nuevo_pedido['TM_cierre'] = tm_cierre_open
            elif es_patron_altas_bajas(row['Comentario']):
                nuevo_pedido['N° Pedido'] = row['N° Pedido'][:segundo_guion_pos] + 'M' + row['N° Pedido'][segundo_guion_pos + 1:]
                nuevo_pedido['TM_cierre'] = tm_cierre_altas_bajas

            nuevos_pedidos.append(nuevo_pedido)

    # Crear un DataFrame con los nuevos pedidos
    df_nuevos_pedidos = pd.DataFrame(nuevos_pedidos)

    # Combinar el DataFrame original sin los pedidos duplicados con el nuevo DataFrame de pedidos divididos
    df_resultado = pd.concat([df[~df['N° Pedido'].isin(pedidos_con_fijo_y_otros)], df_nuevos_pedidos], ignore_index=True)
    
    return df_resultado

def tratamiento_cierres_final(df_simulacion):
    
    df_simulacion = df_simulacion.rename(columns={'N° Pedido': 'Pedido'})

    df_simulacion['Tipo_split'] = np.where(df_simulacion['Comentario']=='Fijo', 'Fija',df_simulacion['Tipo_split'])
    df_simulacion['Comentario'] = np.where(df_simulacion['Tipo_split']=='Fija', 'Fijo',df_simulacion['Comentario'])
    df_simulacion['Porcentaje'] = np.where(df_simulacion['Comentario']=='Fijo', df_simulacion['TM_split_cierre']/df_simulacion['TM_cierre'], np.nan)
    df_simulacion['TM_fijo'] = np.where(df_simulacion['Comentario']=='Fijo', df_simulacion['TM_split_cierre'], np.nan)

    # Calcular total TM por cada cierre original y atendido
    df_simulacion['TM_atendido'] = df_simulacion.groupby(['Pedido','Tipo'])['TM_split_atendido'].transform('sum')
    # ##Agregar variables de atendido
    # cols_group = ['Pedido','Tipo']
    # df_simulacion = df_simulacion.groupby(cols_group).agg(TM_atendido=('TM_split_atendido','sum'))
    # df_simulacion.reset_index(inplace=True)

    df_simulacion['TM_cierre_total'] = df_simulacion['TM_cierre']-df_simulacion['TM_atendido']
    #df_simulacion['TM_cierre_total'] = np.where(df_simulacion['Tipo_split']=='Fija',df_simulacion['TM_fijo'],df_simulacion['TM_cierre'])
    # Validar tipo de dato
    # df_simulacion['Fecha Cierre'] = pd.to_datetime(df_simulacion['Fecha Cierre'])
    df_simulacion['Fecha_tope'] = pd.to_datetime(df_simulacion['Fecha_tope'], dayfirst=True)
    df_simulacion['Porcentaje'] = df_simulacion['Porcentaje'].map(float)
    
    # Calcular valor inicial
    df_simulacion['Valor_calidad_inicial'] = df_simulacion['TM_split_cierre']*df_simulacion['Precio_unitario']
    df_simulacion['Valor_inicial'] = df_simulacion.groupby(['Pedido'])['Valor_calidad_inicial'].transform('sum')
    df_simulacion['Valor_inicial_TM'] = df_simulacion['Valor_inicial']/df_simulacion['TM_cierre']
    
    # Tonelaje restante luego de eliminar el fijo
    df_simulacion['TM_asignado'] = df_simulacion.groupby(['Pedido','Tipo'])['TM_fijo'].transform('sum')
    # Hay una diferencia entre cierre y atendido, mejor basarse solo en atendido # para 24-1 seria solo cierre
    df_simulacion['TM_pendiente'] = np.where(df_simulacion['Tipo']=='Split', df_simulacion['TM_cierre'] - df_simulacion['TM_asignado'], 0)
    df_simulacion['TM_repartido'] = 0
    df_simulacion['TM_fijo'] = df_simulacion['TM_fijo'].fillna(0)
    df_simulacion['TM_pendiente_original'] = df_simulacion['TM_pendiente']
    
    # Eliminar casos en los que TM Total y Asignado sea 0
    #df_simulacion = df_simulacion[(df_simulacion['TM_cierre']>0) & (df_simulacion['TM_atendido']>0)].reset_index(drop=True)
    # Eliminar casos en los que TM Total sea 0
    df_simulacion = df_simulacion[(df_simulacion['TM_cierre']>0)].reset_index(drop=True)
  
    # Calcular recomendacion fecha embarque (1 mes y 1 semana antes de la fecha programada) 
    df_simulacion['Fecha_recomendada'] = df_simulacion['Fecha_tope'] - datetime.timedelta(days=35) 

    # Calcular maxima fecha embarque (1 mes antes de la fecha programada) 
    df_simulacion['Fecha_max_rec'] = df_simulacion['Fecha_recomendada'] + datetime.timedelta(days=7)

    # Ordenar de acuerdo a fecha recomendada
    df_simulacion = df_simulacion.sort_values(['Fecha_recomendada','Tipo_split','Valor_inicial_TM','Precio_unitario'], ascending=[True,True,False,False])

    # Generar año semana de fechas limites y maximas
    # Fecha recomendada
    df_simulacion['YEAR_FECHA_MIN'] = df_simulacion['Fecha_recomendada'].dt.year
    df_simulacion['MONTH_FECHA_MIN'] = df_simulacion['Fecha_recomendada'].dt.month
    df_simulacion['WEEK_FECHA_MIN'] = df_simulacion['Fecha_recomendada'].dt.isocalendar().week
    df_simulacion['WEEK_FECHA_MIN'] = np.where((df_simulacion['WEEK_FECHA_MIN']==52) & (df_simulacion['YEAR_FECHA_MIN']==2022), 1, df_simulacion['WEEK_FECHA_MIN'])
    df_simulacion['COD_FECHA_MIN'] = np.where(df_simulacion['WEEK_FECHA_MIN']<10,
                                             df_simulacion['YEAR_FECHA_MIN'].map(str) + '0' + df_simulacion['WEEK_FECHA_MIN'].map(str), 
                                             df_simulacion['YEAR_FECHA_MIN'].map(str) + df_simulacion['WEEK_FECHA_MIN'].map(str))
    df_simulacion['COD_FECHA_MIN'] = df_simulacion['COD_FECHA_MIN'].map(int)

    # Fecha max
    df_simulacion['YEAR_FECHA_MAX'] = df_simulacion['Fecha_max_rec'].dt.year
    df_simulacion['MONTH_FECHA_MAX'] = df_simulacion['Fecha_max_rec'].dt.month
    df_simulacion['WEEK_FECHA_MAX'] = df_simulacion['Fecha_max_rec'].dt.isocalendar().week
    df_simulacion['WEEK_FECHA_MAX'] = np.where((df_simulacion['WEEK_FECHA_MAX']==52) & (df_simulacion['YEAR_FECHA_MAX']==2022), 1, df_simulacion['WEEK_FECHA_MAX'])
    df_simulacion['COD_FECHA_MAX'] = np.where(df_simulacion['WEEK_FECHA_MAX']<10,
                                             df_simulacion['YEAR_FECHA_MAX'].map(str) + '0' + df_simulacion['WEEK_FECHA_MAX'].map(str), 
                                             df_simulacion['YEAR_FECHA_MAX'].map(str) + df_simulacion['WEEK_FECHA_MAX'].map(str))
    df_simulacion['COD_FECHA_MAX'] = df_simulacion['COD_FECHA_MAX'].map(int)

    return df_simulacion

def lp_dist_cierre_min_500(df_precios,df_oferta,df_limites_pedidos):    

    df_precios = df_precios.astype({col: 'float' for col in df_precios.select_dtypes(include=['number']).columns})
    df_oferta = df_oferta.astype({col: 'float' for col in df_oferta.select_dtypes(include=['number']).columns})
    df_limites_pedidos = df_limites_pedidos.astype({col: 'float' for col in df_limites_pedidos.select_dtypes(include=['number']).columns})
      
    print('INICIO MODELO')
    # Crear el problema de optimización
    opt_weekly_model = LpProblem("Asignacion_Pedidos", LpMaximize)
    # Definir las variables de decisión como combinación de Pedido y CALIDAD
    x_vars = LpVariable.dicts("x", 
                              ((row['Pedido'], row['CALIDAD']) for idx, row in df_precios.iterrows()), 
                                lowBound=0, cat='Continuous')

    # Función objetivo: Maximizar los ingresos
    opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] * row['Precio_unitario'] 
                   for idx, row in df_precios.iterrows())

    # Restricción: No exceder el stock disponible por CALIDAD
    for calidad in df_oferta['CALIDAD'].unique():
        stock_disponible = df_oferta.loc[df_oferta['CALIDAD'] == calidad, 'STOCK_INICIAL'].values[0]
        opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                           for idx, row in df_precios.iterrows() if row['CALIDAD'] == calidad) <= stock_disponible

    # Restricción adicional: Cada pedido por CALIDAD debe ser al menos 500 toneladas si el pedido supera 500
    for idx, row in df_precios.iterrows():
        pedido, calidad = row['Pedido'], row['CALIDAD']
        if row['TM_pendiente'] > 500:  # Aplica la restricción solo si TM_pendiente > 500
            opt_weekly_model += x_vars[(pedido, calidad)] >= 500, f"Min_500_{pedido}_{calidad}"

    # Restricción adicional: La suma de todas las x_vars debe ser menor o igual a la suma total de STOCK_INICIAL
    suma_stock_inicial = df_oferta['STOCK_INICIAL'].sum()
    opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                           for idx, row in df_precios.iterrows()) <= suma_stock_inicial

    # Restricción: La suma de x_vars para cada pedido debe ser menor o igual a Tm_pendiente
    for pedido in df_limites_pedidos['Pedido'].unique():
        tm_pendiente = df_limites_pedidos.loc[df_limites_pedidos['Pedido'] == pedido, 'TM_pendiente'].values[0]
        opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                   for idx, row in df_precios.iterrows() if row['Pedido'] == pedido) <= tm_pendiente

    # Resolver el modelo
    opt_weekly_model.solve()

    # Obtener los resultados
    #print(opt_weekly_model)
    # Mostrar el estado de la solución
    lp_status=LpStatus[opt_weekly_model.status]
    print(f"Status: {LpStatus[opt_weekly_model.status]}")

    # Crear un DataFrame con los valores asignados
    resultados = []
    for idx, row in df_precios.iterrows():
        pedido, calidad = row['Pedido'], row['CALIDAD']
        cantidad_asignada = x_vars[(pedido, calidad)].varValue
        resultados.append({'Cierre': pedido, 'Calidad': calidad, 'TM': cantidad_asignada})

    return lp_status,resultados

def lp_dist_cierre(df_precios,df_oferta,df_limites_pedidos):   
    # Convertir todas las columnas numéricas a float
    df_precios = df_precios.astype({col: 'float' for col in df_precios.select_dtypes(include=['number']).columns})
    df_oferta = df_oferta.astype({col: 'float' for col in df_oferta.select_dtypes(include=['number']).columns})
    df_limites_pedidos = df_limites_pedidos.astype({col: 'float' for col in df_limites_pedidos.select_dtypes(include=['number']).columns})
      
    print('INICIO MODELO')
    # Crear el problema de optimización
    opt_weekly_model = LpProblem("Asignacion_Pedidos", LpMaximize)
    # Definir las variables de decisión como combinación de Pedido y CALIDAD
    x_vars = LpVariable.dicts("x", 
                              ((row['Pedido'], row['CALIDAD']) for idx, row in df_precios.iterrows()), 
                                lowBound=0, cat='Continuous')

    # Función objetivo: Maximizar los ingresos
    opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] * row['Precio_unitario'] 
                   for idx, row in df_precios.iterrows())

    # Restricción: No exceder el stock disponible por CALIDAD
    for calidad in df_oferta['CALIDAD'].unique():
        stock_disponible = df_oferta.loc[df_oferta['CALIDAD'] == calidad, 'STOCK_INICIAL'].values[0]
        opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                           for idx, row in df_precios.iterrows() if row['CALIDAD'] == calidad) <= stock_disponible

    # Restricción adicional: La suma de todas las x_vars debe ser menor o igual a la suma total de STOCK_INICIAL
    suma_stock_inicial = df_oferta['STOCK_INICIAL'].sum()
    opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                           for idx, row in df_precios.iterrows()) <= suma_stock_inicial

    # Restricción: La suma de x_vars para cada pedido debe ser menor o igual a Tm_pendiente
    for pedido in df_limites_pedidos['Pedido'].unique():
        tm_pendiente = df_limites_pedidos.loc[df_limites_pedidos['Pedido'] == pedido, 'TM_pendiente'].values[0]
        opt_weekly_model += lpSum(x_vars[(row['Pedido'], row['CALIDAD'])] 
                   for idx, row in df_precios.iterrows() if row['Pedido'] == pedido) <= tm_pendiente

    # Resolver el modelo
    opt_weekly_model.solve()

    # Obtener los resultados
    #print(opt_weekly_model)
    # Mostrar el estado de la solución
    lp_status=LpStatus[opt_weekly_model.status]
    print(f"Status: {LpStatus[opt_weekly_model.status]}")

    # Crear un DataFrame con los valores asignados
    resultados = []
    for idx, row in df_precios.iterrows():
        pedido, calidad = row['Pedido'], row['CALIDAD']
        cantidad_asignada = x_vars[(pedido, calidad)].varValue
        resultados.append({'Cierre': pedido, 'Calidad': calidad, 'TM': cantidad_asignada})

    return lp_status,resultados

def tratamiento_cierres_solucion(df):
    """
    Modifica automáticamente el carácter en la posición 7 de los valores en una columna, reemplazándolo por '0'.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna a modificar.

    Returns:
        pd.DataFrame: DataFrame con los valores corregidos en la columna especificada.
    """
    df['Cierre'] = df['Cierre'].apply(lambda x: x[:7] + '0' + x[8:] if len(x) > 7 else x)
    return df

# =============================================================================
# Logica de modelo distribución de stocks HP a contratos 
# =============================================================================
def ejecutar_distribucion_stock(idBatch):
    # Instancia de DatabaseManager
    db_manager = DatabaseManager()
    # =============================================================================
    # Lectura de Datos Cierres de Venta
    # =============================================================================
    # Obtener registros filtrados y ordenados
    
    db_cierres=db_manager.get_cierres_by_idbatch(idBatch)
    # Obtener registros filtrados y ordenados
    data_calidades=db_manager.get_all_ord_calidad()
    data_calidades_extra=db_manager.get_all_ord_calidad_extra()
    df_calidades_normal_extra = pd.concat([data_calidades, data_calidades_extra], ignore_index=True)
    ## Parametros
    temporada = db_manager.get_parametros_distribucion_stock()
    
    tipo_hp = ['ETOX','BHT']  
    # =============================================================================
    # Tratamiento de datos
    # =============================================================================

    # 1. Limpieza de calidad y tipo HP (ETOX, BHT)
    db_cierres_temp = limpieza_cierres_calidades(db_cierres, temporada)
    db_aux = db_cierres_temp[['N° Pedido', 'Producto', 'Destino', 'Calidad','Comprador']].copy()
    
    # 2. Limpieza de columnas y agregaciones de columnas
    #db_cierres_cleaned = limpieza_cierres_columnas(db_cierres_fijo)
    db_cierres_cleaned = limpieza_cierres_columnas(db_cierres_temp)
    
    # 3. Dividir cierres por observaciones mixtos fijos y open
    db_cierres_dividido = dividir_pedidos_fijo_open_altas_bajas(db_cierres_cleaned)

    # 4. Dividir cierres por observaciones de X% altas y X% bajas
    db_cierres_mixtos = dividir_pedidos_generales(db_cierres_dividido)

    # 5. Tratamiento de cierres final
    db_cierres_final = tratamiento_cierres_final(db_cierres_mixtos)
    
    df_calidades_etox = df_calidades_normal_extra.copy()
    df_calidades_etox['TIPO'] = 'ETOX'
    df_calidades_bht = df_calidades_normal_extra.copy()
    df_calidades_bht['TIPO'] = 'BHT'
    df_calidades_total = pd.concat([df_calidades_etox,df_calidades_bht], axis=0)

    db_cierres_final['COD_UNICO'] = db_cierres_final['Pedido'] + '-' + db_cierres_final['Calidad'] + '-' + db_cierres_final['Tipo']
    # Aca si es importante no separar por fijo y split, pq el momento de asignacion es crucial
    df_cierres_full = db_cierres_final[['Pedido','Tipo']].drop_duplicates()
    df_full_plantilla = df_calidades_total.merge(df_cierres_full, how='cross')
    df_full_plantilla = df_full_plantilla[df_full_plantilla['TIPO']==df_full_plantilla['Tipo']].reset_index(drop=True)
    del df_full_plantilla['Tipo']
    df_semanal = df_full_plantilla.merge(db_cierres_final, how='left', left_on=['CALIDAD','Pedido','TIPO'], right_on=['Calidad','Pedido','Tipo'])

    ## Completar na's con ceros o split
    df_semanal['TM_split_cierre'] = df_semanal['TM_split_cierre'].fillna(0)
    df_semanal['TM_split_atendido'] = df_semanal['TM_split_atendido'].fillna(0)
    df_semanal['Precio_unitario'] = df_semanal['Precio_unitario'].fillna(0)
    df_semanal['TM_fijo'] = df_semanal['TM_fijo'].fillna(0)
    df_semanal['Valor_calidad_inicial'] = df_semanal['Valor_calidad_inicial'].fillna(0)
    df_semanal['TM_asignado'] = df_semanal['TM_asignado'].fillna(0)
    df_semanal['TM_repartido'] = df_semanal['TM_repartido'].fillna(0)
    df_semanal['Tipo_split'] = df_semanal['Tipo_split'].fillna('Split')
    del df_semanal['Calidad']
    del df_semanal['Tipo']

    ##Agregar TM_cierre_total para los pedidos que no tenian Total en otras calidades
    df_cierre_aux = db_cierres_final[['Pedido','Tipo','Fecha_recomendada', 'Fecha_max_rec','TM_cierre_total']].drop_duplicates().reset_index(drop=True)
    ## Match con df_semanal
    df_semanal = df_semanal.merge(df_cierre_aux, how='left', left_on=['Pedido','TIPO'], right_on=['Pedido','Tipo'])
    del df_semanal['TM_cierre_total_x']
    del df_semanal['Fecha_recomendada_x']
    del df_semanal['Fecha_max_rec_x']
    del df_semanal['Tipo']
    df_semanal.rename(columns={'Fecha_recomendada_y':'Fecha_recomendada','Fecha_max_rec_y':'Fecha_max_rec','TM_cierre_total_y':'TM_cierre_total'}, inplace=True)

    # Formacion del dataframe para el modelo de optimizacion
    cols_semanal = ['Pedido','TIPO','Tipo_split','CALIDAD','ORDEN','TM_cierre_total','Precio_unitario','Fecha_recomendada', 'Fecha_max_rec']
    df_opt_semanal = df_semanal[cols_semanal].copy()

    df_aux_tm = df_semanal[['Pedido','TIPO','Tipo_split','TM_cierre']].drop_duplicates()
    df_aux_tm = df_aux_tm[(df_aux_tm['TM_cierre'].notnull()) & (df_aux_tm['Tipo_split']=='Split')].reset_index(drop=True)
    df_aux_fijo = df_semanal[['Pedido','CALIDAD','TIPO','Tipo_split','TM_fijo']].drop_duplicates()
    df_aux_fijo = df_aux_fijo[df_aux_fijo['Tipo_split']=='Fija'].reset_index(drop=True)
    df_opt_semanal = df_opt_semanal.merge(df_aux_tm, how='left', on=['Pedido','TIPO','Tipo_split'])
    df_opt_semanal['TM_cierre'] = df_opt_semanal['TM_cierre'].fillna(0)
    df_opt_semanal = df_opt_semanal.merge(df_aux_fijo, how='left', on=['Pedido','CALIDAD','TIPO','Tipo_split'])
    df_opt_semanal['TM_fijo'] = df_opt_semanal['TM_fijo'].fillna(0)
    df_opt_semanal['TM_minimo'] = df_opt_semanal['TM_fijo']
    df_opt_semanal['TM_maximo'] = df_opt_semanal['TM_cierre_total']
    df_aux_fechas = df_semanal[['Pedido','COD_FECHA_MIN','COD_FECHA_MAX']].drop_duplicates()
    df_aux_fechas = df_aux_fechas[df_aux_fechas['COD_FECHA_MIN'].notnull()].reset_index(drop=True)
    df_opt_semanal = df_opt_semanal.merge(df_aux_fechas, how='left', on='Pedido')
    df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_cierre_total']
    df_opt_semanal['CALIDAD_NUM'] = df_opt_semanal['ORDEN'].map(str) + '. ' + df_opt_semanal['CALIDAD']
    # Adicionar columna para stock de seguridad
    df_opt_semanal['COD_FECHA_SS'] = df_opt_semanal['COD_FECHA_MAX'] + 1
    #df_opt_semanal_bkp = df_opt_semanal.copy()

    # =============================================================================
    # Lectura de Datos Stocks por Calidad
    # =============================================================================

    ## Lectura de datos stocks
    # db_stocks=pd.read_excel(r'C:\Users\admin\Desktop\asignaciones\Docs_Distribucion_StockHP_Contratos\Docs_Distribucion_StockHP_Contratos\Input_1\Stock_Calidades.xlsx',sheet_name='Hoja1')
    db_stocks=db_manager.get_last_tonelaje_por_calidad()
    ## Cambiar nombre
    db_stocks = db_stocks.rename(columns={'Ton Upgrade':'STOCK_INICIAL','Fecha':'Fecha_stock'})
    ##Agregar variable de manera provisional

    # Obtener la fecha más reciente
    last_date = db_stocks['Fecha_stock'].max()  
    # Filtrar fecha para resultados finales 
    fecha_stock = last_date.strftime('%Y-%m-%d') 
    # Mascara para filtro de fecha stocks
    mask_last_date = db_stocks['Fecha_stock'] == last_date 

    ## Crear lista para guardar solucion final
    list_sol_valida = list()

    # =============================================================================
    # Logica de modelo distribución de stocks HP a contratos 
    # =============================================================================

    for id_tipo in tipo_hp:
        print(f"Solucion para tipo: {id_tipo}")
        
        ## Filtrar stock para tipo de HP
        db_stocks['TIPO'] = id_tipo 
        # Mascara para filtro de tipo de stocks
        mask_stocks_tipo = db_stocks['TIPO'] == id_tipo
        db_stocks_tipo = db_stocks[mask_stocks_tipo & mask_last_date]
        mask_calidades_tipo = df_calidades_total['TIPO'] == id_tipo 
        db_calidades_tipo = df_calidades_total[mask_calidades_tipo]
        
        ## Obtener stock semanal
        db_stocks_step = db_calidades_tipo.merge(db_stocks_tipo, how='left', on = ['CALIDAD','TIPO'])
        
        db_stocks_step['Fecha_stock'] = db_stocks_step['Fecha_stock'].fillna(last_date)
        db_stocks_step['STOCK_INICIAL'] = db_stocks_step['STOCK_INICIAL'].fillna(0)
        db_stocks_step['CALIDAD_NUM'] = db_stocks_step['ORDEN'].map(str) + '. ' + db_stocks_step['CALIDAD']
        
        ## Obtener la fecha actual
        fecha_actual = pd.Timestamp.now()
        #fecha_actual = pd.Timestamp.now() - pd.Timedelta(days=7)
        fecha_ejec = fecha_actual.strftime('%Y-%m-%d')
        
        ## Calcular la diferencia en días y convertirla a semanas
        df_opt_semanal['Semanas_Diferencia'] = (df_opt_semanal['Fecha_max_rec'] - fecha_actual).dt.days // 7
        
        ## Filtrar los pedidos que podrian atenderse en la ejecucion
        mask_semana_ant_min = df_opt_semanal['Semanas_Diferencia'] >= -5
        mask_semana_ant_max = df_opt_semanal['Semanas_Diferencia'] <= 7
        mask_semana_tipo = df_opt_semanal['TIPO'] == id_tipo
        mask_pendiente = df_opt_semanal['TM_pendiente'] > 0 
        
        df_pedidos_step = df_opt_semanal[mask_semana_ant_min & mask_semana_ant_max & mask_semana_tipo & mask_pendiente].reset_index(drop=True)

        df_pedidos_step['TM_asignado'] = 0
        
        # Validar % de TM fijo alcanzado respecto a stock actual
        mask_fijo = df_opt_semanal['Tipo_split']=='Fija'
        df_tm_fijo = df_opt_semanal[mask_fijo & mask_semana_tipo & mask_pendiente]
        df_tm_fijo = df_tm_fijo.groupby(['CALIDAD_NUM','TIPO']).agg(TM_min=('TM_fijo','sum'))
        df_tm_fijo.reset_index(inplace=True)
        
        # Armar df stock actual (pre asignaciones) vs requerimiento tm fijo total
        df_minimo_stock = db_stocks_step.merge(df_tm_fijo, how='left', left_on=['CALIDAD_NUM','TIPO'], right_on=['CALIDAD_NUM','TIPO'])
        df_minimo_stock['TM_min'] = df_minimo_stock['TM_min'].fillna(0)
        df_minimo_stock = df_minimo_stock[df_minimo_stock['STOCK_INICIAL']>=0]

        #df_minimo_stock = df_minimo_stock[(df_minimo_stock['TIPO']==tipo) & (df_minimo_stock['TM_min']>0)]
        df_fijo_semanal = df_opt_semanal[mask_semana_ant_min & mask_semana_ant_max & mask_fijo & mask_semana_tipo & mask_pendiente]
        df_fijo_semanal = df_fijo_semanal.groupby(['CALIDAD_NUM']).agg(TM_fijo_req=('TM_fijo','sum'))
        df_fijo_semanal.reset_index(inplace=True)
        df_minimo_stock = df_minimo_stock.merge(df_fijo_semanal, how='left', on=['CALIDAD_NUM'])
        
        ###AGREGUE POR EROR EN VALORES NAN DE TM_fijo_req
        #if semana!=max_mes_prod:
        df_minimo_stock['TM_fijo_req'] = df_minimo_stock['TM_fijo_req'].fillna(0)
        df_minimo_stock['fijo_semanal'] = np.where((df_minimo_stock['TM_fijo_req']>df_minimo_stock['STOCK_INICIAL'])| (df_minimo_stock['TM_fijo_req'].isna()), 1, 0)
        # else:
        #     df_minimo_stock['fijo_semanal'] = np.where(df_minimo_stock['TM_min']>df_minimo_stock['STOCK_FINAL'], 1, 0)
        
        deficiencia_fijo = df_minimo_stock['fijo_semanal'].sum()
        
        ## SEGUNDA VERSION DE OPTIMIZACION
        if (df_minimo_stock['fijo_semanal'] == 0).any(): 
            print('Distribuir Contratos Fijos')
            ##Filtrar las calidades fijas para este semana 
            calidad_fija = df_minimo_stock.loc[(df_minimo_stock['fijo_semanal']==0),'CALIDAD'].tolist()
            
            ##Genero resultado para asignacion de calidades fijas:
            mask_step_fijo = df_pedidos_step['Tipo_split']=='Fija'
            df_asignado_fijo = df_pedidos_step[mask_step_fijo & (df_pedidos_step['CALIDAD'].isin(calidad_fija))]
            df_asignado_fijo.loc[:,'TM_asignado']=df_asignado_fijo['TM_fijo']
            cols_cierre_fijo = ['Pedido','CALIDAD','Precio_unitario','TM_asignado']
            df_asignado_fijo = df_asignado_fijo[cols_cierre_fijo]
            df_asignado_fijo = df_asignado_fijo.rename(columns={'CALIDAD': 'Calidad','TM_asignado':'TM'})
            df_asignado_fijo['Tipo']=id_tipo
            #df_asignado_fijo['Semana']=semana
            df_asignado_fijo['Tipo_split']='Fija'
            df_asignado_fijo['Cierre']=df_asignado_fijo['Pedido']
            df_asignado_fijo['Fecha_ejecucion'] = fecha_ejec
            df_asignado_fijo['Fecha_stock'] = fecha_stock
            cols_order = ['Cierre','Calidad','Tipo','Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']#,'Semana']
            df_asignado_fijo = df_asignado_fijo[cols_order]

            ##Guardar stock original
            cols_stock_orig=['CALIDAD','TIPO','STOCK_INICIAL']
            df_stock_step_orig = df_minimo_stock[cols_stock_orig].copy()

            print('Completar resto de Contratos Fijos')
            ##AGREGO POR ERROR EN EL TIPO DE DATO DECIMAL Y FLOAT
            # Convertir columnas relevantes a float
            df_minimo_stock['STOCK_INICIAL'] = df_minimo_stock['STOCK_INICIAL'].astype(float)
            df_minimo_stock['TM_fijo_req'] = df_minimo_stock['TM_fijo_req'].fillna(0).astype(float)

            ##Restar stock por TM fijo 
            df_minimo_stock['STOCK_INICIAL_2'] = np.where(
                df_minimo_stock['fijo_semanal'] == 0,
                df_minimo_stock['STOCK_INICIAL'] - df_minimo_stock['TM_fijo_req'],
                df_minimo_stock['STOCK_INICIAL']
            )
            ##Guardar stock con resta de TM Fijo
            cols_stock=['CALIDAD','TIPO','STOCK_INICIAL_2']
            df_stock_step = df_minimo_stock[cols_stock]
            ##Cambiar nombre de columnas
            df_stock_step.columns = ['CALIDAD','Tipo','STOCK_INICIAL']
            df_oferta=df_stock_step.copy()
            #print(df_oferta)

            ##Restar pedidos asignados para TM fijo
            df_pedidos_step = pd.merge(df_pedidos_step, df_asignado_fijo[['Cierre', 'Calidad', 'Tipo','TM']],  how='left',left_on=['Pedido', 'CALIDAD', 'TIPO'], right_on=['Cierre', 'Calidad', 'Tipo'])
            df_pedidos_step['TM'] = df_pedidos_step['TM'].fillna(0)
            #df_pedidos_step = df_pedidos_step.rename(columns={'Tipo_split_x':'Tipo_split','Precio_unitario_x':'Precio_unitario'})
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'] - df_opt_semanal['TM']
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal.groupby(['Pedido','TIPO'])['TM_pendiente'].transform(min)
            df_pedidos_step['TM_asignado'] = df_pedidos_step.groupby(['Pedido','TIPO'])['TM'].transform('sum')
            df_pedidos_step['TM_asignado_calidad'] = df_pedidos_step.groupby(['Pedido','TIPO','CALIDAD'])['TM'].transform('sum')
            #df_pedidos_step['TM_pendiente'] = df_pedidos_step['TM_pendiente'] - df_pedidos_step['TM_asignado']
            df_pedidos_step['TM_pendiente'] = np.where(df_pedidos_step['Tipo_split']=='Fija',df_pedidos_step['TM_fijo'] - df_pedidos_step['TM_asignado_calidad'],df_pedidos_step['TM_pendiente'])
            df_pedidos_step['TM_fijo'] = df_pedidos_step['TM_fijo'] - df_pedidos_step['TM_asignado_calidad']
            del df_pedidos_step['Cierre']
            del df_pedidos_step['Calidad']
            del df_pedidos_step['Tipo']
            del df_pedidos_step['TM']
            del df_pedidos_step['TM_asignado_calidad'] 
            del df_pedidos_step['TM_asignado'] 

            ##Filtrar cierres fijo
            mask_step_fijo = df_pedidos_step['Tipo_split']=='Fija'
            df_pendiente_fijo = df_pedidos_step[mask_step_fijo]
            ##Filtrar solo columnas para cierres
            cols_cierre = ['Pedido','CALIDAD','Precio_unitario','TM_pendiente']
            df_pendiente_fijo = df_pendiente_fijo[cols_cierre]
            df_pendiente_fijo = df_pendiente_fijo[(df_pendiente_fijo['Precio_unitario']>0) & (df_pendiente_fijo['TM_pendiente']>0)]
            #df_pendiente_fijo = df_pendiente_fijo.rename(columns={'CALIDAD_NUM': 'CALIDAD'})
            df_precios = df_pendiente_fijo.copy()

            ##Generar limite de pedidos
            df_limites_pedidos = df_precios.groupby(["Pedido"]).agg({"TM_pendiente":"mean"}).reset_index()

            ##Asignar solo los contratos fijos que esten disponibles
            if df_precios.empty:
                print('No hay Contratos Fijos')
                cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM', 'Precio_unitario', 'Semana']
                # Crear un DataFrame vacío con las columnas
                df_sol_opt_weekly_fijo = pd.DataFrame(columns=cols_order)
                df_sol_valida=pd.concat([df_asignado_fijo,df_sol_opt_weekly_fijo]).reset_index(drop=True)

            else: 
                print('Optimizacion Contratos Fijos con min 500')

                status_min_500,resultados = lp_dist_cierre_min_500(df_precios,df_oferta,df_limites_pedidos)
                
                
                if status_min_500=='Infeasible':
                    print('Optimizacion Contratos Fijos')
                    # Convertir columnas numericas a float en df_precios
                    df_precios['Precio_unitario'] = df_precios['Precio_unitario'].astype(float)
                    df_precios['TM_pendiente'] = df_precios['TM_pendiente'].astype(float)

                    # Convertir columnas numericas a float en df_oferta
                    df_oferta['STOCK_INICIAL'] = df_oferta['STOCK_INICIAL'].astype(float)

                    # Convertir columnas numericas a float en df_limites_pedidos
                    df_limites_pedidos['TM_pendiente'] = df_limites_pedidos['TM_pendiente'].astype(float)

                    status,resultados = lp_dist_cierre(df_precios,df_oferta,df_limites_pedidos)
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Fija'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_fijo = df_sol_opt_weekly[cols_order].copy() 
                else:    
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Fija'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_fijo = df_sol_opt_weekly[cols_order].copy()

                df_sol_valida=pd.concat([df_asignado_fijo,df_sol_opt_weekly_fijo]).reset_index(drop=True)

            print('Distribucion para Contratos Open')
            # Filtrar solucion que se atendio
            mask_tm = df_sol_valida['TM']>0
            #df_sol_valida = df_sol_valida[mask_tm].reset_index(drop=True)
            list_cierres_opt = df_sol_valida.loc[mask_tm, 'Cierre'].unique().tolist()
            df_sol_final = df_sol_valida[df_sol_valida['Cierre'].isin(list_cierres_opt)].reset_index(drop=True)
            # Ajuste por si se asigna TM a calidades con precio negativo
            #df_sol_final['TM_original_opt'] = df_sol_final['TM']

            # Agrupar solucion por calidad para stock de calidades
            df_opt_asig = df_sol_final.groupby(['Calidad','Tipo']).agg(TM_opt=('TM','sum'))
            df_opt_asig.reset_index(inplace=True)
            # Actualizar stock
            df_minimo_stock = df_minimo_stock.merge(df_opt_asig, how='outer', left_on=['CALIDAD','TIPO'], right_on=['Calidad','Tipo'])
            df_minimo_stock['TM_opt'] = df_minimo_stock['TM_opt'].fillna(0)
            df_minimo_stock['STOCK_INICIAL_3'] = df_minimo_stock['STOCK_INICIAL'] - df_minimo_stock['TM_opt']
            df_minimo_stock['STOCK_INICIAL_3'] = np.where(df_minimo_stock['STOCK_INICIAL_3']<1e-02, 0, df_minimo_stock['STOCK_INICIAL_3'])
            del df_minimo_stock['TM_opt']
            del df_minimo_stock['Calidad']
            del df_minimo_stock['Tipo']

            ##Guardar stock con resta de TM Fijo
            cols_stock=['CALIDAD','TIPO','STOCK_INICIAL_3']
            df_stock_step = df_minimo_stock[cols_stock]
            ##Cambiar nombre de columnas
            df_stock_step.columns = ['CALIDAD','Tipo','STOCK_INICIAL']
            df_oferta=df_stock_step.copy()

            ##Restar pedidos asignados para TM Open
            df_pedidos_step = pd.merge(df_pedidos_step, df_sol_opt_weekly_fijo[['Cierre', 'Calidad', 'Tipo', 'TM']],  how='left',left_on=['Pedido', 'CALIDAD', 'TIPO'], right_on=['Cierre', 'Calidad', 'Tipo'])
            #df_pedidos_step = df_pedidos_step.rename(columns={'TM_y':'TM'})
            df_pedidos_step['TM'] = df_pedidos_step['TM'].fillna(0)
            df_pedidos_step['TM_asignado'] = df_pedidos_step.groupby(['Pedido','TIPO'])['TM'].transform('sum')
            df_pedidos_step['TM_asignado_calidad'] = df_pedidos_step.groupby(['Pedido','TIPO','CALIDAD'])['TM'].transform('sum')
            #df_pedidos_step['TM_pendiente'] = df_pedidos_step['TM_pendiente'] - df_pedidos_step['TM_asignado']
            df_pedidos_step['TM_pendiente'] = np.where(df_pedidos_step['Tipo_split']=='Fija',df_pedidos_step['TM_fijo'] - df_pedidos_step['TM_asignado_calidad'],df_pedidos_step['TM_pendiente'])
            df_pedidos_step['TM_fijo'] = df_pedidos_step['TM_fijo'] - df_pedidos_step['TM_asignado_calidad']
            del df_pedidos_step['Cierre']
            del df_pedidos_step['Calidad']
            del df_pedidos_step['Tipo']
            del df_pedidos_step['TM']
            del df_pedidos_step['TM_asignado_calidad'] 
            del df_pedidos_step['TM_asignado'] 

            ##Filtrar cierres Open
            mask_step_open = df_pedidos_step['Tipo_split']=='Split'
            df_pendiente_open = df_pedidos_step[mask_step_open]
            ##Filtrar solo columnas para cierres
            cols_cierre = ['Pedido','CALIDAD','Precio_unitario','TM_pendiente']
            df_pendiente_open = df_pendiente_open[cols_cierre]
            df_pendiente_open = df_pendiente_open[df_pendiente_open['Precio_unitario']>0]
            #df_pendiente_fijo = df_pendiente_fijo.rename(columns={'CALIDAD_NUM': 'CALIDAD'})
            df_precios = df_pendiente_open.copy()

            ##Generar limite de pedidos
            df_limites_pedidos = df_precios.groupby(["Pedido"]).agg({"TM_pendiente":"mean"}).reset_index()

            ##Asignar solo los contratos open que esten disponibles
            if df_precios.empty:
                print('No hay Contratos Open')                
                cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM', 'Precio_unitario', 'Semana']
                # Crear un DataFrame vacío con las columnas
                df_sol_opt_weekly_open = pd.DataFrame(columns=cols_order)
                df_sol_valida = pd.concat([df_asignado_fijo,df_sol_opt_weekly_fijo,df_sol_opt_weekly_open]).reset_index(drop=True)
                
            else: 
                print('Optimizacion para Contratos Open con min 500')
                # Convertir columnas numericas a float en df_precios
                df_precios['Precio_unitario'] = df_precios['Precio_unitario'].astype(float)
                df_precios['TM_pendiente'] = df_precios['TM_pendiente'].astype(float)

                # Convertir columnas numericas a float en df_oferta
                df_oferta['STOCK_INICIAL'] = df_oferta['STOCK_INICIAL'].astype(float)

                # Convertir columnas numericas a float en df_limites_pedidos
                df_limites_pedidos['TM_pendiente'] = df_limites_pedidos['TM_pendiente'].astype(float)
                status_min_500,resultados = lp_dist_cierre_min_500(df_precios,df_oferta,df_limites_pedidos)

                if status_min_500=='Infeasible':
                    print('Optimizacion Contratos Open')       
                    status,resultados = lp_dist_cierre(df_precios,df_oferta,df_limites_pedidos)
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Split'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_open = df_sol_opt_weekly[cols_order].copy() 
                else:    
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Split'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_open = df_sol_opt_weekly[cols_order].copy()

                ##Solucion valida para contratos fijos y open        
                df_sol_valida = pd.concat([df_asignado_fijo,df_sol_opt_weekly_fijo,df_sol_opt_weekly_open]).reset_index(drop=True)
            
            # Filtrar solucion que se atendio
            mask_tm = df_sol_valida['TM']>0
            #df_sol_valida = df_sol_valida[mask_tm].reset_index(drop=True)
            list_cierres_opt = df_sol_valida.loc[mask_tm, 'Cierre'].unique().tolist()
            df_sol_final = df_sol_valida[df_sol_valida['Cierre'].isin(list_cierres_opt)].reset_index(drop=True)
            # Ajuste por si se asigna TM a calidades con precio negativo
            #df_sol_final['TM_original_opt'] = df_sol_final['TM']

            # Agrupar solucion por calidad para stock de calidades
            df_opt_asig = df_sol_final.groupby(['Calidad','Tipo']).agg(TM_opt=('TM','sum'))
            df_opt_asig.reset_index(inplace=True)
            # Actualizar stock
            
            df_stock_step = df_stock_step_orig.merge(df_opt_asig, how='outer', left_on=['CALIDAD','TIPO'], right_on=['Calidad','Tipo'])

            df_stock_step['TM_opt'] = df_stock_step['TM_opt'].fillna(0)   
            ##AUMENTE         
            df_stock_step['STOCK_INICIAL'] = df_stock_step['STOCK_INICIAL'].astype(float)
            df_stock_step['TM_opt'] = df_stock_step['TM_opt'].astype(float)
            
            df_stock_step['STOCK_FINAL'] = df_stock_step['STOCK_INICIAL'] - df_stock_step['TM_opt']
            
            df_stock_step['STOCK_FINAL'] = np.where(df_stock_step['STOCK_FINAL']<1e-02, 0, df_stock_step['STOCK_FINAL'])
            del df_stock_step['TM_opt']
            del df_stock_step['Calidad']
            del df_stock_step['Tipo']
           
            # Devolver a variable opt original
            df_sol_valida = df_sol_final.copy()
            
            # df_sol_opt_weekly_tiempo = df_sol_valido.copy()
            cols_reducidas = ['Cierre', 'Calidad', 'Tipo', 'Precio_unitario','Fecha_ejecucion','Fecha_stock']#, 'TM_original_opt']
            df_sol_valida = df_sol_final.groupby(cols_reducidas).agg(TM=('TM','sum'))
            df_sol_valida.reset_index(inplace=True)
            cols_order = ['Cierre', 'Calidad', 'Tipo','Precio_unitario','TM','Fecha_ejecucion','Fecha_stock']#,'Semana']
            df_sol_valida = df_sol_valida[cols_order]
            
            #list_opt_weekly.append(df_sol_valida)
            print('Toneladas distribuidas:',df_sol_valida['TM'].sum())
            mask_tm = df_sol_valida['TM']>0
            print('Contratos Atentidos',len(df_sol_valida[mask_tm]['Cierre'].unique().tolist()),df_sol_valida[mask_tm]['Cierre'].unique().tolist())
            print('Contratos Sin atender:',len(df_sol_valida[~mask_tm]['Cierre'].unique().tolist()),df_sol_valida[~mask_tm]['Cierre'].unique().tolist())
            df_sol_valida = df_sol_valida[mask_tm].reset_index(drop=True)
            print('FIN SOLUCION')#,df_sol_opt_weekly) # Revisar solucion

            ## Guardar solucion por tipo ETOX o BHT
            list_sol_valida.append(df_sol_valida)

            # Actualizar pedidos atendidos
            cols_opt_act = ['Cierre', 'Calidad', 'Tipo', 'TM']
            #df_opt_semanal = df_opt_semanal.merge(df_sol_opt_weekly[cols_opt_act], how='left', left_on=['Pedido','CALIDAD_NUM','TIPO'], right_on=['Cierre','Calidad','Tipo'])
            ##Correccion uso de memoria eficiente
            print('ACTUALIZAR PEDIDOS ATENDIDOS')     
            df_opt_semanal = pd.merge(df_opt_semanal, df_sol_valida[cols_opt_act],  how='left',left_on=['Pedido', 'CALIDAD', 'TIPO'],
                                       right_on=['Cierre', 'Calidad', 'Tipo'])
            
            df_opt_semanal['TM'] = df_opt_semanal['TM'].fillna(0)
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'] - df_opt_semanal['TM']
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal.groupby(['Pedido','TIPO'])['TM_pendiente'].transform(min)
            
            
            
            df_opt_semanal['TM_asignado'] = df_opt_semanal.groupby(['Pedido','TIPO'])['TM'].transform('sum')

            df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'].astype(float)
            
            df_opt_semanal['TM_asignado'] = df_opt_semanal['TM_asignado'].astype(float)

            df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'] - df_opt_semanal['TM_asignado']
            
            # df_opt_semanal['TM_pendiente'] = np.where(df_opt_semanal['TM_pendiente']>0, df_opt_semanal['TM_pendiente'])
            del df_opt_semanal['Cierre']
            del df_opt_semanal['Calidad']
            del df_opt_semanal['Tipo']
            del df_opt_semanal['TM']   
            
        else:
            print('Distribucion para Contratos Open')
            ## Restar stock por utilizar el stock sin restar stock fijos 
            cols_stock=['CALIDAD','TIPO','STOCK_INICIAL']
            df_stock_step = df_minimo_stock[cols_stock]
            ## Para guardar stock original
            df_stock_step_orig = df_stock_step.copy()
            ## Para modelo
            df_oferta=df_stock_step.copy()
            #print(df_oferta)
            ##Filtrar cierres open 
            mask_step_open = df_pedidos_step['Tipo_split']=='Split'
            df_pendiente_open = df_pedidos_step[mask_step_open]

            ##Filtrar solo columnas para cierres
            cols_cierre = ['Pedido','CALIDAD','Precio_unitario','TM_pendiente']
            df_pendiente_open = df_pendiente_open[cols_cierre]
            df_pendiente_open = df_pendiente_open[(df_pendiente_open['Precio_unitario']>0) & (df_pendiente_open['TM_pendiente']>0)]
            df_pendiente_open = df_pendiente_open.rename(columns={'CALIDAD': 'CALIDAD'})
            df_precios = df_pendiente_open.copy()
            ##Generar limite de pedidos
            df_limites_pedidos = df_precios.groupby(["Pedido"]).agg({"TM_pendiente":"mean"}).reset_index()

            if df_precios.empty:
                print('No hay Contratos Open')
                # ELIMINAR?
                # Considerar que no hay asignacion, crear dataframe vacio, todos los pedidos se acumulan a la siguiente semana
                cols_opt = ['Pedido','CALIDAD','TIPO','Tipo_split','TM_asignado','Precio_unitario']             
                df_sol_opt_weekly = df_pedidos_step[cols_opt].copy()
                df_sol_opt_weekly.rename(columns={'TM_asignado':'TM','TIPO':'Tipo'}, inplace=True)
                df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                df_sol_opt_weekly['Fecha_stock'] = fecha_stock  
                cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                df_sol_opt_weekly.columns = cols_order
                df_sol_opt_weekly['TM'] = 0

                ##Solucion valida open
                df_sol_valida = df_sol_opt_weekly.copy()
                #columnas_para_comparar = ['Cierre', 'Calidad', 'Tipo']  # Lista de columnas para comparar duplicados (Otra opcion)
                #df_sol_opt_weekly = df_sol_opt_weekly.drop_duplicates(subset=columnas_para_comparar).reset_index(drop=True)
                # Resetear stock
                df_stock_step_orig['STOCK_FINAL'] = df_stock_step_orig['STOCK_INICIAL']
            else:
                print('Optimizacion para Contratos Open con min 500')
                # Convertir columnas numericas a float en df_precios
                df_precios['Precio_unitario'] = df_precios['Precio_unitario'].astype(float)
                df_precios['TM_pendiente'] = df_precios['TM_pendiente'].astype(float)

                # Convertir columnas numericas a float en df_oferta
                df_oferta['STOCK_INICIAL'] = df_oferta['STOCK_INICIAL'].astype(float)

                # Convertir columnas numericas a float en df_limites_pedidos
                df_limites_pedidos['TM_pendiente'] = df_limites_pedidos['TM_pendiente'].astype(float)
                status_min_500,resultados = lp_dist_cierre_min_500(df_precios,df_oferta,df_limites_pedidos)

                if status_min_500=='Infeasible':
                    print('Optimizacion Contratos Open')

                    status,resultados = lp_dist_cierre(df_precios,df_oferta,df_limites_pedidos)
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Split'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_open = df_sol_opt_weekly[cols_order].copy() 
                else:    
                    df_sol_opt_weekly = pd.DataFrame(resultados)
                    df_sol_opt_weekly['Tipo'] = id_tipo
                    df_sol_opt_weekly['Fecha_ejecucion'] = fecha_ejec
                    df_sol_opt_weekly['Fecha_stock'] = fecha_stock
                    df_sol_opt_weekly['Tipo_split'] = 'Split'
                    df_sol_opt_weekly = pd.merge(df_sol_opt_weekly, df_precios[["Pedido","CALIDAD","Precio_unitario"]], left_on=["Cierre","Calidad"], right_on=["Pedido","CALIDAD"], how="left")
                    del df_sol_opt_weekly['Pedido']
                    del df_sol_opt_weekly['CALIDAD']
                    cols_order = ['Cierre', 'Calidad', 'Tipo', 'Tipo_split', 'TM','Precio_unitario','Fecha_ejecucion','Fecha_stock']
                    df_sol_opt_weekly_open = df_sol_opt_weekly[cols_order].copy()

                ##Solucion valida open
                df_sol_valida = df_sol_opt_weekly_open.copy()
            
            # Filtrar solucion que se atendio
            mask_tm = df_sol_valida['TM']>0
            #df_sol_valida = df_sol_valida[mask_tm].reset_index(drop=True)
            list_cierres_opt = df_sol_valida.loc[mask_tm, 'Cierre'].unique().tolist()
            df_sol_final = df_sol_valida[df_sol_valida['Cierre'].isin(list_cierres_opt)].reset_index(drop=True)
            # Ajuste por si se asigna TM a calidades con precio negativo
            #df_sol_final['TM_original_opt'] = df_sol_final['TM']

            # Agrupar solucion por calidad para stock de calidades
            df_opt_asig = df_sol_final.groupby(['Calidad','Tipo']).agg(TM_opt=('TM','sum'))
            df_opt_asig.reset_index(inplace=True)
            # Actualizar stock
            df_stock_step = df_stock_step_orig.merge(df_opt_asig, how='outer', left_on=['CALIDAD','TIPO'], right_on=['Calidad','Tipo'])
            df_stock_step['TM_opt'] = df_stock_step['TM_opt'].fillna(0)
            df_stock_step['STOCK_FINAL'] = df_stock_step['STOCK_INICIAL'] - df_stock_step['TM_opt']
            df_stock_step['STOCK_FINAL'] = np.where(df_stock_step['STOCK_FINAL']<1e-02, 0, df_stock_step['STOCK_FINAL'])
            del df_stock_step['TM_opt']
            del df_stock_step['Calidad']
            del df_stock_step['Tipo']

            # Devolver a variable opt original
            df_sol_valida = df_sol_final.copy()
            # df_sol_opt_weekly_tiempo = df_sol_valido.copy()
            cols_reducidas = ['Cierre', 'Calidad', 'Tipo', 'Precio_unitario','Fecha_ejecucion','Fecha_stock']#, 'TM_original_opt']
            df_sol_valida = df_sol_final.groupby(cols_reducidas).agg(TM=('TM','sum'))
            df_sol_valida.reset_index(inplace=True)
            cols_order = ['Cierre', 'Calidad', 'Tipo','Precio_unitario','TM','Fecha_ejecucion','Fecha_stock']#,'Semana']
            df_sol_valida = df_sol_valida[cols_order]
            #list_opt_weekly.append(df_sol_valida)
            print('Toneladas distribuidas:',df_sol_valida['TM'].sum())
            mask_tm = df_sol_valida['TM']>0
            print('Contratos Atentidos',len(df_sol_valida[mask_tm]['Cierre'].unique().tolist()),df_sol_valida[mask_tm]['Cierre'].unique().tolist())
            print('Contratos Sin atender:',len(df_sol_valida[~mask_tm]['Cierre'].unique().tolist()),df_sol_valida[~mask_tm]['Cierre'].unique().tolist())
            df_sol_valida = df_sol_valida[mask_tm].reset_index(drop=True)
            print('FIN SOLUCION')#,df_sol_opt_weekly) # Revisar solucion

            ## Guardar solucion por tipo ETOX o BHT
            list_sol_valida.append(df_sol_valida)

            # Actualizar pedidos atendidos
            cols_opt_act = ['Cierre', 'Calidad', 'Tipo', 'TM']
            #df_opt_semanal = df_opt_semanal.merge(df_sol_opt_weekly[cols_opt_act], how='left', left_on=['Pedido','CALIDAD_NUM','TIPO'], right_on=['Cierre','Calidad','Tipo'])
            ##Correccion uso de memoria eficiente
            print('ACTUALIZAR PEDIDOS ATENDIDOS')
            
            df_opt_semanal = pd.merge(df_opt_semanal, df_sol_valida[cols_opt_act],  how='left',left_on=['Pedido', 'CALIDAD', 'TIPO'], right_on=['Cierre', 'Calidad', 'Tipo'])
            df_opt_semanal['TM'] = df_opt_semanal['TM'].fillna(0)
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'] - df_opt_semanal['TM']
            # df_opt_semanal['TM_pendiente'] = df_opt_semanal.groupby(['Pedido','TIPO'])['TM_pendiente'].transform(min)
            
            df_opt_semanal['TM_pendiente'] =df_opt_semanal['TM_pendiente'].astype(float)
            df_opt_semanal['TM_asignado'] = df_opt_semanal['TM_asignado'].astype(float)

            df_opt_semanal['TM_asignado'] = df_opt_semanal.groupby(['Pedido','TIPO'])['TM'].transform('sum')
            df_opt_semanal['TM_pendiente'] = df_opt_semanal['TM_pendiente'] - df_opt_semanal['TM_asignado']

            # df_opt_semanal['TM_pendiente'] = np.where(df_opt_semanal['TM_pendiente']>0, df_opt_semanal['TM_pendiente'])
            del df_opt_semanal['Cierre']
            del df_opt_semanal['Calidad']
            del df_opt_semanal['Tipo']
            del df_opt_semanal['TM']
            print('*********9***'*15)

    # =============================================================================
    # Tratamiento de resultados del modelo  
    # =============================================================================
    
    ##Guardar db de solucion por ejecucion 
    df_sol_final = pd.concat(list_sol_valida)
    df_sol_final = df_sol_final.reset_index(drop=True)    
    df_sol_final = tratamiento_cierres_solucion(df_sol_final)
    
    resultado = pd.merge(
        db_aux,
        df_sol_final, 
        left_on=['N° Pedido', 'Calidad'],  # Campos del primer DataFrame
        right_on=['Cierre', 'Calidad'],    # Campos del segundo DataFrame
        how='inner'                        # Tipo de join (inner, left, right, outer)
    )
    
    # =============================================================================
    # Exportar resultados del modelo  
    # =============================================================================
    resultado["idBatch"] = idBatch
    db_manager.save_tonelaje_por_calidad_contrato(resultado)
    #df_sol_final.to_excel('C:/Users/admin/Desktop/asignaciones/Documentacion Asignaciones/3. Carpeta/repuestastres/outputfinal_DistribucionStock.xlsx',index=False)