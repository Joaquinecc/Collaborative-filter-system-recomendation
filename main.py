import datetime
# BDay is business day, not birthday...
from pandas.tseries.offsets import BDay
import pandas as pd
from multiprocessing import  Pool,cpu_count
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
from dateutil.relativedelta import relativedelta
import datetime
import json
import numpy as np
import scipy
import time
# -----------------------------------------
# Variables globales
cut_year=1 # Cota inferior en años que indica hasta que fecha se hace la consulta. Ej:1. Se examina las ventas desde hace 1 año
columns=["n_cliente","cluster","codigo_barras","incidencia","unidades_compradas","unidades_prom_ticket","facturacion","costo","sucursales"]
BYTES_TO_MB_DIV = 0.000001
# Opening JSON file setting
f = open('settings.json') 
#Read setting params
settings_var = json.load(f)
topNProduct=settings_var['topNProduct']
topNSimilarity=settings_var['topNSimilarity']
# -----------------------------

def ranking_similarity_client(cx,index,N=10):
    #Por cada cliente se ordena de forma descente y se obtiene los top N clientes con mas similitud
    # @Params:
    #     N: uint : Numero Top . Ej top 10, so n=10
    #     cx: spare matrix
    #     index: index of each row
    # return:
    #     Array con el siguiente formato. 
    #     [["current client","neighbor","cosine distance","rank position"],..]
    #     Ej:
    #     [['0000250', '0000250', 1.0000000000000002, 0],
    #     ['0000250', '015877', 0.718819103532109, 1],
    #  ...]
    current_row=0
    data=[]
    temp=[]

    for i,j,v in zip(cx.row, cx.col, cx.data):
        if(i != current_row):
            #sort in descending order
            temp=sorted(temp, key=lambda d: d['cosine distance']) 
            
            #only rank the top N element
            size= N if N< len(temp) else len(temp) 
            [element.update({"rank":index+1}) for index,element in enumerate(temp[0:size]) ]
            for element in temp[0:size]:
                            #Current Client       Neighbor       cosine distance  rank position"
                data.append([index[current_row],element["Neighbor"],element["cosine distance"],element["rank"]])
            current_row=i
            temp=[]
                                        #cosine distance
        temp.append({"Neighbor":index[j],"cosine distance":1-v})
    return data
def rank_sku(args):
    # Chose the N product with the highest weigh
    # Params
    #     df_ratings: Panda dataframe : containig user ratings
    #     rows: Panda dataframe : containing neighbor cosine similarity
    #     N: unit:  Number of product to recomend
    # return: 
    #     Panda Dataframe, with columns 0 [n_client,producto,weigth,rank]
    
    df_ratings,rows,n_client,N = args
    print(n_client)

    #Neighbor ratings products 
    df_rating_neighbor= df_ratings.loc[rows["nearest neighbor - <RowID>"]].drop(["cluster"], axis=1)
    #drop producto columns fill with 0 (product that where not consumed)
    df_rating_neighbor=df_rating_neighbor.loc[:, (df_rating_neighbor != 0).any(axis=0)]


    #Drop products n_client already consume
    #df_rating_n_client= df_rating_neighbor.loc[[n_client]]
    df_rating_n_client= df_ratings.loc[[n_client]].drop(["cluster"], axis=1)
    df_rating_n_client=df_rating_n_client.loc[:, (df_rating_n_client != 0).any(axis=0)]
    product_already_consumed=df_rating_n_client.columns.tolist()
    df_rating_neighbor=df_rating_neighbor.drop(product_already_consumed, axis=1)


    #Calc ratingXdistance
    df_rating_neighbor["distance"]=rows["distance"].to_numpy().astype(float)
    for cols in df_rating_neighbor:
        if cols != "distance":
            df_rating_neighbor[cols]=df_rating_neighbor[cols]*df_rating_neighbor["distance"]
    
    #Calc weights
    df_temp=df_rating_neighbor.sum()
    if (df_temp["distance"] == 0):
        df_temp=pd.DataFrame(np.full(df_temp.shape, 0.1),index=df_temp.index,columns=["weigth"]).drop(["distance"],axis=0)
    else:
        df_temp=df_temp.div(df_temp["distance"], axis=0, fill_value=0)
        df_temp=df_temp.drop(["distance"],axis=0)
        df_temp=pd.DataFrame((df_temp),columns=["weigth"])
    
    #current client and product"
    df_temp["producto"]=df_temp.index
    df_temp["n_client"]=[n_client]*len(df_temp)
    #rank by weight
    df_temp=df_temp.sort_values(by=['weigth'],ascending=False)
    df_temp["rank"]=np.arange(1,df_temp.shape[0]+1)
    df_temp=df_temp[df_temp["rank"]<=N]    
    return df_temp


def make_query(postgreSQL_select_Query):
    # Consutla a la base de datos
    # @parmas
    #   postgreSQL_select_Query:string _> Consulta 
    #  return
    #    array of object
    try:
        f= open("settings.json","r")
        credential = json.loads(f.read())
        connection = psycopg2.connect(user=credential["user"],
                                      password=credential["password"],
                                      host=credential["host"],
                                      port=credential["port"],
                                      database=credential["database"])
        cursor = connection.cursor()
        cursor.execute(postgreSQL_select_Query)
        dataset = cursor.fetchall()
        print("Consulta terminada")

    except FileNotFoundError:
        print("Archivo credenciales.json not found")
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)
    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
        return dataset

def filtro_requisitos(df):
    #Filtras los clientes que no cuumplen los requisitos
    #Filtrar los clientes con compra de sku menor a 2
    # @parmas
    #   df:panda.DataFrame
    #  return
    #    panda.DataFrame
    temp=df.groupby("n_cliente").agg({"codigo_barras":"count"}).codigo_barras
    temp=temp[temp>2].index.tolist()
    return df[df.apply(lambda x: x['n_cliente'] in temp, axis=1)]

if __name__ == "__main__":
    start_time = time.time()
    print("Start..")
    #ULtimo dia laboral habil.
    last_labour_day=str((datetime.datetime.today() - BDay()).date())
    # Consultamos los productos validos consumido por el cliente y otros valores, como el monto total invertido, la incidencia del cliente,ect..
    print("Realizando consulta al postgresql")
    dataset=make_query(f"""
    select 
    trim(v."Codigo_Cliente") n_cliente,
    max(s."Cluster_Cuantitativo"),
    trim(v."Codigo_Producto")  codigo_barras,
    count(distinct v."Nro_Factura") as incidencia,
    sum(v."Cantidad_Venta") as unidades_compradas,
    avg(v."Cantidad_Venta") as unidades_prom_ticket,
    sum(v."Importe_Venta_Total") as facturacion ,
    sum(v."Costo_Total"*v."Cantidad_Venta") as costo,
    count(distinct v.sucursal_vendedor) as sucursales
    from public."Ventas" v 
    left join  public.segmentacion s  on trim(s.codigo_cliente) = trim(v."Codigo_Cliente") 
    where date_trunc('day', v."Fecha_Venta" ) <= to_date('{last_labour_day}', 'YYYY-MM-DD')
    and date_trunc('day', v."Fecha_Venta" ) > (to_date('{last_labour_day}', 'YYYY-MM-DD') - interval '[{cut_year} years')
    and cliente_juridico = 0 
    and trim(v."Codigo_Producto") not in (
    select va.articulocodigo from public.v_articulos va 
    WHERE va.articulocodigo in 
    ('CHASSIS',
    'FLETE',
    'INSTMI12',
    'MOTOR',
    'PELI',
    'SERTEC',
    'VALEBR',
    'VCARRIER12',
    'VCARRIER18',
    'VCARRIER24',
    'VCARRIER36',
    'VCOMB',
    'VCOMFEE12',
    'VCOMFEE18',
    'VCOMFEE24',
    'VMIDEA',
    'VMIDEA09',
    'VMIDEA12',
    'VMIDEA18',
    'VMIDEA24',
    'FLETE',
    'SERTEC',
    'VALEBR')
    OR va.articulocodigo IN 
    ( SELECT a.articulocodigo
    FROM articulo a
    WHERE a.obsequio = true)
    or va.familia in ('Flete', 'Vales', 'Regalos',null, 'Vinos', 'Kit De Instalación')

    ) 
    group by trim(v."Codigo_Cliente"),trim(v."Codigo_Producto")  

    """)
    #Utilizamos panda para un manejo de datos mas eficiente
    df=pd.DataFrame(dataset,columns=columns)
    #Cast to numeric value
    df[["facturacion","costo"]]=df[["facturacion","costo"]].astype(float)
    df[["incidencia","unidades_compradas","unidades_prom_ticket","sucursales"]]=df[["incidencia","unidades_compradas","unidades_prom_ticket","sucursales"]].astype(int)
    #Calcular el margen generado
    df["margen"]=df["facturacion"]-df["costo"]
    #Filtras los clientes que no cuumplen los requisitos
    df=filtro_requisitos(df)
    print("Calculando los ratings")
    #Calculamos los ratings
    df_ratings=pd.DataFrame()
    for name, group  in df.groupby("n_cliente"):
        # Nomalización min max
        for column in ['incidencia', 'unidades_compradas','unidades_prom_ticket', 'facturacion', 'costo', 'sucursales', 'margen']:
            group[column] = (group[column] - group[column].min()) / (group[column].max() - group[column].min()) 
            group.fillna(1,inplace=True)
        #Calcular el rating
        group["rating"]=group["incidencia"]+group["unidades_compradas"]+group["unidades_prom_ticket"]+group["sucursales"]+group["margen"]+1
        #Normalizar el rating
        # group["rating_adjusted"]=group["rating"]- group["rating"].mean()
        df_ratings=pd.concat([df_ratings,group])
    #Pivoteamos
    df_ratings=pd.pivot_table(df_ratings, values='rating', index=['n_cliente', 'cluster'],columns=['codigo_barras'], aggfunc=np.mean).fillna(0)
    df_ratings.reset_index(inplace=True)
    df_ratings.set_index("n_cliente",inplace=True)

    print("Calculando similitud de coseno entre los clietnes")
    #Calculando similitud de coseno entre los clietnes
    cluster_group = df_ratings.groupby("cluster")
    data=[]
    for cluster, rows in cluster_group:
        indexs=rows.index
        #Calc Cosine similarity
        sp_cl=scipy.sparse.csr_matrix(rows.drop(['cluster'], axis=1))
        sp_cl=scipy.sparse.coo_matrix(cosine_similarity(sp_cl,dense_output=False))
        #calc rank data
        sp_cl=ranking_similarity_client(sp_cl,indexs,N=topNSimilarity)
        if(len(data) == 0):
            data=sp_cl.copy()
        else: 
            data = np.concatenate((data, sp_cl))
    

    df_rank=pd.DataFrame(data,columns=["n_client","nearest neighbor - <RowID>","distance","nearest neighbor - index"])
    print(df_rank.head())

    clients_group = df_rank.groupby("n_client")
    df_ranking_sku=pd.DataFrame([],columns=["weigth","producto","n_client","rank"])
    print("Corriendo pool de procesos")
    with Pool(cpu_count()) as pool:
        df_ranking_sku=df_ranking_sku.append(pool.map( rank_sku, [(df_ratings,rows,n_client,topNProduct) for n_client, rows in clients_group]))
    #save
    df_ranking_sku.to_csv("recomendacion.csv",index=False)
    print("Finalizo..")
    print("--- %s seconds ---" % (time.time()-start_time))

  
