import logging
from Modelos.DistribucionStock import ejecutar_distribucion_stock
from jwt import decode as jwt_decode
import azure.functions as func
from azure.functions import HttpResponse
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger function processed a request.")
    try:
        # Obtener el token del encabezado
        # auth_header = req.headers.get("Authorization")
        # if not auth_header or not auth_header.startswith("Bearer "):
        #     return HttpResponse(
        #         json.dumps({"error": "Token no proporcionado o formato incorrecto"}), 
        #         status_code=401,
        #         mimetype="application/json"
        #     )

        # # Extraer y decodificar el token
        # token = auth_header.split(" ")[1]
        # try:
        #     jwt_payload = jwt_decode(token, options={"verify_signature": False})
        # except Exception as e:
        #     return HttpResponse(
        #         json.dumps({"error": f"Error al decodificar el token: {str(e)}"}), 
        #         status_code=400,
        #         mimetype="application/json"
        #     )
        
        # # Verificar el parámetro 'parameter'
        # if jwt_payload.get("parameter") != "true":
        #     return HttpResponse(
        #         json.dumps({"error": "No autorizado"}), 
        #         status_code=403,
        #         mimetype="application/json"
        #     )

        # Obtener el id_batch
        id_batch = req.params.get("id_batch")
        if not id_batch:
            return HttpResponse(
                json.dumps({"error": "id_batch is required"}), 
                status_code=400,
                mimetype="application/json"
            )

        # Ejecutar la lógica de distribución
        ejecutar_distribucion_stock(id_batch)

        return HttpResponse(
            json.dumps({"message": "Distribución Stock ejecutada correctamente."}), 
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error durante la ejecución: {str(e)}")
        return HttpResponse(
            json.dumps({"error": f"Hubo un error: {str(e)}"}), 
            status_code=500,
            mimetype="application/json"
        )