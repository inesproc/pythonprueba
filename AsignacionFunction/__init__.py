import logging
from Modelos.Asignacion import ejecutar_asignacion
from jwt import decode as jwt_decode
import azure.functions as func
from azure.functions import HttpResponse
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("HTTP trigger function for exec_asignacion processed a request.")
    try:
        # # Obtener el token del encabezado
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

        # Ejecutar la lógica de asignación
        ejecutar_asignacion()

        return HttpResponse(
            json.dumps({"message": "Asignación ejecutada correctamente."}), 
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error durante la ejecución de asignación: {str(e)}")
        return HttpResponse(
            json.dumps({"error": f"Hubo un error durante la ejecución de asignación: {str(e)}"}), 
            status_code=500,
            mimetype="application/json"
        )
